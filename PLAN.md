# 任务4：MiMo-V2.5-Pro 接入 ATOM SGLang Plugin —— 具体改动方案

> 前置结论见 `FINDINGS.md`（任务1/2/3）。本文把"哪些文件、哪个函数、改成什么"落到可执行粒度。
> 所有路径均为容器 `fanwu103-mimo-sgl-plugin` 内 `/app/ATOM`（editable）/`/app/sglang`（editable）。

## 0. 一句话结论

核心模型 `atom/models/mimo_v2.py` 已存在（原生 atom 已能跑），接入插件是 **glue（注册）+ 一段真活（backend 的 SWA+sink dense MHA 路径）**。
注册部分几乎零成本（照抄 `Qwen3MoeForCausalLM` 的空 spec）；**唯一的真难点 = decode 时的滑窗+attention sink 内核路径**，其余都是模板复制。

---

## 1. 两条接入路线的选择

插件下，无论模型侧走哪条注意力封装，**实际 attention 计算都会落到 ATOM 注入的 `full_attention_backend.py`**（`_register_custom_attention_to_sglang` 用 "aiter" 之名全局替换了 SGLang 的后端）。所以两条路线躲不开同一个内核缺口，区别只在 glue 多少：

| | 路线 A（推荐，最小 glue） | 路线 B（自定义 attention 桥，仿 minimax_m3） |
|---|---|---|
| 模型侧 | 保留 `mimo_v2.py` 现有 `Attention` → `AttentionForSGLang(RadixAttention)` | 新建 `SGLangATOMMiMoV2Attention(BaseAttention)` + construction context + install_adapters |
| 注意力 metadata | 继承 RadixAttention，天然透传（任务3已验证） | 需自建 `build_forward_metadata` |
| `sinks`/`sliding_window` 到 backend | 由 RadixAttention 适配器写到 `layer.sliding_window_size`/`layer.sinks`，backend 从 `layer` 读 | 走 `unified_attention_with_output_base`，仍进同一 backend |
| 代码量 | 少（spec 近乎空 + backend 改动 + 池绑定） | 多（多一套 wrapper/context/adapter，仅在需要 metadata 定制时才值） |
| 结论 | **默认选 A**。除非 metadata/池绑定证明必须定制，否则不引入 B 的额外封装 | 作为 A 不足时的回退 |

下面按 **路线 A** 展开。

---

## 2. 改动清单（按依赖顺序）

### 模块①　注册 arch → spec　`atom/plugin/sglang/runtime/model_arch.py`
- 在 `MODEL_TYPE_ADAPTER_ARCHES`（文件头 :17 附近）加 `"mimo_v2": "MiMoV2ForCausalLM"`。
- 在 `MODEL_ADAPTER_SPECS` dict 加一条。**起步先照抄 Qwen3Moe 的空 spec**，只在需要时补 hook：
  ```python
  "MiMoV2ForCausalLM": SGLangModelAdapterSpec(
      prepare_config=_prepare_mimo_v2_config,      # 见模块⑥（quant/hybrid 预处理）
      bind_cache_views=_bind_mimo_v2_cache_views,  # 见模块④（hybrid SWA 池）；先可为 None 验证 full-attn
  ),
  ```
- 把 `"MiMoV2ForCausalLM"` 加入底部 `MODEL_ARCH_SPECS` 的 key 元组（否则 `base_model_wrapper` 不会为它生成 EntryClass）。
- 成本：**低**。这一步做完，SGLang 就能解析该 arch 并生成插件 wrapper 类。

### 模块②　arch → 真模型类　`atom/plugin/register.py`
- `_ATOM_SUPPORTED_MODELS`（:24）加：
  ```python
  from atom.models.mimo_v2 import MiMoV2ForCausalLM
  ...
  "MiMoV2ForCausalLM": MiMoV2ForCausalLM,
  ```
- MTP draft（可选，模块⑧）：`atom.models.mimo_v2_mtp` 加进 `_ATOM_SUPPORTED_DRAFT_MODELS`。
- 成本：**低**。

### 模块③【真难点】　backend 的 SWA + sink dense MHA 路径　`atom/plugin/sglang/attention_backend/full_attention/full_attention_backend.py`
MiMo `head_dim=192`（≠256），**永远进不了 `_should_use_native_dense_mha`（:1754）**，因此：

- **Prefill/Extend（可接通）**：MiMo 落在 `_forward_extend_mha`（:1998），内核 `flash_attn_varlen_func` 已暴露 `window_size`/`sink_ptr`（当前写死 `(-1,-1,0)` / `None`，:2017-2018）。改为按层读取：
  ```python
  sw = getattr(layer, "sliding_window_size", None)
  window = (sw - 1, 0, 0) if sw and sw > 0 else (-1, -1, 0)   # 左窗口=sw-1，含自身
  sink = getattr(layer, "sinks", None)                        # attention_sink_bias 张量
  # ... window_size=window, sink_ptr=sink
  ```
  注意 MiMo 语义：`sliding_window=128` + `add_swa_attention_sink_bias` + `attention_value_scale=0.612`（value 缩放已在 `mimo_v2.py` 模型侧 `v = v * v_scale` 处理，backend 不重复）。

- **Decode（头号风险，非"改个参数"）**：MiMo decode 默认落到 `pa_fwd_asm` / `pa_persistent_fwd`（:2646-2690），**这两个内核调用根本没有 window/sink 形参**。只有 `ATOM_FORCE_ATTN_TRITON` 分支的 `run_pa_decode_gluon`（:2732）暴露 `sinks`/`sliding_window`（当前也写死 `None`/`-1`，:2760-2761）。两个选项：
  1. **（推荐先验证）把 SWA 层的 decode 强制走 gluon/triton PA**：为 SWA 层设置/分支到 `run_pa_decode_gluon` 并传真实 `sinks=layer.sinks, sliding_window=layer.sliding_window_size`。代价：可能牺牲 `pa_fwd_asm` 的性能；需确认 gluon PA 在 gfx950 上的 sink 数值正确性。
  2. 给 `pa_fwd_asm`/`pa_persistent_fwd` 增加 window+sink 支持 —— **内核级改动，成本高**，非必要不做。
- 别忘了 `_should_use_native_dense_mha` 里 `head_dim==256` 的限制意味着即便未来 head_dim 对上，也需确认 SWA 分支；MiMo 当前不受此影响。
- 成本：**高**（本方案的关键路径）。主要风险=正确性（SWA 因果掩码边界、sink 数值、fp8 descale 与 sink 的相互作用）。

#### 模块③附：v head_dim 非对称（128 vs 192）—— 插件已天然解决，勿破坏
- **背景**：MiMo 是 `qk_head_dim=192 / v_head_dim=128` 非对称；aiter PA 内核（`pa_decode_gluon`/`pa_fwd_asm`）要求 k、v head_dim 一致。这正是**原生 SGLang 只能用 triton、不能用 aiter 跑 MiMo** 的原因（SGLang 单一 head_dim 的 MHA 池 + 不做补齐）。
- **插件为何不受影响**：插件实例化的是 `atom.models.mimo_v2.MiMoV2ForCausalLM`，它在**模型 forward 内**把 v 零填充成对称 192（`mimo_v2.py` L306-310），attn 后再截回 128（L315-318）；且 `self.attn = Attention(head_dim=192)`（L277）**不传** v_head_dim → RadixAttention 适配器（`radix_attention.py` L76-82）回落 `_v_head_dim = head_dim = 192` → SGLang KV 池按**对称 192**分配 → aiter "k/v head_dim 相等" 前提满足。这段模型代码在插件下原样执行，无需额外编码。
- **要守住的**：模块⑥ `prepare_config`、模块④ hybrid 子池构建时，**不要**用 `config.v_head_dim(128)`/`swa_v_head_dim` 去覆盖 attn 层或 KV 池的几何——一律按层报告的 192 走。任何路径都别给 MiMo 启用融合 rope-cache 写（`fused_qk_rope_reshape_and_cache` 不支持 192，见 L300 注释）。
- **代价**：v 以 192 存储，约 33% 零填充显存浪费（与原生 atom 一致，既定权衡）。
- **净效果**：此项把"k/v head_dim 不等"这个 aiter 拦路块在插件里消除；模块③ decode 剩下的真活只有 SWA 窗口 + sink 接线。

### 模块④　Hybrid SWA KV 池绑定　`atom/plugin/sglang/runtime/model_arch.py` + 新 bridge 文件
- SGLang 侧已完备（任务3验证：`HybridReqToTokenPool`、`HybridLinearKVPool`（带 `full_attention_layer_ids`）、`swa_memory_pool`、`is_hybrid_swa`）。
- 需要一个 `_bind_mimo_v2_cache_views(model, runtime)`（模板：`_bind_deepseek_v4_cache_views` :295 / `_bind_minimax_m3_cache_views` :343），让 SWA 层读窗口子池、full 层读全量池，并把 `full_attention_layer_id_mapping` 喂给 backend（backend :116 已在读）。
- **验证顺序建议**：第一版先不绑 hybrid 池（`bind_cache_views=None`），全部层用全量池 + 模块③的 window mask 跑通正确性；正确后再引入 hybrid 池省显存。
- **注意**：窗口子池同样按**对称 v head_dim=192** 建（见模块③附），别从 config 拉真实 128。
- 成本：**高（次要风险）**，但可推迟。

### 模块⑤　forward metadata
- 路线 A 下 RadixAttention 继承 SGLang 标准 metadata，**通常不需要自定义 `build_forward_metadata`**（任务3已确认 metadata 桥是继承来的）。
- 仅当 hybrid 池需要 per-step 区分 full/SWA 索引时才补，模板 `_build_minimax_m3_forward_metadata`（:72）。
- 成本：**低～中（可能为 0）**。

### 模块⑥　量化重映射　`_prepare_mimo_v2_config`
- MiMo = fp8 block quant 128×128，且 `o_proj` 全在 `ignored_layers`。base wrapper 已 `sglang_skip_quant_config=True`；`register.py` 的量化 bypass 大体适用（fp8 而非 mxfp8）。
- 需要一个 `_prepare_mimo_v2_config(atom_config, model_arch)`（模板 `_prepare_minimax_m2_config`），设置 `quant_exclude_name_mapping` / `quant_default_exclude_layers` 覆盖 `o_proj`，并确认 fp8 descale 路径与模块③ sink 不冲突。
- 成本：**中**。

### 模块⑦　权重映射
- `mimo_v2.py` 已有 `packed_modules_mapping`（fused qkv）。若 SGLang 权重名与 atom 名不一致，补 `hf_to_atom_mapper`。
- 成本：**低～中**（多数已具备）。

### 模块⑧　MTP/EAGLE 投机（可选，性能对齐用）
- `atom/models/mimo_v2_mtp.py` 已存在；加 draft wrapper + `prepare_draft_model_config`（模板 `eagle3_llama` / `qwen3_5`）。
- 成本：**中，可选**，功能正确性不依赖它。

---

## 3. 最小可跑验证路径（建议里程碑）

1. **M1 起服务**：做模块①②⑥⑦ → 用 full-attention 近似（模块③ window 先不接、hybrid 池不绑）能加载权重并起 server，输出可能数值不对但不崩。验证注册链通。
2. **M2 SWA prefill 正确**：接通模块③的 extend（`flash_attn_varlen` 真 window+sink）→ 短序列 prefill-only 与原生 sglang+aiter（`mi_sgl/single/server.sh`）逐 token 对齐 logits。
3. **M3 SWA decode 正确**：接通模块③ decode（走 gluon PA + 真 sinks/window）→ 多步生成对齐。**这是风险收敛点**。
4. **M4 hybrid 池**：引入模块④省显存，回归 M2/M3 正确性。
5. **M5 cuda graph + 性能**：graph capture/replay 下 hybrid 池索引正确；对齐吞吐。（模块⑧ 视需要）

对齐基线：`/shared/amdgpu/home/fan_wu2_7kq/mi_sgl/single/server.sh`（native sglang+aiter，已能跑 MiMo）。

---

## 4. 工作量与风险汇总

| 模块 | 成本 | 风险 |
|---|---|---|
| ① arch 注册 | 低 | 低 |
| ② model 映射 | 低 | 低 |
| ③ backend SWA+sink | **高** | **高（decode 内核路径 = 关键）** |
| ④ hybrid SWA 池 | 高（可推迟） | 中 |
| ⑤ forward metadata | 低（可能 0） | 低 |
| ⑥ quant 重映射 | 中 | 中 |
| ⑦ 权重映射 | 低-中 | 低 |
| ⑧ MTP（可选） | 中 | 低 |

**总体判断**：这是 glue + 一段内核路径打通，不是从零移植。整体风险 **中～高**，全部压在模块③的 decode 路径（`pa_fwd_asm` 无 window/sink 形参，需改走 gluon PA 或扩内核）。建议按 M1→M5 里程碑推进，M3 是 go/no-go 收敛点。
