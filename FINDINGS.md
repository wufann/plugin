# MiMo-V2.5-Pro on Atom / SGLang Plugin — 调研发现

> 环境：ROCm + AMD MI355X（节点 `vultr-mi355x-6`），容器 `fanwu103-mimo-sgl-plugin`
> 镜像 `docker.io/rocm/atom-dev:sglang-v0.5.17-nightly_20260908`
> 目标：评估基于 AMD 自研推理引擎 Atom 的 SGLang plugin(OOT) 运行 MiMo-V2.5-Pro 的工作量

---

## 0. 环境访问备忘

- 登录链路：本地 → `aac17.amd.com`（Slurm 登录控制节点）→ `vultr-mi355x-6`（GPU 计算节点）。
- 推荐用 **ProxyJump 单跳**（避免 `ssh a ssh b '...;...'` 的分号在登录节点被拆分）：
  ```bash
  ssh -J fan_wu2_7kq@aac17.amd.com fan_wu2_7kq@vultr-mi355x-6 '<远程命令>'
  ```
- 关键包版本：`atom 0.1.6rc1.dev408`、`sglang 0.5.17`、`amd-aiter`、`torch 2.10.0+rocm7.2.4`、`transformers 5.12.1`。
- 源码位置：Atom `/app/ATOM`（可编辑安装），SGLang `/app/sglang/python/sglang`（可编辑安装）。

---

## 1. 任务 1：原生 Atom 运行 MiMo 的报错（已解决）

### 表象
`server.sh`（`python -m atom.entrypoints.openai_server --model .../MiMo-V2.5-Pro --kv_cache_dtype fp8 -tp 8 --trust-remote-code`）在引擎初始化崩溃，`log_atom` 调用链：

```
atom 引擎初始化 → aiter get_gfx() → 调 /opt/rocm-7.2.4/bin/rocminfo
→ rocminfo 非零退出 → RuntimeError: Get GPU arch from rocminfo failed
```

容器内直接跑 `rocminfo` 的真实错误：
```
Unable to open /dev/kfd read-write: Permission denied
root is not member of "nogroup" group, the default DRM access group
```

### 根因（与 Atom / MiMo 无关，是 GPU 设备权限）
- podman 为 **rootless**（宿主 uid 20009 = fan_wu2_7kq）。
- `/dev/kfd`、`/dev/dri/renderD*` 属主 `root:render`(GID 993) 0660；`card*` 属 `root:video`(44)。
- `--group-add keep-groups` 只能把**执行 `podman run` 那个 shell 的附加组**带进容器；rootless userns 下容器 root 的 CAP_DAC_OVERRIDE 对宿主 root 属主的设备无效。
- 对比两个容器 init 进程的附加组（宿主 `/proc/<pid>/status`）：

  | 容器 | 附加组(host gid) | render(993)/video(44) | rocminfo |
  |---|---|---|---|
  | `fanwu103-sa-0907-dsv4` | `44 993 9000 9001 20009` | ✅ | 正常(gfx950) |
  | `fanwu103-mimo-sgl-plugin`(旧) | `9000 9001 20009` | ❌ | Permission denied |

  两容器启动参数完全相同 → **唯一差别是启动它的 shell 有没有 render/video 组**。旧 mimo 容器从缺组的 shell 启动，故失败。

### 修复（不需要管理员）
在 `id` 能看到 render+video 的 shell（即能跑通 rocminfo 的那个上下文）里，用相同参数重建容器：
```bash
podman rm -f fanwu103-mimo-sgl-plugin
podman run -e DISPLAY= --net=host --pid=host --shm-size 64g --privileged -it \
  --device=/dev/kfd --device=/dev/dri --group-add keep-groups \
  -w /shared/amdgpu/home/fan_wu2_7kq \
  -v /mnt/:/mnt/ -v /data/:/data/ -v /home/:/home/ -v /shared/:/shared/ \
  --name fanwu103-mimo-sgl-plugin \
  docker.io/rocm/atom-dev:sglang-v0.5.17-nightly_20260908 /bin/bash
```
**状态：已修复，原生 Atom 运行 MiMo 正常。**
> 验证一句话：容器内 `rocminfo` 能列出 `gfx950` 即 OK。

---

## 2. 任务 2：Atom 的 SGLang plugin(OOT) 机制

### 2.1 两个挂载接缝（均无需改 SGLang 启动脚本）
1. **entry point 组 `sglang.srt.plugins`**
   `atom_sglang = atom.plugin.sglang.register:register_plugin`
   SGLang 启动时自动发现并调用 `register_plugin()`，安装一批**前置 monkeypatch**：量化门禁绕过、loader 跳过原生量化、decode cuda graph 的 forward-context、Kimi 处理器、load_config patch。
2. **环境变量 `SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models`**
   SGLang `srt/models/registry.py:133` → `ModelRegistry.register(external_pkg, overwrite=True)`，用 Atom 的模型类**覆盖** SGLang 自带实现。

### 2.2 注册的单一真相源
`atom/plugin/sglang/runtime/model_arch.py::MODEL_ARCH_SPECS` 是**唯一架构注册表**。
`base_model_wrapper.py` 末尾：
```python
EntryClass = []
for _name in MODEL_ARCH_SPECS:
    _cls = type(_name, (_AtomCausalLMBaseForSglang,), {})
    globals()[_name] = _cls
    EntryClass.append(_cls)
```
→ 每个 arch 自动生成一个 SGLang 面向的 wrapper 类，`EntryClass` 供 SGLang 注册。
每个 `SGLangModelAdapterSpec` 带钩子：`forward_metadata_builder`、`prepare_config`、`install_adapters`、`construction_context`、`prepare_draft_model_config`、cache-view 绑定。

### 2.3 运行流程
```
SGLang 命中 arch
 → Atom wrapper 类 (_AtomCausalLMBaseForSglang 子类)
 → prepare.py::prepare_model(config)
   → resolve_model_arch_spec(config)
   → 查 register.py::_ATOM_SUPPORTED_MODELS  (arch → 真实 atom.models.* 类)
   → generate_atom_config_for_plugin_mode
   → register_ops_to_sglang()      # 核心：_register_custom_attention_to_sglang 替换 SGLang 注意力算子
   → init_aiter_dist()             # aiter TP/EP 集合通信
   → apply_graph_capture_patch()   # 让 aiter 自定义 all-reduce 进入 graph capture
   → 构造真实 atom.models.* 模型
```
> `set_attn_cls()` 现为 no-op——注意力选择已下沉到 `atom.model_ops.base_attention.Attention` 的构造期分发。

**被 Atom 接管/替换的层**：注意力算子、量化加载、权重 loader、forward context、cuda graph capture、分布式集合通信、整套模型类。

### 2.4 当前已支持的架构
Qwen3 / Qwen3Moe、Glm4Moe、DeepseekV3 / V32 / V4、GlmMoeDsa、MiniMaxM2、MiniMaxM3Sparse、Qwen3_5(Moe)、KimiK25 / K3，及各 MTP / Eagle3 draft。

---

## 3. MiMo-V2.5-Pro 现状与缺口

### 3.1 架构（config.json）
- `MiMoV2ForCausalLM`，`model_type=mimo_v2`，`auto_map → modeling_mimo_v2.py`（需 `--trust-remote-code`）。
- 70 层，hidden 6144，128 attn 头 / 8 KV 头（GQA），head_dim 192。
- **MoE**：384 路由专家、top-8、`noaux_tc` 分组受限路由（topk_group=1）、norm_topk、无 shared expert、moe_intermediate 2048、dense/MoE 按 `moe_layer_freq` 混排。
- **混合注意力**：`hybrid_layer_pattern`（0=full，1=SWA），多数层 `sliding_window=128` + **attention sink**（`add_swa_attention_sink_bias`、`attention_value_scale=0.612`），双 rope theta（full 1e7 / swa 1e4）。
- fp8 分块量化(128×128)，所有 `o_proj` 在 `ignored_layers`。
- 有 MTP；vocab 152576；1M 上下文。

### 3.2 现有资产（重要利好）
- ✅ `atom/models/mimo_v2.py`（独立 `MiMoV2ForCausalLM`：MoE、混合注意力含 `per_layer_sliding_window` + `sinks`、`packed_modules_mapping`）——**任务1原生运行用的就是它**。
- ✅ `atom/models/mimo_v2_mtp.py`。
- ✅ aiter/atom 已有滑窗 + sink 的注意力 kernel。

### 3.3 缺口
`atom/plugin` 目录对 `mimo` **零引用**：`_ATOM_SUPPORTED_MODELS` 无、`MODEL_ARCH_SPECS` 无、`plugin/sglang/models/` 无 mimo wrapper、无 attention backend、无 KV cache 桥接。

---

## 4. 需新增/替换的模块（工作量结论）

核心模型已存在 → 这是**接线/桥接**活，不是从零移植。

| # | 模块 | 位置 | 工作量 |
|---|---|---|---|
| 1 | 注册 arch 条目（自动生成 wrapper 类） | `runtime/model_arch.py::MODEL_ARCH_SPECS` 加 `"MiMoV2ForCausalLM"` | 低 |
| 2 | arch→模型映射 | `plugin/register.py::_ATOM_SUPPORTED_MODELS`（+ MTP draft） | 低 |
| 3 | **注意力后端桥** `SGLangATOMMiMoV2Attention`：逐层切 full/SWA、透传 sink bias 与 value_scale、消费 SGLang forward metadata | `plugin/sglang/attention_backend/` + `models/mimo_v2.py`（模板：`minimax_m3.py`） | **高** |
| 4 | **混合 KV cache 视图绑定**：SWA 层窗口池 / full 层全量池，接 SGLang `token_to_kv_pool` | `model_arch.py` cache-view + construction_context（模板：minimax_m3 / deepseek_v4 pool 桥） | **高（主风险）** |
| 5 | `forward_metadata_builder`（混合 full/SWA 元数据） | adapter spec | 中 |
| 6 | 量化排除映射（o_proj `ignored_layers`，fp8） | wrapper 的 `quant_exclude_name_mapping` / `quant_default_exclude_layers`；`sglang_skip_quant_config` 基类已置 True | 中 |
| 7 | 权重名映射（`attention_projection_layout: fused_qkv` → `packed_modules_mapping` / `hf_to_atom_mapper`） | wrapper | 低-中 |
| 8 | MTP / EAGLE 投机解码（对齐原生性能，可选） | draft wrapper + `prepare_draft_model_config`（模板：eagle3_llama / qwen3_5 dflash） | 中，可选 |

**风险/时间集中在 #3、#4**：MiMo 的"混合滑窗 + attention sink"注意力要接到 SGLang 的元数据与混合 KV pool，而现有插件模型没有完全相同的形态（minimax_m3 是 sparse MLA，结构最近但注意力类型不同），需新写、不能纯抄。

### 4.1 两个待确认点的验证结果（任务3，已完成）

**验证点 1：mimo_v2 注意力能否接 SGLang metadata —— ✅ 基本零成本。**
- 调用链：`MiMoV2Attention.attn = Attention(sinks=, per_layer_sliding_window=)`（`atom/models/mimo_v2.py`）→ sglang 模式下分发到 `atom/plugin/sglang/attention.py::AttentionForSGLang`，它继承 `atom/plugin/sglang/attention_backend/full_attention/radix_attention.py::RadixAttention`，内部再包 **SGLang 原生 `sglang.srt.layers.radix_attention.RadixAttention`**。
- 该适配器 **接受并透传 `sinks` 与 `per_layer_sliding_window`**（radix_attention.py:40-41、57-58）。forward_batch / metadata / KV pool 全走 SGLang 标准 RadixAttention 通路。
- 结论：模型侧几乎不用改，metadata 桥接是**继承来的**，不是要新造的。这正是现有插件模型共用的契约。

**验证点 2：SGLang 0.5.17 的 SWA / 混合 KV pool —— ✅ SGLang 侧完备；真正缺口在 ATOM 的 backend。**
- SGLang 侧原生齐全（也正是原生 `mi_sgl` 的 sglang+aiter 能跑 MiMo 的原因）：`mem_cache/` 有 `swa_memory_pool.py`、`base_swa_memory_pool.py`、`swa_radix_cache.py`、`pure_swa_radix_cache.py`、`allocator/swa.py`、`unified_cache/swa_component.py`、`hybrid_cache/`；`memory_pool.py` 有 `HybridReqToTokenPool`、`HybridLinearKVPool`（带 `full_attention_layer_ids` 映射）；`model_runner.py` 有 `is_hybrid_swa` / `attention_chunk_size` / `resolve_sliding_window_size`；原生 `RadixAttention` 已转发 `sinks` 与 `sliding_window_size`。
- **ATOM 插件 backend 是缺口所在**（`attention_backend/full_attention/full_attention_backend.py`）：
  - 已有脚手架：读取 `full_attention_layer_id_mapping`（:116）、`sliding_window_size = getattr(layer, ...)`（:1755）；所调用的 aiter kernel（`mha_batch_prefill_func`、`flash_attn_varlen_func`、`run_pa_decode_gluon`、`paged_attention_ragged`）**都已暴露 `window_size` / `sink_ptr` / `sinks` / `sliding_window` 形参**。
  - 但**当前所有调用点都传的是禁用值**：prefill `window_size=(-1,-1), sink_ptr=None`（:1953）、extend `window_size=(-1,-1,0), sink_ptr=None`（:2017）、decode `sinks=None, sliding_window=-1`（:2760）；且 `_should_use_native_dense_mha` 在设了 sliding_window 时直接跳出。
  - 后果：MiMo 若现在走插件，SWA 层会被当成 full attention 跑（结果错误）、attention sink 被忽略、SWA 层也没绑定窗口子池。

**结论修正（相对第 4 节的重新评估）：**
- 因为 kernel 已支持、metadata 桥已继承，#3/#4 **不是"从零写 backend"，而是"把 ATOM `full_attention_backend.py` 里已布好线的 SWA+sink 路径真正打通"**：在 prefill / extend / decode / cuda-graph capture+replay 各路径，从 `layer.sliding_window_size` 和 sink bias 参数填入真实的 `window_size`/`sink_ptr`/`sinks`/`sliding_window`，并为窗口层绑定 SGLang 的 hybrid SWA KV 子池（`full_attention_layer_id_mapping`）。
- 风险从"高（需新写注意力后端）"下调为**中**，但仍是核心工作项，重点在正确性验证：SWA 掩码、sink 数值、以及 cuda-graph 模式下 hybrid 池索引。

---

## 附：关键路径速查
- Atom 插件入口：`/app/ATOM/atom/plugin/sglang/register.py`、`prepare.py`
- 架构注册表：`/app/ATOM/atom/plugin/sglang/runtime/model_arch.py`
- 支持模型映射：`/app/ATOM/atom/plugin/register.py`（`_ATOM_SUPPORTED_MODELS`）
- wrapper 基类：`/app/ATOM/atom/plugin/sglang/models/base_model_wrapper.py`
- 现有 MiMo 模型：`/app/ATOM/atom/models/mimo_v2.py`、`mimo_v2_mtp.py`
- MiMo 权重/配置：`/shared/amdgpu/home/fan_wu2_7kq/models/MiMo-V2.5-Pro/`
- 原生 Atom 脚本：`/shared/amdgpu/home/fan_wu2_7kq/mi_atom/single/{server.sh,log_atom}`
- 原生 SGLang 脚本：`/shared/amdgpu/home/fan_wu2_7kq/mi_sgl/single/server.sh`
