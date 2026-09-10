# MiMo-V2.5-Pro × ATOM SGLang Plugin — 文档

> 更新日期：2026-09-10　　配套文档：[FINDINGS.md](FINDINGS.md)（任务1-3 调研）、[PLAN.md](PLAN.md)（模块级实施方案）、[TASK.md](TASK.md)
> 目标：评估 / 打通在 ROCm + MI355X 上，用 AMD 自研 Atom 推理引擎的 **SGLang plugin(OOT)** 运行 MiMo-V2.5-Pro 的工作量。

---

## 1. 一句话现状

插件已能**端到端启动并服务请求(batch 64 不崩)**；**prefill 正确**(首 token 连贯),但 **decode 从第 2 个 token 起输出乱码**(多语言随机 token,lm_eval 精度=0)。根因已定位到 **fp8 KV cache 在 decode(gluon PA)路径的写/读布局与 scale 不一致**,尚未修好。**这是唯一的正确性阻塞点**——prefill、注册、加载、显存全部已通。

---

## 2. 环境与访问

### 2.1 登录(务必用 ProxyJump,不要两段式 `ssh a ssh b`)
```bash
ssh -J fan_wu2_7kq@aac17.amd.com fan_wu2_7kq@vultr-mi355x-6 '<远端命令>'
```
- `aac17.amd.com` 只是 Slurm 登录控制器(无 podman/GPU);`vultr-mi355x-6` 才是 MI355X 计算节点。
- 两段式 `ssh a ssh b '...; ...'` 会在 `;` 处把后半段命令跑在 aac17 上,产生误导性输出。

### 2.2 容器
- 名称:`fanwu103-mimo-sgl-plugin`
- 镜像:`docker.io/rocm/atom-dev:sglang-v0.5.17-nightly_20260908`
- 进入:`podman exec -ti fanwu103-mimo-sgl-plugin /bin/bash`
- 关键包:`sglang 0.5.17`、`atom 0.1.6rc1.dev408+g2655fd25d`、`amd-aiter (/app/aiter-test)`、`torch 2.10.0+rocm7.2.4`、`sglang-kernel 0.4.5`
- rootless podman,host uid 20009。GPU 访问正常(rocminfo 显示 gfx950)。

### 2.3 ⚠️ 编辑代码的头号大坑:PYTHONPATH 遮蔽
容器里有**两份 ATOM**:
- `/app/ATOM` —— 镜像 baked 的旧副本(root 所有)。
- `/shared/amdgpu/home/fan_wu2_7kq/mi_atom/ATOM` —— **我们改动所在的共享 git 仓库**(可写、可 git)。

容器默认 `PYTHONPATH=/app/sglang/python:/app/ATOM`,`/app/ATOM` 会**遮蔽**共享路径,导致对共享路径的编辑**不生效**(`pip uninstall` 也删不掉 /app/ATOM——它在 venv 之外)。
**解决**(已写进 server_sgl.sh):
```bash
export PYTHONPATH=/app/sglang/python:/shared/amdgpu/home/fan_wu2_7kq/mi_atom/ATOM
```
验证:`podman exec fanwu103-mimo-sgl-plugin bash -c 'PYTHONPATH=/app/sglang/python:/shared/.../ATOM python3 -c "import atom;print(atom.__file__)"'` 应指向 `/shared/...`。**任何新脚本都必须带这行,否则跑的是 baked 旧代码(现象:SGLang 用它自带的原生 `/app/sglang/.../models/mimo_v2.py`,而不是 atom 的)。**

---

## 3. 代码分支与改动

- 仓库:`/shared/amdgpu/home/fan_wu2_7kq/mi_atom/ATOM`
- 分支:**`mimo_sgl_plugin`**,基线 commit `2655fd25`
- 状态:**未提交**(5 个文件 working-tree 改动,`git diff` 可见,约 82 行新增)。交接后建议先 `git commit` 固化。

改动一览(`git diff --stat`）：
| 文件 | 作用 | 状态 |
|---|---|---|
| `atom/plugin/sglang/runtime/model_arch.py` | 注册 arch(MODEL_TYPE→arch、`_prepare_mimo_v2_config` quant 重映射、ADAPTER_SPEC、ARCH_SPECS 元组) | ✅ 完成 |
| `atom/plugin/register.py` | `_ATOM_SUPPORTED_MODELS` 映射 arch→`atom.models.mimo_v2.MiMoV2ForCausalLM` + import | ✅ 完成 |
| `atom/plugin/sglang/register.py` | `_install_mimo_v2_pool_symmetry_patch`:wrap `ModelRunner.alloc_memory_pool`,建池前把 `model_config.v_head_dim/swa_v_head_dim` 顶到 head_dim=192(**不动 hf_config**) | ✅ 完成 |
| `.../full_attention/radix_attention.py` | 适配器建 SGLang RadixAttention 时传 `sliding_window_size`,并挂 `self.attn.sinks=sinks`(SWA plumbing) | ✅ 完成 |
| `.../full_attention/full_attention_backend.py` | `_layer_sink_ptr` 助手;extend 接 `window_size`/`sink_ptr`;decode gluon 接 `sinks`/`sliding_window` | ✅ 完成(但见 §6 decode 仍有 fp8 布局问题) |

> 说明:所有调试打点(`[ATOM-MiMoV2-KV-PATCH]`、`DBG-EXT/DBG-REF`)已清除。核心模型 `atom/models/mimo_v2.py`(standalone 已能正确跑)未改动。

---

## 4. 启动 & 测试脚本

### 4.1 启动 server(插件模式)
- 脚本:`/shared/amdgpu/home/fan_wu2_7kq/mi_atom/single/server_sgl.sh`
- 日志:`/shared/amdgpu/home/fan_wu2_7kq/mi_atom/single/log_atom_sgl`
- 当前内容(关键项均为必需,逐条说明见 §5):
```bash
export SGLANG_USE_AITER=1
export ATOM_FORCE_ATTN_TRITON=1        # MiMo head=192,asm PA decode 只支持128,强制 triton gluon
export SGLANG_ENABLE_TORCH_COMPILE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export PYTHONPATH=/app/sglang/python:/shared/.../mi_atom/ATOM   # 见 §2.3
MODEL_PATH=/shared/amdgpu/home/fan_wu2_7kq/models/MiMo-V2.5-Pro
python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
  --host localhost --port 10086 --trust-remote-code --tp-size 8 \
  --mem-fraction-static 0.8 --disable-radix-cache \
  --attention-backend aiter --page-size 64 --chunked-prefill-size -1 \
  --kv-cache-dtype fp8_e4m3
```
启动约需数分钟(权重加载 fp8 ~140GB/rank tp8)。就绪标志:日志出现 `The server is fired up and ready to roll!`。

### 4.2 快速验证(server 用 --net=host,计算节点上直接可达 localhost:10086)
```bash
curl -s http://localhost:10086/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"/shared/amdgpu/home/fan_wu2_7kq/models/MiMo-V2.5-Pro","prompt":"The capital of France is","max_tokens":20,"temperature":0}'
```
- **当前表现**:`max_tokens=1` → `' one'`(连贯,prefill OK);`max_tokens≥2` → `' one инвестици ...'`(第2个 token 起乱码,decode 坏)。

### 4.3 精度测试
- lm_eval(同事常用);gsm8k 输出样例目录:`/shared/amdgpu/home/fan_wu2_7kq/mi_sgl/single/`(有历史 `acc_5d_*.txt`、`tmp_output_gsm8k.txt`)。当前精度=0(decode 未修好)。

### 4.4 对照基线(正确参照物)
- **原生 atom(standalone)**:`/shared/amdgpu/home/fan_wu2_7kq/mi_atom/single/server.sh`(`atom.entrypoints.openai_server`)——**同模型正确**,是逐层/逐 kernel 对拍的黄金参照。
- **原生 sglang+aiter/triton**:`/shared/amdgpu/home/fan_wu2_7kq/mi_sgl/single/server*.sh`(用 SGLang 自带的 mimo 模型;注意原生 aiter 后端不支持 MiMo 的非对称 v,只能 triton)。

---

## 5. 已解决的阻塞点(按出现顺序,含配置由来)

1. **PYTHONPATH 遮蔽** → server_sgl.sh 里把 PYTHONPATH 指向共享 ATOM(§2.3)。
2. **v head_dim 非对称(qk=192/v=128)导致 KV 池 OOB**:SGLang 按 config 的 v=128 分配 V 池,但 atom 模型把 V 零填充到 192 再写 → 越界。→ `_install_mimo_v2_pool_symmetry_patch` 让池按对称 192 分配(代价:V cache 多~33% 零填充,与 standalone 一致)。
3. **page_size=1 时 fp8 KV 写 reshape 崩**(`kv_cache.py:138`,`block_size//x=1//16=0`)→ `--page-size 64`。
4. **aiter ASM PA decode 只支持 head_size=128,MiMo=192**(`asm_pa.cu:194`)→ `ATOM_FORCE_ATTN_TRITON=1`,decode 改走 triton `run_pa_decode_gluon`(支持任意 head_dim,且带 sinks/sliding_window 形参)。
5. **batch 64 extend 崩**(分块 prefill 使 `_forward_extend_mha` 的 `cu_seqlens`(用 `seq_lens` cumsum)与实际 q chunk 不符)→ `--chunked-prefill-size -1` 临时禁用分块。**真正修法**:`_forward_extend_mha` 应改用正确的 extend 元数据(`forward_metadata.qo_indptr/kv_indptr/kv_indices`)并读 KV cache,而非拿 `seq_lens` 硬 cumsum。
6. **prefill 首 token 之外全乱** → 见 §6(decode fp8 问题)。

---

## 6. ⛔ 当前开放问题:decode fp8 布局/scale(唯一阻塞正确性)

### 6.1 已确证的定位过程(结论可靠)
- **前向数值全程健康**:插桩 dump layer 0/1/2/34/35/68/69 的 q/k/v/o,全部无 NaN、量级合理。
- **注意力计算正确**:layer 0(全注意力层)用 torch 手算参考对拍 flash_attn_varlen 的输出,`maxdiff~5e-5`(bf16 精度级)→ **prefill 注意力内核完全正确**。
- **prefill 正确、decode 坏**:`max_tokens=1` 首 token 连贯,`≥2` 起乱码 → 坏在 decode(读 KV cache 那步),不在 prefill。
- 权重加载:仅 140/971 参数"未加载"= 70 层×2 的 `self_attn.attn.attn.k_scale/v_scale`(fp8 KV 运行时 per-block 量化缓冲,非 checkpoint 权重,烟雾弹);`lm_head`、`attention_sink_bias` 均已正确加载。

### 6.2 根因(decode gluon 路径的三处不一致)
decode 走 `full_attention_backend.py::_forward_decode_native_dense_mha`(ATOM_FORCE_ATTN_TRITON 分支)→ `run_pa_decode_gluon`:
1. **compute_type 错**:传 `torch.bfloat16`,但 cache 是 fp8(1 字节)→ 当 bf16(2 字节)读 → 字节错位。
2. **无 descale**:传 `k_scale=None/v_scale=None`,但 fp8 KV 是带 scale 写入的。
3. **写/读布局可能不一致**:写用 `_set_kv_buffer_native_dense`→`launch_reshape_and_cache_flash`(flash 布局);读时 gluon 又把 cache reshape/permute 成 **x-packed shuffle 布局**。

### 6.3 参照修法(standalone `atom/model_ops/attention_mha.py` 的 gluon 调用,~L562)
standalone 对 fp8 是:`compute_type=aiter.dtypes.fp8`;`k_scale/v_scale` 传 `layer.k_scale/v_scale`(numel>1 时 `unsqueeze(-1)`);且**写和读都用同一种 shuffle 布局**(fused rope_cache 写 shuffle、gluon 读 shuffle)。
> 我曾试过只改 `compute_type`+`k_scale/v_scale`(已回退),**仍乱码**——说明**写/读布局一致性**是还没打通的关键。下一步应:核对 `launch_reshape_and_cache_flash` 的输出布局 vs `_forward_decode_native_dense_mha` 里 gluon 读前的 reshape/permute 是否匹配;很可能 decode 的 KV 写应改用 `set_kv_buffer_with_layout_shuffle` 的 **fp8 asm** 分支(aiter `reshape_and_cache_with_pertoken_quant`,产出 shuffle 布局),而不是 `_set_kv_buffer_native_dense`(flash 布局)。
> ⚠️ 用户明确要求:**不要走 `reshape_and_cache_shuffle_triton` 这个 bf16 融合算子**(它在 head=192 上 `tl.arange` 报"非2的幂";fp8 的 asm 写路径与它不同)。

### 6.4 已排除的岔路
- **bf16 KV**:写要走 `reshape_and_cache_shuffle_triton`(head=192 的 `tl.arange` 报错)→ 不可行。
- **`--disable-hybrid-swa-memory`(统一全量池,绕 M4)**:bf16 下会撞 `swa_loc is not None`(SGLang SWAKVPool 写需要 swa_loc);且统一池 + 全量上下文会 OOM(单块 97GiB,需配 `--context-length` 压)。此路暂搁置。

---

## 7. 尚未做 / 后续(即使 decode 修好也还需要)

- **M4 hybrid SWA 池对接**:当前靠内核层 `sliding_window` 保证 SWA 正确性 + 对称池;SGLang 的 hybrid SWA 子池(swa_loc、windowed kv_indices)未真正对接。省显存 & 长上下文需要它。参照 SGLang 原生 aiter 后端的 `use_sliding_window_kv_pool` 机制。
- **分块 prefill 真修**:见 §5.5(`_forward_extend_mha` 用真元数据 + 读 cache),以去掉 `--chunked-prefill-size -1`。
- **MTP/EAGLE 投机**(可选,性能对齐):`atom/models/mimo_v2_mtp.py` 已存在,需 draft wrapper + `prepare_draft_model_config`。
- **精度验证**:decode 修好后用 lm_eval 对齐原生 triton 基线。

---

## 8. 记忆/上下文位置

本人(Claude)在 `~/.claude/projects/-apps-fanwu103-semi-cc-mimo-sgl-plugin/memory/` 有详细过程记忆(env-access、pool-symmetry、timing-gotcha、decode 诊断等),与本文档一致,可作补充。
