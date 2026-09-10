export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_USE_AITER=1
export ATOM_FORCE_ATTN_TRITON=1
export SGLANG_ENABLE_TORCH_COMPILE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export PYTHONPATH=/app/sglang/python:/shared/amdgpu/home/fan_wu2_7kq/mi_atom/ATOM
MODEL_PATH=/shared/amdgpu/home/fan_wu2_7kq/models/MiMo-V2.5-Pro

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host localhost \
    --port 10086 \
    --trust-remote-code \
    --tp-size 8 \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --attention-backend aiter \
    --page-size 64 \
    --chunked-prefill-size -1 \
    --kv-cache-dtype fp8_e4m3
