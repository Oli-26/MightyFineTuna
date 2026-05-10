#!/usr/bin/env bash
# One-shot setup for running training + HF inference on the AMD GPU PC.
# Picker / pending-gen via llama.cpp already works on PC — this adds the
# PyTorch+ROCm path so train_*.py and bulk_infer.py can run too.
#
# Usage on PC: ./setup_pc.sh [rocm_version]

set -euo pipefail
ROCM="${1:-6.4}"   # check supported with: rocm-smi --version

echo "[setup] checking GPU..."
if ! command -v rocm-smi >/dev/null; then
  echo "WARN: rocm-smi not found — install ROCm first:"
  echo "  https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
  exit 1
fi
rocm-smi --showproductname --showmeminfo vram | tail -10

echo
echo "[setup] PyTorch + ROCm $ROCM via uv..."
# This pins torch to a ROCm wheel. uv inline-deps in train_*.py will use these
# from a uv-managed venv tied to that script.
uv venv --python 3.11 .venv-rocm || true
source .venv-rocm/bin/activate
pip install --pre torch torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/nightly/rocm${ROCM}"

echo
echo "[setup] core deps..."
pip install \
  "transformers==4.46.3" \
  "trl==0.12.2" \
  "peft==0.13.2" \
  "accelerate>=1.0" \
  "datasets>=3.0" \
  "huggingface_hub>=0.25" \
  "sentencepiece>=0.2" \
  "protobuf>=4" \
  "safetensors" \
  "httpx" \
  "fastapi" \
  "uvicorn[standard]"

echo
echo "[setup] NOTE: bitsandbytes 4-bit quant has flaky AMD support."
echo "       Train scripts default to fp16/bf16 on this venv — that's fine for 16GB VRAM."
echo "       Use --quant none everywhere on PC."

echo
echo "[setup] checking torch sees the GPU..."
python -c "
import torch
print('cuda available (HIP):', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  [{i}] {torch.cuda.get_device_name(i)}')
"

echo
echo "[setup] DONE. Activate with: source .venv-rocm/bin/activate"
echo "       Then run train_*.py / bulk_infer.py with --quant none."
