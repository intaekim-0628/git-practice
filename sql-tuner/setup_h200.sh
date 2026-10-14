#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Use a CUDA-enabled PyTorch/H200 host. No trust/signature verification bypass.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python - <<'PY'
import unsloth
import torch
assert torch.cuda.is_available(), 'CUDA-enabled PyTorch and NVIDIA driver required'
assert torch.cuda.device_count() == 1, 'Expose exactly one H200'
assert torch.cuda.is_bf16_supported(), 'BF16 support required'
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30)
PY
python -m pip freeze > runtime-lock.txt
python -m unittest discover -p 'test_*.py' -v
python train_gpt_oss_120b_qlora.py --validate-only
