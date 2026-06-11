#!/usr/bin/env bash
set -u

for py in \
  /home/yezhe/.virtualenvs/Kivi/bin/python \
  /home/yezhe/.virtualenvs/yuancheng项目/bin/python \
  /home/yezhe/.virtualenvs/yanwutang/bin/python \
  /home/yezhe/my_lmcache/bin/python \
  /home/yezhe/data/miniconda3/bin/python \
  /home/yezhe/C2C/bin/python \
  /home/yezhe/yanwutang/cacheblend/bin/python \
  /root/software/LLM/.venv/bin/python \
  /mnt/hxt/miniconda3/bin/python
do
  echo "PY=$py"
  "$py" - <<'PY' 2>/dev/null || true
import importlib.util as u
print(
    "torch", bool(u.find_spec("torch")),
    "transformers", bool(u.find_spec("transformers")),
    "accelerate", bool(u.find_spec("accelerate")),
    "datasets", bool(u.find_spec("datasets")),
    "numpy", bool(u.find_spec("numpy")),
)
if u.find_spec("torch"):
    import torch
    print("torchver", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY
done
