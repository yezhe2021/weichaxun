#!/usr/bin/env bash
set -u

for py in \
  /home/yezhe/data/miniconda3/bin/python \
  /home/yezhe/data/miniconda3/envs/attnkv/bin/python \
  /home/yezhe/data/miniconda3/envs/bishai01/bin/python \
  /home/yezhe/data/miniconda3/envs/download/bin/python \
  /home/yezhe/data/miniconda3/envs/flashrag/bin/python \
  /home/yezhe/data/miniconda3/envs/hello_agent/bin/python \
  /home/yezhe/data/miniconda3/envs/lmcache/bin/python \
  /home/yezhe/data/miniconda3/envs/minimind/bin/python \
  /home/yezhe/data/miniconda3/envs/proj_mydeepspeed/bin/python \
  /home/yezhe/data/miniconda3/envs/rosetta/bin/python \
  /home/yezhe/data/miniconda3/envs/test01/bin/python \
  /home/yezhe/data/miniconda3/envs/test02/bin/python \
  /home/yezhe/data/miniconda3/envs/yanwutang/bin/python \
  /mnt/hxt/miniconda3/bin/python \
  /mnt/hxt/miniconda3/envs/qwen35/bin/python
do
  echo "PY=$py"
  if [ ! -x "$py" ]; then
    echo "missing"
    continue
  fi
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
