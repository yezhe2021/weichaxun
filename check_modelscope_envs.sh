#!/usr/bin/env bash
set -u

for py in \
  /home/yezhe/data/miniconda3/bin/python \
  /home/yezhe/data/miniconda3/envs/attnkv/bin/python \
  /home/yezhe/data/miniconda3/envs/flashrag/bin/python \
  /home/yezhe/data/miniconda3/envs/minimind/bin/python \
  /home/yezhe/data/miniconda3/envs/rosetta/bin/python \
  /home/yezhe/data/miniconda3/envs/yanwutang/bin/python \
  /mnt/hxt/miniconda3/envs/qwen35/bin/python
do
  echo "PY=$py"
  if [ ! -x "$py" ]; then
    echo "missing"
    continue
  fi
  "$py" - <<'PY' 2>&1 || true
import importlib.util as u
print("modelscope", bool(u.find_spec("modelscope")), "addict", bool(u.find_spec("addict")), "datasets", bool(u.find_spec("datasets")))
try:
    from modelscope.msdatasets import MsDataset
    print("msdataset_ok")
except Exception as e:
    print("msdataset_fail", type(e).__name__, str(e)[:200])
PY
done
