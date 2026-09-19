#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/异构模型/openbookqa_official_full34_headmix256_train4096_gemma3_4b_to_qwen3_4b_seed1234"
PYTHON_BIN="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
cd "$ROOT"
mkdir -p logs

if [[ -f logs/full_pipeline.pid ]] && kill -0 "$(<logs/full_pipeline.pid)" 2>/dev/null; then
    echo "Pipeline already running with PID $(<logs/full_pipeline.pid)"
    exit 0
fi

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; refusing to launch study pipeline")
print(torch.cuda.get_device_name(0), flush=True)
PY

nohup "$PYTHON_BIN" -u run_pipeline.py --mode study all \
    > logs/full_pipeline.log 2>&1 < /dev/null &
echo $! > logs/full_pipeline.pid
echo "Started study pipeline PID $(<logs/full_pipeline.pid)"
