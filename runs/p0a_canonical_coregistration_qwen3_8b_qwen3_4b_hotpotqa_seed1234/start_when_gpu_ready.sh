#!/usr/bin/env bash
set -uo pipefail

cd /home/yezhe/可拔插/p0a_canonical_coregistration_qwen3_8b_qwen3_4b_hotpotqa_seed1234
PYTHON=/home/yezhe/data/miniconda3/envs/attnkv/bin/python

while true; do
  if nvidia-smi >/dev/null 2>&1 &&
     "$PYTHON" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    printf '[%s] CUDA is ready; starting P0-A pipeline\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    exec env PYTHONUNBUFFERED=1 "$PYTHON" run_pipeline.py --config config.json
  fi
  printf '[%s] CUDA unavailable; waiting 60 seconds\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  sleep 60
done
