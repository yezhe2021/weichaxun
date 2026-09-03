#!/usr/bin/env bash
set -u
ROOT="/home/yezhe/可拔插/r0_raw_native_kv_cross_model_qwen3_8b_to_4b_hotpotqa_seed1234"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
mkdir -p "$ROOT/logs"
cd "$ROOT"

while true; do
  if "$PYTHON" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "[$(date '+%F %T')] CUDA ready; starting/resuming R0 pipeline"
    if "$PYTHON" -u run_pipeline.py; then
      echo "[$(date '+%F %T')] R0 pipeline finished"
      exit 0
    fi
    echo "[$(date '+%F %T')] Pipeline stopped; retrying after GPU recovery"
  else
    echo "[$(date '+%F %T')] CUDA unavailable; waiting 60 seconds"
  fi
  sleep 60
done
