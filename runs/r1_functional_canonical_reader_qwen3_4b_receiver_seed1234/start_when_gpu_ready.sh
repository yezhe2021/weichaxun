#!/usr/bin/env bash
set -u
R1_ROOT="/home/yezhe/可拔插/r1_functional_canonical_reader_qwen3_4b_receiver_seed1234"
R1_PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
mkdir -p "$R1_ROOT/logs"
cd "$R1_ROOT"

while true; do
  if "$R1_PYTHON" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "[$(date '+%F %T')] CUDA ready; starting/resuming R1 pipeline"
    if "$R1_PYTHON" -u run_pipeline.py; then
      echo "[$(date '+%F %T')] R1 pipeline finished"
      exit 0
    fi
    echo "[$(date '+%F %T')] R1 stopped; retrying after GPU recovery"
  else
    echo "[$(date '+%F %T')] CUDA unavailable; waiting 60 seconds"
  fi
  sleep 60
done
