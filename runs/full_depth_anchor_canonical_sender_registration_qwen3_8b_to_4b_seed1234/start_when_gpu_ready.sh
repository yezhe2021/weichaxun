#!/usr/bin/env bash
set -u
ANCHOR_ROOT="/home/yezhe/可拔插/full_depth_anchor_canonical_sender_registration_qwen3_8b_to_4b_seed1234"
ANCHOR_PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
mkdir -p "$ANCHOR_ROOT/logs"
cd "$ANCHOR_ROOT"

while true; do
  if "$ANCHOR_PYTHON" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "[$(date '+%F %T')] CUDA ready; starting/resuming full-depth anchor pipeline"
    if "$ANCHOR_PYTHON" -u run_pipeline.py; then
      echo "[$(date '+%F %T')] Full-depth anchor pipeline finished"
      exit 0
    fi
    echo "[$(date '+%F %T')] Pipeline stopped; retrying after GPU recovery"
  else
    echo "[$(date '+%F %T')] CUDA unavailable; waiting 60 seconds"
  fi
  sleep 60
done
