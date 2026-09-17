#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/yezhe/异构模型/openbookqa_llama_qwen_gemma_composability_audit_seed1234"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
cd "$ROOT"
mkdir -p logs

while true; do
  if "$PYTHON" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)'; then
    echo "[$(date '+%F %T')] CUDA available; starting frozen composability audit" >> logs/launcher.log
    exec "$PYTHON" -u run_pipeline.py >> logs/full_pipeline.log 2>&1
  fi
  echo "[$(date '+%F %T')] CUDA unavailable; retrying in 60 seconds" >> logs/launcher.log
  sleep 60
done
