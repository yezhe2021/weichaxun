#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
MODE="${1:-development}"
mkdir -p "$ROOT/logs"
while true; do
  if ! nvidia-smi >/dev/null 2>&1; then
    printf '[%s] CUDA unavailable; retry in 30 seconds\n' "$(date '+%F %T')"
    sleep 30
    continue
  fi
  "$PYTHON" -u "$ROOT/run_pipeline.py" --mode "$MODE"
  status=$?
  if [ "$status" -eq 0 ]; then
    printf '[%s] Audit completed\n' "$(date '+%F %T')"
    exit 0
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    printf '[%s] CUDA failure detected; retry after recovery\n' "$(date '+%F %T')"
    sleep 30
    continue
  fi
  printf '[%s] Non-CUDA failure; stopping with status=%s\n' "$(date '+%F %T')" "$status"
  exit "$status"
done
