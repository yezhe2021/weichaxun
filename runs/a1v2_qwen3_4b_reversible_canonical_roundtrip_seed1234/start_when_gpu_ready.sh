#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
MODE="${1:-smoke}"
LOG_DIR="$ROOT/logs"
PID_FILE="$LOG_DIR/pipeline_${MODE}.pid"

mkdir -p "$LOG_DIR"
echo "$$" > "$PID_FILE"

while true; do
  while true; do
    if nvidia-smi >/dev/null 2>&1 && "$PYTHON" -c 'import torch; assert torch.cuda.is_available(); torch.zeros(1, device="cuda")' >/dev/null 2>&1; then
      break
    fi
    printf '[%s] CUDA unavailable; waiting 30 seconds\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    sleep 30
  done

  printf '[%s] CUDA ready; starting/resuming mode=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$MODE"
  export PYTHONUNBUFFERED=1
  export EXPERIMENT_PYTHON="$PYTHON"
  "$PYTHON" -u "$ROOT/run_pipeline.py" --mode "$MODE"
  status=$?
  if [ "$status" -eq 0 ]; then
    exit 0
  fi
  if nvidia-smi >/dev/null 2>&1 && "$PYTHON" -c 'import torch; assert torch.cuda.is_available(); torch.zeros(1, device="cuda")' >/dev/null 2>&1; then
    printf '[%s] Non-CUDA failure; stopping with status=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$status"
    exit "$status"
  fi
  printf '[%s] CUDA failed during a stage; waiting before resume\n' "$(date '+%Y-%m-%d %H:%M:%S')"
done
