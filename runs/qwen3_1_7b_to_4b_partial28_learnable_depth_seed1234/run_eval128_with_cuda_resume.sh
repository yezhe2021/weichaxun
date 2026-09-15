#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/logs"
while true; do
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] CUDA unavailable; retrying in 30 seconds"
    sleep 30
    continue
  fi
  "/home/yezhe/data/miniconda3/envs/attnkv/bin/python" -u "$ROOT/run_eval128.py"
  status=$?
  if [ "$status" -eq 0 ]; then exit 0; fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] CUDA failure detected; waiting to resume"
    sleep 30
    continue
  fi
  echo "[$(date '+%F %T')] Non-CUDA failure; stopping with status=$status"
  exit "$status"
done
