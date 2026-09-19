#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/yezhe/重实现/mmlupro_full36_head8_writer_b1_qwen3_4b_to_8b_seed1234"
mkdir -p "$ROOT/logs"
nohup bash -c '
  set -euo pipefail
  while true; do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
    if [[ "$free_mib" -ge 22000 ]]; then
      printf "[%s] GPU ready: %s MiB free\n" "$(date "+%Y-%m-%d %H:%M:%S")" "$free_mib"
      break
    fi
    printf "[%s] Waiting for GPU: %s MiB free, require >=22000 MiB\n" "$(date "+%Y-%m-%d %H:%M:%S")" "$free_mib"
    sleep 60
  done
  exec bash "$1/run_all_experiments.sh"
' _ "$ROOT" > "$ROOT/logs/full_pipeline.log" 2>&1 </dev/null &
echo $! > "$ROOT/logs/full_pipeline.pid"
echo "Started PID=$(cat "$ROOT/logs/full_pipeline.pid")"
echo "tail -f '$ROOT/logs/full_pipeline.log'"
