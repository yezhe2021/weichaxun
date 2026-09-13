#!/usr/bin/env bash
set -u
ROOT="/home/yezhe/异构模型/mmlupro_standardprompt_rawanchor_full28_mlp_llama3_2_3b_to_qwen3_4b_seed1234"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
LOG="$ROOT/logs/pipeline.log"
cd "$ROOT" || exit 1
mkdir -p logs
stamp() { date '+[%Y-%m-%d %H:%M:%S]'; }
MIN_FREE_MIB=18000
while true; do
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "$(stamp) CUDA unavailable; waiting 60 seconds" >> "$LOG"
  else
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    if [ "${FREE_MIB:-0}" -ge "$MIN_FREE_MIB" ]; then break; fi
    echo "$(stamp) CUDA available but only ${FREE_MIB:-0} MiB free; waiting for ${MIN_FREE_MIB} MiB" >> "$LOG"
  fi
  sleep 60
done
echo "$(stamp) CUDA has ${FREE_MIB} MiB free; starting smoke" >> "$LOG"
if ! "$PYTHON" -u run_pipeline.py --mode smoke all >> "$LOG" 2>&1; then
  if tail -n 160 "$LOG" | grep -Eqi 'CUDA unavailable|NVML|driver|CUDA error'; then
    echo "$(stamp) Smoke interrupted by CUDA; rerun launcher after recovery" >> "$LOG"
  else echo "$(stamp) Non-CUDA smoke failure; stopping" >> "$LOG"; fi
  exit 1
fi
echo "$(stamp) Smoke passed; starting study" >> "$LOG"
if "$PYTHON" -u run_pipeline.py --mode study all >> "$LOG" 2>&1; then
  echo "$(stamp) ALL EXPERIMENTS COMPLETED" >> "$LOG"; exit 0
fi
if tail -n 160 "$LOG" | grep -Eqi 'CUDA unavailable|NVML|driver|CUDA error'; then
  echo "$(stamp) Study interrupted by CUDA; rerun launcher (completed stages are reused)" >> "$LOG"
else echo "$(stamp) Non-CUDA study failure; stopping" >> "$LOG"; fi
exit 1
