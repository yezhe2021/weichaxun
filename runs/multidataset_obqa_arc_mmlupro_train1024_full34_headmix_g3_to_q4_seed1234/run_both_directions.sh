#!/usr/bin/env bash
set -u

PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
FORWARD=/home/yezhe/异构模型/multidataset_obqa_arc_mmlupro_train1024_full34_headmix_g3_to_q4_seed1234
REVERSE=/home/yezhe/异构模型/multidataset_obqa_arc_mmlupro_train1024_full36_headmix_q4_to_g3_seed1234
QUEUE_LOG="$FORWARD/logs/bidirectional_queue.log"
mkdir -p "$FORWARD/logs" "$REVERSE/logs"
stamp() { date '+[%Y-%m-%d %H:%M:%S]'; }

wait_for_gpu() {
  while true; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= 18000 )); then
      echo "$(stamp) GPU concurrency window ready: ${free_mb} MiB free" | tee -a "$QUEUE_LOG"
      return
    fi
    echo "$(stamp) waiting for >=18000 MiB free GPU memory: ${free_mb:-unknown}" >> "$QUEUE_LOG"
    sleep 30
  done
}

run_resumable() {
  local root=$1 mode=$2 log=$3
  while true; do
    wait_for_gpu
    echo "$(stamp) START $root mode=$mode" | tee -a "$QUEUE_LOG" "$log"
    if (cd "$root" && "$PY" -u run_pipeline.py --mode "$mode" all) >> "$log" 2>&1; then
      echo "$(stamp) DONE $root mode=$mode" | tee -a "$QUEUE_LOG" "$log"
      return
    fi
    if tail -n 250 "$log" | grep -Eqi 'CUDA|out of memory|CUBLAS|NVIDIA|driver'; then
      echo "$(stamp) CUDA/OOM interruption; preserving completed stages and retrying" | tee -a "$QUEUE_LOG" "$log"
      sleep 60
    else
      echo "$(stamp) NON-CUDA FAILURE $root mode=$mode" | tee -a "$QUEUE_LOG" "$log"
      return 1
    fi
  done
}

run_resumable "$FORWARD" smoke "$FORWARD/logs/smoke_pipeline.log" || exit 1
run_resumable "$REVERSE" smoke "$REVERSE/logs/smoke_pipeline.log" || exit 1
run_resumable "$FORWARD" study "$FORWARD/logs/full_pipeline.log" || exit 1
run_resumable "$REVERSE" study "$REVERSE/logs/full_pipeline.log" || exit 1
echo "$(stamp) ALL GEMMA-QWEN MULTIDATASET EXPERIMENTS COMPLETED" | tee -a "$QUEUE_LOG"
