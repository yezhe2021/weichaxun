#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/yezhe/异构模型/mmlupro_memoryfirst_receiver_oracle_audit_llama3_2_3b_to_qwen3_4b_seed1234"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
LOG="$ROOT/logs/forced_stageb.log"
mkdir -p "$ROOT/logs" "$ROOT/runs/study"
cd "$ROOT"
exec 9>"$ROOT/runs/study/forced_stageb.lock"
flock -n 9 || { echo "A forced Stage-B pipeline is already running"; exit 1; }
stamp() { date '+[%Y-%m-%d %H:%M:%S]'; }
echo "$(stamp) Forced Memory-first Stage-B pipeline started" >> "$LOG"
nvidia-smi >> "$LOG" 2>&1
echo "$(stamp) Preparing Memory-first Oracle teachers" >> "$LOG"
"$PYTHON" -u run_pipeline.py --mode study prepare_teachers >> "$LOG" 2>&1
echo "$(stamp) Training 512-step Memory-first Stage-B" >> "$LOG"
"$PYTHON" -u run_pipeline.py --mode study stage_b >> "$LOG" 2>&1
echo "$(stamp) Running final evaluation" >> "$LOG"
"$PYTHON" -u run_pipeline.py --mode study evaluate >> "$LOG" 2>&1
echo "$(stamp) FORCED STAGE-B EXPERIMENT COMPLETED" >> "$LOG"
