#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/重实现/mmlupro_b1_scale1024_qwen3_1_7b_to_4b_seed1234"
PID_FILE="$ROOT/logs/full_pipeline.pid"
LOG_FILE="$ROOT/logs/full_pipeline.log"

cd "$ROOT"
mkdir -p logs

if [[ -f "$PID_FILE" ]]; then
  previous_pid="$(cat "$PID_FILE")"
  if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
    printf 'Pipeline is already running with PID %s\n' "$previous_pid"
    exit 0
  fi
fi

nohup bash ./run_all_experiments.sh >"$LOG_FILE" 2>&1 </dev/null &
pipeline_pid=$!
printf '%s\n' "$pipeline_pid" >"$PID_FILE"
printf 'Started pipeline with PID %s\n' "$pipeline_pid"
