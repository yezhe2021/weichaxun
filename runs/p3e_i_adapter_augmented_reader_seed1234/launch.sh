#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/伪查询/runs/p3e_i_adapter_augmented_reader_seed1234"
LOG="$ROOT/p3e_i_run.log"
PID="$ROOT/p3e_i_run.pid"
mkdir -p "$ROOT"
if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "P3-E-I is already running with PID $(cat "$PID")"
  exit 0
fi
nohup bash "$ROOT/run_all.sh" >"$LOG" 2>&1 &
echo $! >"$PID"
echo "started PID $(cat "$PID")"
echo "log: $LOG"
