#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/伪查询/runs/p3e_j_canonical_soft_prefix_executability_seed1234"
LOG="$ROOT/p3e_j_run.log"
PID="$ROOT/p3e_j_run.pid"
mkdir -p "$ROOT"
if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "P3-E-J is already running with PID $(cat "$PID")"
  exit 0
fi
nohup bash "$ROOT/run_all.sh" >"$LOG" 2>&1 &
echo $! >"$PID"
echo "started PID $(cat "$PID")"
echo "log: $LOG"
