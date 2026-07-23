#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/伪查询/runs/p3e_h_evidence_assimilation_reader_v2_seed1234"
LOG="$ROOT/p3e_h_run.log"
PID="$ROOT/p3e_h_run.pid"
mkdir -p "$ROOT"
if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "P3-E-H is already running with PID $(cat "$PID")"
  exit 0
fi
nohup bash "$ROOT/run_all.sh" >"$LOG" 2>&1 &
echo $! >"$PID"
echo "started PID $(cat "$PID")"
echo "log: $LOG"
