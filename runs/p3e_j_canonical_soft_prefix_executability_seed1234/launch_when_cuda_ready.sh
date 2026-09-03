#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/伪查询/runs/p3e_j_canonical_soft_prefix_executability_seed1234"
PY="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
WAIT_LOG="$ROOT/p3e_j_cuda_wait.log"
WAIT_PID="$ROOT/p3e_j_cuda_wait.pid"

mkdir -p "$ROOT"
if [[ -f "$WAIT_PID" ]] && kill -0 "$(cat "$WAIT_PID")" 2>/dev/null; then
  echo "CUDA waiter is already running with PID $(cat "$WAIT_PID")"
  exit 0
fi

nohup bash -c "
  while true; do
    if \"$PY\" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
      printf '%s CUDA available; starting P3-E-J\n' \"\$(date -Is)\" >>\"$WAIT_LOG\"
      bash \"$ROOT/launch.sh\" >>\"$WAIT_LOG\" 2>&1
      exit 0
    fi
    printf '%s CUDA unavailable; retrying in 60 seconds\n' \"\$(date -Is)\" >>\"$WAIT_LOG\"
    sleep 60
  done
" >/dev/null 2>&1 &

echo $! >"$WAIT_PID"
echo "started CUDA waiter PID $(cat "$WAIT_PID")"
echo "wait log: $WAIT_LOG"
