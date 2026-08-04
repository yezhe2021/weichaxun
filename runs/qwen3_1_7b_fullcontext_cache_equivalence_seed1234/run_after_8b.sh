#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
UPSTREAM="/home/yezhe/可拔插/qwen3_8b_to_4b_writer_v2_seed1234"
UPSTREAM_MARKER="$UPSTREAM/artifacts/development/stage_markers/evaluate.done"
mkdir -p "$ROOT/logs"

echo "[$(date '+%F %T')] waiting for completed 8B Writer v2 pipeline"
while [ ! -f "$UPSTREAM_MARKER" ]; do
  if ! pgrep -f "$UPSTREAM/run_pipeline.py --mode development" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] upstream 8B pipeline stopped without completion marker"
    exit 2
  fi
  sleep 60
done

echo "[$(date '+%F %T')] 8B completed; starting 1.7B cache smoke"
bash "$ROOT/run_with_cuda_resume.sh" smoke
smoke_status=$?
if [ "$smoke_status" -ne 0 ]; then
  echo "[$(date '+%F %T')] 1.7B cache smoke failed with status=$smoke_status"
  exit "$smoke_status"
fi

echo "[$(date '+%F %T')] smoke passed; starting 1.7B formal cache audit"
bash "$ROOT/run_with_cuda_resume.sh" development
