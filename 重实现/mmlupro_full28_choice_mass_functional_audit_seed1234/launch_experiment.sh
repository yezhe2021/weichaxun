#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/logs"
nohup bash "$ROOT/run_experiment.sh" > "$ROOT/logs/pipeline.log" 2>&1 &
echo $! > "$ROOT/logs/pipeline.pid"
echo "started pid=$(cat "$ROOT/logs/pipeline.pid")"
echo "tail -f '$ROOT/logs/pipeline.log'"
