#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
mkdir -p "$ROOT/logs"
nohup "$PYTHON" -u "$ROOT/hybrid_audit.py" evaluate > "$ROOT/logs/pipeline.log" 2>&1 &
echo $! > "$ROOT/logs/pipeline.pid"
echo "started pid=$(cat "$ROOT/logs/pipeline.pid")"
