#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
mkdir -p "$ROOT/logs"
cd "$ROOT"
nohup "$PYTHON" -u run_pipeline.py --mode study all > logs/pipeline.log 2>&1 &
echo $! > logs/pipeline.pid
echo "Started PID $(cat logs/pipeline.pid)"
