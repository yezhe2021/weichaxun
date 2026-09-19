#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
mkdir -p "$ROOT/logs"
cd "$ROOT"
nohup "$PYTHON" -u run_pipeline.py --mode study phase1 > logs/phase1_pipeline.log 2>&1 &
echo $! > logs/phase1_pipeline.pid
echo "Started PID $(cat logs/phase1_pipeline.pid)"
