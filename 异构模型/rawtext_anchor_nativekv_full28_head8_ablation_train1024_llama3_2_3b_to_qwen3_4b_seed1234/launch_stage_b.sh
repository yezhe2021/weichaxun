#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
mkdir -p "$ROOT/logs"
cd "$ROOT"
nohup "$PYTHON" -u run_pipeline.py --mode study stage_b > logs/stage_b_pipeline.log 2>&1 &
echo $! > logs/stage_b_pipeline.pid
echo "Started Stage B PID $(cat logs/stage_b_pipeline.pid)"
