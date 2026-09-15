#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mode="${1:-smoke}"
[[ "$mode" == smoke || "$mode" == pilot ]] || { echo 'Expected smoke or pilot'; exit 2; }
mkdir -p logs
# Wrapper lock prevents duplicate launches, including accidental concurrent log writers.
nohup flock -n ".${mode}.launch.lock" /home/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py --mode "$mode" >> "logs/${mode}_pipeline.log" 2>&1 < /dev/null &
printf 'Launched %s wrapper PID=%s; log=logs/%s_pipeline.log\n' "$mode" "$!" "$mode"
