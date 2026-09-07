#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p logs
nohup flock -n .raw-anchor.launch.lock /home/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py --mode study >> logs/pipeline.log 2>&1 < /dev/null &
printf 'Launched wrapper PID=%s; log=logs/pipeline.log\n' "$!"
