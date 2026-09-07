#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p logs
nohup flock -n .hardchunk16.launch.lock /home/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py >> logs/pipeline.log 2>&1 < /dev/null &
printf 'Launched wrapper PID=%s; log=logs/pipeline.log\n' "$!"
