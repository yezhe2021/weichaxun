#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/yezhe/重实现/mmlupro_full28_head8_writer_b1_qwen3_1_7b_to_4b_seed1234"
mkdir -p "$ROOT/logs"
nohup bash "$ROOT/run_all_experiments.sh" > "$ROOT/logs/full_pipeline.log" 2>&1 </dev/null &
echo $! > "$ROOT/logs/full_pipeline.pid"
echo "Started pipeline PID=$(cat "$ROOT/logs/full_pipeline.pid")"
echo "tail -f '$ROOT/logs/full_pipeline.log'"
