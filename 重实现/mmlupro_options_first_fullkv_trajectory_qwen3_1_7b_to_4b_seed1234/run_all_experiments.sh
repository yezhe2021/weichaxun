#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/重实现/mmlupro_options_first_fullkv_trajectory_qwen3_1_7b_to_4b_seed1234"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
CONFIG="$ROOT/config.json"

cd "$ROOT"

run_step() {
  printf '[%s] START %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
  "$@"
  printf '[%s] DONE  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_step "$PYTHON" -u run_unit_tests.py
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" prepare
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" audit

run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" cache --split train --family both
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" cache --split validation --family both
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" cache --split test --family both
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" calibrate

run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" train --writer d2 --stage a
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" gradient_audit --writer d2 --checkpoint checkpoints/quick/d2/stage_a/best.pt
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" overfit --writer d2 --stage both

run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" train --writer d0 --stage both
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" train --writer d1 --stage both
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" train --writer d2 --stage b

run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" evaluate --writer d0 --stage both
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" evaluate --writer d1 --stage both
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" evaluate --writer d2 --stage both

printf '[%s] ALL EXPERIMENTS COMPLETED\n' "$(date '+%Y-%m-%d %H:%M:%S')"
