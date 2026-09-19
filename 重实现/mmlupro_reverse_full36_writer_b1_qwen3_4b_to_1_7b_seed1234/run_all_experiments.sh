#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/重实现/mmlupro_reverse_full36_writer_b1_qwen3_4b_to_1_7b_seed1234"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
CONFIG="$ROOT/config.json"

cd "$ROOT"

run_step() {
  printf '[%s] START %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
  "$@"
  printf '[%s] DONE  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_step "$PYTHON" -u run_unit_tests.py
run_step "$PYTHON" -u audit_reused_assets.py --config "$CONFIG"
run_step "$PYTHON" -u calibrate.py --config "$CONFIG"
run_step "$PYTHON" -u run_gpu_tests.py
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" train --writer full36 --stage a
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" gradient_audit --writer full36 --functional-mode final --checkpoint "$ROOT/checkpoints/quick/full36/stage_a/best.pt"
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" train --writer full36 --stage b --functional-mode final
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" evaluate --scope formal
run_step "$PYTHON" -u compare_forward.py --config "$CONFIG"

printf '[%s] ALL REVERSE FULL-36 WRITER EXPERIMENTS COMPLETED\n' "$(date '+%Y-%m-%d %H:%M:%S')"
