#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/yezhe/重实现/mmlupro_full28_head8_writer_b1_qwen3_1_7b_to_4b_seed1234"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
CONFIG="$ROOT/config.json"
cd "$ROOT"

run_step() {
  printf '[%s] START %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
  "$@"
  printf '[%s] DONE  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_step "$PYTHON" -u run_unit_tests.py
run_step "$PYTHON" -u audit_head_extension.py --config "$CONFIG" --samples 2
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" train --writer full28_head8 --stage a
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" gradient_audit --writer full28_head8 --functional-mode final --checkpoint "$ROOT/checkpoints/quick/full28_head8/stage_a/best.pt"
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" train --writer full28_head8 --stage b --functional-mode final
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" evaluate --scope formal
run_step "$PYTHON" -u analyze_head_mixing.py --config "$CONFIG"
printf '[%s] ALL FULL28-HEAD8 EXPERIMENTS COMPLETED\n' "$(date '+%Y-%m-%d %H:%M:%S')"
