#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/yezhe/重实现/mmlupro_full36_head8_writer_b1_qwen3_8b_to_4b_seed1234"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
CONFIG="$ROOT/config.json"
cd "$ROOT"
run_step() { printf '[%s] START %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; "$@"; printf '[%s] DONE  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
run_step "$PYTHON" -u run_unit_tests.py
for split in train validation test; do run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" cache --family source8 --split "$split"; done
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" calibrate
run_step "$PYTHON" -u audit_direct_head8.py --config "$CONFIG" --samples 2
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" train --writer full36_head8 --stage a
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" gradient_audit --writer full36_head8 --functional-mode final --checkpoint "$ROOT/checkpoints/quick/full36_head8/stage_a/best.pt"
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" train --writer full36_head8 --stage b --functional-mode final
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" evaluate --scope formal
run_step "$PYTHON" -u analyze_head_mixing.py --config "$CONFIG"
printf '[%s] ALL 8B-TO-4B FULL36-HEAD8 EXPERIMENTS COMPLETED\n' "$(date '+%Y-%m-%d %H:%M:%S')"
