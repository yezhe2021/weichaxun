#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/重实现/mmlupro_b1_epoch_batch8_clip30_qwen3_1_7b_to_4b_seed1234"
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
run_step "$PYTHON" -u run_gpu_tests.py
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" gradient_audit --writer d2 --functional-mode final --checkpoint /home/yezhe/重实现/mmlupro_b1_scale1024_qwen3_1_7b_to_4b_seed1234/checkpoints/quick/d2/stage_a/best.pt
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" train --writer d2 --stage b --functional-mode final
run_step "$PYTHON" -u run_pipeline.py --config "$CONFIG" evaluate --scope formal

printf '[%s] ALL B1 EPOCH-BATCH8 EXPERIMENTS COMPLETED\n' "$(date '+%Y-%m-%d %H:%M:%S')"
