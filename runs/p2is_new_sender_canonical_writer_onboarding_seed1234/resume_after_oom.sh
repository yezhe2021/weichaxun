#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p2is_new_sender_canonical_writer_onboarding_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PY}" "${ROOT}/calibrate_writer.py" \
  --config full \
  --q4-index "${ROOT}/cache/qwen3_4b_native/train/index.json" \
  --old-index /home/yezhe/伪查询/runs/p2ir_token_preserving_canonical_reader_onboarding_seed1234/cache/canonical/train/index.json \
  --ridge "${ROOT}/ridge/q4_to_old_canonical.pt" \
  --init-writer "${ROOT}/imitation/train/checkpoint_best.pt" \
  --teacher4 "${ROOT}/cache/teacher/qwen3_4b/index.json" \
  --teacher35 "${ROOT}/cache/teacher/qwen3_5_4b/index.json" \
  --model4 /home/yezhe/all_models/models/Qwen/Qwen3-4B \
  --model35 /home/yezhe/all_models/models/Qwen/Qwen3___5-4B \
  --reader4 /home/yezhe/伪查询/runs/p2ir_token_preserving_canonical_reader_onboarding_seed1234/qwen3_4b/full/train/checkpoint_best.pt \
  --reader35 /home/yezhe/伪查询/runs/p2ir_token_preserving_canonical_reader_onboarding_seed1234/qwen3_5_4b/full/train/checkpoint_best.pt \
  --out "${ROOT}/calibration/full" \
  --train-pairs 448 \
  --epochs 3 \
  --chunk-pairs 16 \
  --lr 2e-4 \
  --seed 1234 \
  --device cuda \
  --resume "${ROOT}/calibration/full/checkpoint_latest.pt"

bash "${ROOT}/run_all.sh" all
