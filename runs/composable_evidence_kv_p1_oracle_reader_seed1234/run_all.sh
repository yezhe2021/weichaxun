#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
SENDER_MODEL="${SENDER_MODEL:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B}"
RECEIVER_MODEL="${RECEIVER_MODEL:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}"
TRAIN_DATA="${TRAIN_DATA:-/home/yezhe/数据集/HotpotQA/raw/hotpot_train_v1.1.json}"
EVAL_DATA="${EVAL_DATA:-/home/yezhe/数据集/HotpotQA/raw/hotpot_dev_distractor_v1.json}"
P0_ROOT="${P0_ROOT:-${PROJECT}/runs/composable_evidence_kv_p0_hotpotqa_seed1234/results}"
TRAIN_CACHE="${TRAIN_CACHE:-${ROOT}/cache/train_oracle_slots.pt}"
EVAL_CACHE="${EVAL_CACHE:-${ROOT}/cache/eval_oracle_slots.pt}"
TRAIN_OUT="${TRAIN_OUT:-${ROOT}/train}"
EVAL_OUT="${EVAL_OUT:-${ROOT}/eval}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-1024}"
EVAL_SAMPLES="${EVAL_SAMPLES:-128}"
EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

case "${1:-help}" in
  prepare-train)
    "${PY}" "${ROOT}/prepare_oracle_slots.py" \
      --sender-model "${SENDER_MODEL}" \
      --data "${TRAIN_DATA}" \
      --out "${TRAIN_CACHE}" \
      --max-samples "${TRAIN_SAMPLES}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  prepare-eval)
    "${PY}" "${ROOT}/prepare_oracle_slots.py" \
      --sender-model "${SENDER_MODEL}" \
      --data "${EVAL_DATA}" \
      --manifest "${P0_ROOT}/manifest.jsonl" \
      --out "${EVAL_CACHE}" \
      --max-samples "${EVAL_SAMPLES}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  train)
    "${PY}" "${ROOT}/train_p1.py" \
      --receiver-model "${RECEIVER_MODEL}" \
      --train-cache "${TRAIN_CACHE}" \
      --out "${TRAIN_OUT}" \
      --epochs "${EPOCHS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  eval)
    "${PY}" "${ROOT}/eval_p1.py" \
      --receiver-model "${RECEIVER_MODEL}" \
      --eval-cache "${EVAL_CACHE}" \
      --checkpoint "${TRAIN_OUT}/checkpoint_latest.pt" \
      --out "${EVAL_OUT}" \
      --max-samples "${EVAL_SAMPLES}" \
      --p0-summary "${P0_ROOT}/summary.json" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  *)
    cat <<'USAGE'
Usage, in order:
  bash run_all.sh prepare-train
  bash run_all.sh prepare-eval
  bash run_all.sh train
  bash run_all.sh eval

This script intentionally has no automatic all-in-one target. Inspect each stage's
outputs before starting the next one.
USAGE
    ;;
esac
