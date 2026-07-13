#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
MODEL="${MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-8B}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/cache_native_kv}"
TEXT_GATE_OUT="${TEXT_GATE_OUT:-${ROOT}/text_gate}"
TRAIN_OUT="${TRAIN_OUT:-${ROOT}/train_gate_only}"
EVAL_OUT="${EVAL_OUT:-${ROOT}/eval_gate_only}"
TRAIN_PAIRS="${TRAIN_PAIRS:-64}"
TEST_PAIRS="${TEST_PAIRS:-16}"
EVAL_PAIRS="${EVAL_PAIRS:-16}"
EPOCHS="${EPOCHS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

case "${1:-help}" in
  generate-data)
    "${PY}" "${ROOT}/generate_counterfactual_data.py" \
      --model "${MODEL}" \
      --out "${DATA_ROOT}" \
      --train-pairs "${TRAIN_PAIRS}" \
      --test-pairs "${TEST_PAIRS}" \
      --seed "${SEED}"
    ;;
  text-gate)
    "${PY}" "${ROOT}/check_text_gate.py" \
      --model "${MODEL}" \
      --data "${DATA_ROOT}/test.jsonl" \
      --out "${TEXT_GATE_OUT}" \
      --max-pairs "${EVAL_PAIRS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  cache-train)
    "${PY}" "${ROOT}/cache_pre_rope_native_kv.py" \
      --model "${MODEL}" \
      --data "${DATA_ROOT}/train.jsonl" \
      --out "${CACHE_ROOT}/train" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  cache-test)
    "${PY}" "${ROOT}/cache_pre_rope_native_kv.py" \
      --model "${MODEL}" \
      --data "${DATA_ROOT}/test.jsonl" \
      --out "${CACHE_ROOT}/test" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  train)
    "${PY}" "${ROOT}/train_native_kv_reader.py" \
      --model "${MODEL}" \
      --train-index "${CACHE_ROOT}/train/index.json" \
      --out "${TRAIN_OUT}" \
      --max-pairs "${TRAIN_PAIRS}" \
      --epochs "${EPOCHS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  eval)
    "${PY}" "${ROOT}/eval_native_kv_reader.py" \
      --model "${MODEL}" \
      --test-index "${CACHE_ROOT}/test/index.json" \
      --checkpoint "${TRAIN_OUT}/checkpoint_latest.pt" \
      --out "${EVAL_OUT}" \
      --max-pairs "${EVAL_PAIRS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  smoke)
    bash "${ROOT}/run_all.sh" generate-data
    bash "${ROOT}/run_all.sh" text-gate
    bash "${ROOT}/run_all.sh" cache-train
    bash "${ROOT}/run_all.sh" cache-test
    bash "${ROOT}/run_all.sh" train
    bash "${ROOT}/run_all.sh" eval
    ;;
  *)
    cat <<'USAGE'
P2-A stages:
  bash run_all.sh generate-data
  bash run_all.sh text-gate
  bash run_all.sh cache-train
  bash run_all.sh cache-test
  bash run_all.sh train
  bash run_all.sh eval

Quick end-to-end smoke run:
  bash run_all.sh smoke
USAGE
    ;;
esac
