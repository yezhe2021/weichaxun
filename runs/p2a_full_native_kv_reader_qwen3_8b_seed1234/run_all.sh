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
A1_CACHE_ROOT="${A1_CACHE_ROOT:-${ROOT}/cache_native_kv_a1}"
A1_TRAIN_OUT="${A1_TRAIN_OUT:-${ROOT}/train_a1_rank32}"
A1_DIAG_OUT="${A1_DIAG_OUT:-${ROOT}/diagnose_a1_rank32}"
A1_EVAL_OUT="${A1_EVAL_OUT:-${ROOT}/eval_a1_rank32}"
TRAIN_PAIRS="${TRAIN_PAIRS:-64}"
TEST_PAIRS="${TEST_PAIRS:-16}"
EVAL_PAIRS="${EVAL_PAIRS:-16}"
EPOCHS="${EPOCHS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
A1_EPOCHS="${A1_EPOCHS:-2}"
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
  a1-cache-train)
    "${PY}" "${ROOT}/cache_pre_rope_native_kv.py" \
      --model "${MODEL}" \
      --data "${DATA_ROOT}/train.jsonl" \
      --out "${A1_CACHE_ROOT}/train" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  a1-cache-test)
    "${PY}" "${ROOT}/cache_pre_rope_native_kv.py" \
      --model "${MODEL}" \
      --data "${DATA_ROOT}/test.jsonl" \
      --out "${A1_CACHE_ROOT}/test" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  a1-train)
    "${PY}" "${ROOT}/train_native_kv_reader_a1.py" \
      --model "${MODEL}" \
      --train-index "${A1_CACHE_ROOT}/train/index.json" \
      --out "${A1_TRAIN_OUT}" \
      --max-pairs "${TRAIN_PAIRS}" \
      --epochs "${A1_EPOCHS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  a1-diagnose)
    "${PY}" "${ROOT}/diagnose_native_kv_reader_a1.py" \
      --model "${MODEL}" \
      --test-index "${A1_CACHE_ROOT}/test/index.json" \
      --checkpoint "${A1_TRAIN_OUT}/checkpoint_latest.pt" \
      --out "${A1_DIAG_OUT}" \
      --max-pairs "${EVAL_PAIRS}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  a1-eval)
    "${PY}" "${ROOT}/eval_native_kv_reader_a1.py" \
      --model "${MODEL}" \
      --test-index "${A1_CACHE_ROOT}/test/index.json" \
      --checkpoint "${A1_TRAIN_OUT}/checkpoint_latest.pt" \
      --out "${A1_EVAL_OUT}" \
      --max-pairs "${EVAL_PAIRS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}"
    ;;
  a1)
    bash "${ROOT}/run_all.sh" generate-data
    bash "${ROOT}/run_all.sh" text-gate
    bash "${ROOT}/run_all.sh" a1-cache-train
    bash "${ROOT}/run_all.sh" a1-cache-test
    bash "${ROOT}/run_all.sh" a1-train
    bash "${ROOT}/run_all.sh" a1-diagnose
    bash "${ROOT}/run_all.sh" a1-eval
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

Complete P2-A1 (rank-32 Reader, paired diagnosis, free-running):
  bash run_all.sh a1

P2-A1 individual stages:
  bash run_all.sh a1-cache-train
  bash run_all.sh a1-cache-test
  bash run_all.sh a1-train
  bash run_all.sh a1-diagnose
  bash run_all.sh a1-eval

Quick end-to-end smoke run:
  bash run_all.sh smoke
USAGE
    ;;
esac
