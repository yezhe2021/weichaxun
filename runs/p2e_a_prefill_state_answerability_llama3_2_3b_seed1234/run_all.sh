#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
MODEL="${MODEL:-/home/yezhe/all_models/models/LLM-Research/Llama-3___2-3B-Instruct}"
DATA_ROOT="${DATA_ROOT:-${PROJECT}/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234/data}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/cache_prefill_states}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/probe_results}"
TRAIN_PAIRS="${TRAIN_PAIRS:-512}"
TEST_PAIRS="${TEST_PAIRS:-64}"
SUMMARY_SLOTS="${SUMMARY_SLOTS:-16}"
LAYERS="${LAYERS:-auto}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

require_file() {
  [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }
}

audit() {
  require_file "${DATA_ROOT}/train.jsonl"
  require_file "${DATA_ROOT}/test.jsonl"
  "${PY}" -m py_compile \
    "${ROOT}/prefill_prompt.py" \
    "${ROOT}/cache_prefill_states.py" \
    "${ROOT}/prefill_probes.py" \
    "${ROOT}/train_eval_prefill_probes.py" \
    "${ROOT}/smoke_prefill_probes.py"
  "${PY}" "${ROOT}/smoke_prefill_probes.py"
}

cache_train() {
  if [[ -f "${CACHE_ROOT}/train/CACHE_SUCCESS.json" ]]; then
    echo "Skipping completed train cache"
    return
  fi
  "${PY}" "${ROOT}/cache_prefill_states.py" \
    --model "${MODEL}" \
    --data "${DATA_ROOT}/train.jsonl" \
    --out "${CACHE_ROOT}/train" \
    --max-pairs "${TRAIN_PAIRS}" \
    --conditions correct \
    --summary-slots "${SUMMARY_SLOTS}" \
    --layers "${LAYERS}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

cache_test() {
  if [[ -f "${CACHE_ROOT}/test/CACHE_SUCCESS.json" ]]; then
    echo "Skipping completed test cache"
    return
  fi
  "${PY}" "${ROOT}/cache_prefill_states.py" \
    --model "${MODEL}" \
    --data "${DATA_ROOT}/test.jsonl" \
    --out "${CACHE_ROOT}/test" \
    --max-pairs "${TEST_PAIRS}" \
    --conditions correct,question_only,a_only,b_only,answer_masked \
    --summary-slots "${SUMMARY_SLOTS}" \
    --layers "${LAYERS}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

train_eval() {
  require_file "${CACHE_ROOT}/train/index.json"
  require_file "${CACHE_ROOT}/test/index.json"
  if [[ -f "${RESULT_ROOT}/SUCCESS.json" ]]; then
    echo "Skipping completed probe training/evaluation"
    return
  fi
  "${PY}" "${ROOT}/train_eval_prefill_probes.py" \
    --train-index "${CACHE_ROOT}/train/index.json" \
    --test-index "${CACHE_ROOT}/test/index.json" \
    --out "${RESULT_ROOT}" \
    --configs all \
    --seed "${SEED}" \
    --device "${DEVICE}"
}

case "${1:-help}" in
  audit) audit ;;
  cache-train) cache_train ;;
  cache-test) cache_test ;;
  train-eval) train_eval ;;
  all)
    audit
    cache_train
    cache_test
    train_eval
    ;;
  status)
    [[ -f "${CACHE_ROOT}/train/CACHE_SUCCESS.json" ]] && echo "cache_train=complete" || echo "cache_train=pending"
    [[ -f "${CACHE_ROOT}/test/CACHE_SUCCESS.json" ]] && echo "cache_test=complete" || echo "cache_test=pending"
    [[ -f "${RESULT_ROOT}/SUCCESS.json" ]] && echo "probes=complete" || echo "probes=pending"
    ;;
  *)
    echo "Usage: bash run_all.sh {audit|cache-train|cache-test|train-eval|all|status}"
    ;;
esac
