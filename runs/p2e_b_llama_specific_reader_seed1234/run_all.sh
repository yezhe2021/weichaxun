#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
RECEIVER_MODEL="${RECEIVER_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-8B}"
P2A2_ROOT="${P2A2_ROOT:-${PROJECT}/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234}"
P2E_ROOT="${P2E_ROOT:-${PROJECT}/runs/p2e_llama3_2_3b_to_qwen3_8b_writer_seed1234}"
SENDER_CACHE_ROOT="${SENDER_CACHE_ROOT:-${P2E_ROOT}/cache_llama3_2_3b_native_kv_pairs}"
NATIVE_CACHE_ROOT="${NATIVE_CACHE_ROOT:-${P2A2_ROOT}/cache_native_kv_pairs}"
NATIVE_READER_CHECKPOINT="${NATIVE_READER_CHECKPOINT:-${P2A2_ROOT}/train/query_only/checkpoint_latest.pt}"
TRAIN_ROOT="${TRAIN_ROOT:-${ROOT}/train}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/eval}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${ROOT}/comparison}"
TRAIN_PAIRS="${TRAIN_PAIRS:-512}"
EVAL_PAIRS="${EVAL_PAIRS:-64}"
EPOCHS="${EPOCHS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
VARIANTS=(minimal_reader routed_reader)

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

require_file() {
  [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }
}

audit() {
  require_file "${SENDER_CACHE_ROOT}/train/index.json"
  require_file "${SENDER_CACHE_ROOT}/test/index.json"
  require_file "${NATIVE_CACHE_ROOT}/test/index.json"
  require_file "${NATIVE_READER_CHECKPOINT}"
  "${PY}" -m py_compile \
    "${ROOT}/p2a_common.py" \
    "${ROOT}/llama_specific_reader.py" \
    "${ROOT}/train_llama_reader.py" \
    "${ROOT}/eval_llama_reader.py" \
    "${ROOT}/summarize_llama_readers.py" \
    "${ROOT}/smoke_llama_reader.py"
  "${PY}" "${ROOT}/smoke_llama_reader.py"
}

train_variant() {
  local variant="$1"
  local out="${TRAIN_ROOT}/${variant}"
  if [[ -f "${out}/TRAIN_SUCCESS.json" ]]; then
    echo "Skipping completed training: ${variant}"
    return
  fi
  "${PY}" "${ROOT}/train_llama_reader.py" \
    --receiver-model "${RECEIVER_MODEL}" \
    --sender-index "${SENDER_CACHE_ROOT}/train/index.json" \
    --out "${out}" \
    --variant "${variant}" \
    --max-pairs "${TRAIN_PAIRS}" \
    --epochs "${EPOCHS}" \
    --top-k 2 \
    --query-rank 32 \
    --output-rank 32 \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

eval_variant() {
  local variant="$1"
  local out="${EVAL_ROOT}/${variant}"
  require_file "${TRAIN_ROOT}/${variant}/checkpoint_latest.pt"
  if [[ -f "${out}/SUCCESS.json" ]]; then
    echo "Skipping completed evaluation: ${variant}"
    return
  fi
  "${PY}" "${ROOT}/eval_llama_reader.py" \
    --receiver-model "${RECEIVER_MODEL}" \
    --reader-checkpoint "${TRAIN_ROOT}/${variant}/checkpoint_latest.pt" \
    --native-reader-checkpoint "${NATIVE_READER_CHECKPOINT}" \
    --sender-index "${SENDER_CACHE_ROOT}/test/index.json" \
    --native-index "${NATIVE_CACHE_ROOT}/test/index.json" \
    --out "${out}" \
    --max-pairs "${EVAL_PAIRS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

summarize() {
  "${PY}" "${ROOT}/summarize_llama_readers.py" \
    --eval-root "${EVAL_ROOT}" \
    --out "${SUMMARY_ROOT}"
}

case "${1:-help}" in
  audit) audit ;;
  train-*) train_variant "${1#train-}" ;;
  eval-*) eval_variant "${1#eval-}" ;;
  summarize) summarize ;;
  all)
    audit
    for variant in "${VARIANTS[@]}"; do
      train_variant "${variant}"
      eval_variant "${variant}"
    done
    summarize
    ;;
  status)
    for variant in "${VARIANTS[@]}"; do
      [[ -f "${TRAIN_ROOT}/${variant}/TRAIN_SUCCESS.json" ]] && train=complete || train=pending
      [[ -f "${EVAL_ROOT}/${variant}/SUCCESS.json" ]] && eval=complete || eval=pending
      echo "${variant}: train=${train} eval=${eval}"
    done
    [[ -f "${SUMMARY_ROOT}/SUCCESS.json" ]] && echo "summary=complete" || echo "summary=pending"
    ;;
  *)
    echo "Usage: bash run_all.sh {audit|train-VARIANT|eval-VARIANT|summarize|all|status}"
    ;;
esac
