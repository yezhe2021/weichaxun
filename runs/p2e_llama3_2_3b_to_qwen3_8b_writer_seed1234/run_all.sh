#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
SENDER_MODEL="${SENDER_MODEL:-/home/yezhe/all_models/models/LLM-Research/Llama-3___2-3B-Instruct}"
RECEIVER_MODEL="${RECEIVER_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-8B}"
P2A2_ROOT="${P2A2_ROOT:-${PROJECT}/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234}"
P2C2_ROOT="${P2C2_ROOT:-${PROJECT}/runs/p2c2_enhanced_global_sparse_writer_qwen3_1_7b_to_8b_seed1234}"
P2D_ROOT="${P2D_ROOT:-${PROJECT}/runs/p2d_sender_answerability_demo_qwen3_1_7b_seed1234}"
TEACHER_CACHE_ROOT="${TEACHER_CACHE_ROOT:-${P2A2_ROOT}/cache_native_kv_pairs}"
SENDER_CACHE_ROOT="${SENDER_CACHE_ROOT:-${ROOT}/cache_llama3_2_3b_native_kv_pairs}"
READER_CHECKPOINT="${READER_CHECKPOINT:-${P2A2_ROOT}/train/query_only/checkpoint_latest.pt}"
TEACHER_K_RMS="${TEACHER_K_RMS:-${P2C2_ROOT}/teacher_stats/teacher_k_rms.pt}"
TRAIN_DATA="${TRAIN_DATA:-${P2A2_ROOT}/data/train.jsonl}"
TEST_DATA="${TEST_DATA:-${P2A2_ROOT}/data/test.jsonl}"
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
VARIANTS=(task_only shared_span_relation)

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

require_file() {
  [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }
}

audit() {
  require_file "${READER_CHECKPOINT}"
  require_file "${TEACHER_K_RMS}"
  require_file "${TEACHER_CACHE_ROOT}/train/index.json"
  require_file "${TEACHER_CACHE_ROOT}/test/index.json"
  "${PY}" -m py_compile \
    "${ROOT}/cache_llama_native_kv.py" \
    "${ROOT}/p2e_writer.py" \
    "${ROOT}/p2e_structure.py" \
    "${ROOT}/train_p2e_writer.py" \
    "${ROOT}/eval_p2e_writer.py" \
    "${ROOT}/summarize_p2e.py"
  "${PY}" "${ROOT}/smoke_p2e.py"
  "${PY}" "${ROOT}/audit_p2e.py" \
    --sender-model "${SENDER_MODEL}" \
    --receiver-model "${RECEIVER_MODEL}" \
    --out "${ROOT}/model_compatibility_audit.json"
}

cache_split() {
  local split="$1"
  local data="$2"
  local pairs="$3"
  local out="${SENDER_CACHE_ROOT}/${split}"
  if [[ -f "${out}/CACHE_SUCCESS.json" ]]; then
    echo "Skipping completed Llama cache: ${split}"
    return
  fi
  "${PY}" "${ROOT}/cache_llama_native_kv.py" \
    --model "${SENDER_MODEL}" \
    --qwen-tokenizer "${RECEIVER_MODEL}" \
    --data "${data}" \
    --teacher-index "${TEACHER_CACHE_ROOT}/${split}/index.json" \
    --out "${out}" \
    --max-pairs "${pairs}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

train_variant() {
  local variant="$1"
  local out="${TRAIN_ROOT}/${variant}"
  if [[ -f "${out}/TRAIN_SUCCESS.json" ]]; then
    echo "Skipping completed training: ${variant}"
    return
  fi
  "${PY}" "${ROOT}/train_p2e_writer.py" \
    --receiver-model "${RECEIVER_MODEL}" \
    --reader-checkpoint "${READER_CHECKPOINT}" \
    --sender-index "${SENDER_CACHE_ROOT}/train/index.json" \
    --teacher-index "${TEACHER_CACHE_ROOT}/train/index.json" \
    --teacher-k-rms "${TEACHER_K_RMS}" \
    --out "${out}" \
    --variant "${variant}" \
    --max-pairs "${TRAIN_PAIRS}" \
    --epochs "${EPOCHS}" \
    --top-k 6 \
    --adapter-rank 32 \
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
  "${PY}" "${ROOT}/eval_p2e_writer.py" \
    --receiver-model "${RECEIVER_MODEL}" \
    --reader-checkpoint "${READER_CHECKPOINT}" \
    --writer-checkpoint "${TRAIN_ROOT}/${variant}/checkpoint_latest.pt" \
    --sender-index "${SENDER_CACHE_ROOT}/test/index.json" \
    --teacher-index "${TEACHER_CACHE_ROOT}/test/index.json" \
    --out "${out}" \
    --max-pairs "${EVAL_PAIRS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

summarize() {
  local gate="${P2D_ROOT}/llama_prompt_gate/SUCCESS.json"
  "${PY}" "${ROOT}/summarize_p2e.py" \
    --eval-root "${EVAL_ROOT}" \
    --llama-gate-reference "${gate}" \
    --out "${SUMMARY_ROOT}"
}

run_all() {
  audit
  cache_split train "${TRAIN_DATA}" "${TRAIN_PAIRS}"
  cache_split test "${TEST_DATA}" "${EVAL_PAIRS}"
  for variant in "${VARIANTS[@]}"; do
    train_variant "${variant}"
    eval_variant "${variant}"
  done
  summarize
}

case "${1:-help}" in
  audit) audit ;;
  cache-train) cache_split train "${TRAIN_DATA}" "${TRAIN_PAIRS}" ;;
  cache-test) cache_split test "${TEST_DATA}" "${EVAL_PAIRS}" ;;
  train-*) train_variant "${1#train-}" ;;
  eval-*) eval_variant "${1#eval-}" ;;
  summarize) summarize ;;
  all) run_all ;;
  status)
    [[ -f "${SENDER_CACHE_ROOT}/train/CACHE_SUCCESS.json" ]] && echo "cache_train=complete" || echo "cache_train=pending"
    [[ -f "${SENDER_CACHE_ROOT}/test/CACHE_SUCCESS.json" ]] && echo "cache_test=complete" || echo "cache_test=pending"
    for variant in "${VARIANTS[@]}"; do
      [[ -f "${TRAIN_ROOT}/${variant}/TRAIN_SUCCESS.json" ]] && train=complete || train=pending
      [[ -f "${EVAL_ROOT}/${variant}/SUCCESS.json" ]] && eval=complete || eval=pending
      echo "${variant}: train=${train} eval=${eval}"
    done
    ;;
  *)
    echo "Usage: bash run_all.sh {audit|cache-train|cache-test|train-VARIANT|eval-VARIANT|summarize|all|status}"
    ;;
esac
