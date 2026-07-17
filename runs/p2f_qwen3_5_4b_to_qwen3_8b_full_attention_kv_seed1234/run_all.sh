#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
SENDER_MODEL="${SENDER_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3___5-4B}"
RECEIVER_MODEL="${RECEIVER_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-8B}"
P2A2_ROOT="${P2A2_ROOT:-${PROJECT}/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234}"
P2C2_ROOT="${P2C2_ROOT:-${PROJECT}/runs/p2c2_enhanced_global_sparse_writer_qwen3_1_7b_to_8b_seed1234}"
TEACHER_CACHE_ROOT="${TEACHER_CACHE_ROOT:-${P2A2_ROOT}/cache_native_kv_pairs}"
READER_CHECKPOINT="${READER_CHECKPOINT:-${P2A2_ROOT}/train/query_only/checkpoint_latest.pt}"
TEACHER_K_RMS="${TEACHER_K_RMS:-${P2C2_ROOT}/teacher_stats/teacher_k_rms.pt}"
TRAIN_DATA="${TRAIN_DATA:-${P2A2_ROOT}/data/train.jsonl}"
TEST_DATA="${TEST_DATA:-${P2A2_ROOT}/data/test.jsonl}"
SENDER_CACHE_ROOT="${SENDER_CACHE_ROOT:-${ROOT}/cache_qwen35_full_attention_kv}"
GATE_ROOT="${GATE_ROOT:-${ROOT}/sender_gate}"
TRAIN_ROOT="${TRAIN_ROOT:-${ROOT}/train}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/eval}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${ROOT}/comparison}"
TRAIN_PAIRS="${TRAIN_PAIRS:-512}"
EVAL_PAIRS="${EVAL_PAIRS:-64}"
EPOCHS="${EPOCHS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
VARIANTS=(matched_task_only reader_aligned)

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

require_file() {
  [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }
}

audit() {
  require_file "${SENDER_MODEL}/config.json"
  require_file "${RECEIVER_MODEL}/config.json"
  require_file "${READER_CHECKPOINT}"
  require_file "${TEACHER_K_RMS}"
  require_file "${TEACHER_CACHE_ROOT}/train/index.json"
  require_file "${TEACHER_CACHE_ROOT}/test/index.json"
  "${PY}" -m py_compile \
    "${ROOT}/cache_qwen35_native_kv.py" \
    "${ROOT}/eval_qwen35_sender_gate.py" \
    "${ROOT}/audit_p2f.py" \
    "${ROOT}/smoke_qwen35_runtime.py" \
    "${ROOT}/p2e_writer.py" \
    "${ROOT}/p2e_structure.py" \
    "${ROOT}/train_p2f_writer.py" \
    "${ROOT}/eval_p2f_writer.py" \
    "${ROOT}/summarize_p2f.py"
  "${PY}" "${ROOT}/smoke_p2e.py"
  "${PY}" "${ROOT}/audit_p2f.py" \
    --sender-model "${SENDER_MODEL}" \
    --receiver-model "${RECEIVER_MODEL}" \
    --out "${ROOT}/model_compatibility_audit.json"
}

runtime_smoke() {
  "${PY}" "${ROOT}/smoke_qwen35_runtime.py" \
    --model "${SENDER_MODEL}" --device "${DEVICE}" --dtype "${DTYPE}"
}

sender_gate() {
  [[ -f "${GATE_ROOT}/SUCCESS.json" ]] && return
  "${PY}" "${ROOT}/eval_qwen35_sender_gate.py" \
    --model "${SENDER_MODEL}" --data "${TEST_DATA}" --out "${GATE_ROOT}" \
    --max-pairs "${EVAL_PAIRS}" --max-new-tokens "${MAX_NEW_TOKENS}" \
    --device "${DEVICE}" --dtype "${DTYPE}"
}

cache_split() {
  local split="$1" data="$2" pairs="$3"
  local out="${SENDER_CACHE_ROOT}/${split}"
  [[ -f "${out}/CACHE_SUCCESS.json" ]] && return
  "${PY}" "${ROOT}/cache_qwen35_native_kv.py" \
    --model "${SENDER_MODEL}" --qwen-tokenizer "${RECEIVER_MODEL}" \
    --data "${data}" --teacher-index "${TEACHER_CACHE_ROOT}/${split}/index.json" \
    --out "${out}" --max-pairs "${pairs}" \
    --device "${DEVICE}" --dtype "${DTYPE}"
}

train_variant() {
  local variant="$1" out="${TRAIN_ROOT}/$1"
  [[ -f "${out}/TRAIN_SUCCESS.json" ]] && return
  local route_weight=0.005 readout_weight=0.005 mass_weight=0.005 aux_every=8
  if [[ "${variant}" == "reader_aligned" ]]; then
    route_weight=0.05
    readout_weight=0.10
    mass_weight=0.05
    aux_every=1
  fi
  "${PY}" "${ROOT}/train_p2f_writer.py" \
    --receiver-model "${RECEIVER_MODEL}" --reader-checkpoint "${READER_CHECKPOINT}" \
    --sender-index "${SENDER_CACHE_ROOT}/train/index.json" \
    --teacher-index "${TEACHER_CACHE_ROOT}/train/index.json" \
    --teacher-k-rms "${TEACHER_K_RMS}" --out "${out}" --variant "${variant}" \
    --max-pairs "${TRAIN_PAIRS}" --epochs "${EPOCHS}" --top-k 6 --adapter-rank 32 \
    --route-weight "${route_weight}" --readout-weight "${readout_weight}" \
    --attention-mass-weight "${mass_weight}" --aux-every "${aux_every}" \
    --seed "${SEED}" --device "${DEVICE}" --dtype "${DTYPE}"
}

eval_variant() {
  local variant="$1" out="${EVAL_ROOT}/$1"
  require_file "${TRAIN_ROOT}/${variant}/checkpoint_latest.pt"
  [[ -f "${out}/SUCCESS.json" ]] && return
  "${PY}" "${ROOT}/eval_p2f_writer.py" \
    --receiver-model "${RECEIVER_MODEL}" --reader-checkpoint "${READER_CHECKPOINT}" \
    --writer-checkpoint "${TRAIN_ROOT}/${variant}/checkpoint_latest.pt" \
    --sender-index "${SENDER_CACHE_ROOT}/test/index.json" \
    --teacher-index "${TEACHER_CACHE_ROOT}/test/index.json" --out "${out}" \
    --max-pairs "${EVAL_PAIRS}" --max-new-tokens "${MAX_NEW_TOKENS}" \
    --device "${DEVICE}" --dtype "${DTYPE}"
}

summarize() {
  "${PY}" "${ROOT}/summarize_p2f.py" \
    --eval-root "${EVAL_ROOT}" --sender-gate "${GATE_ROOT}/SUCCESS.json" \
    --out "${SUMMARY_ROOT}"
}

case "${1:-help}" in
  audit) audit ;;
  runtime-smoke) runtime_smoke ;;
  gate) sender_gate ;;
  cache-train) cache_split train "${TRAIN_DATA}" "${TRAIN_PAIRS}" ;;
  cache-test) cache_split test "${TEST_DATA}" "${EVAL_PAIRS}" ;;
  train-*) train_variant "${1#train-}" ;;
  eval-*) eval_variant "${1#eval-}" ;;
  summarize) summarize ;;
  all)
    audit
    sender_gate
    cache_split train "${TRAIN_DATA}" "${TRAIN_PAIRS}"
    cache_split test "${TEST_DATA}" "${EVAL_PAIRS}"
    for variant in "${VARIANTS[@]}"; do
      train_variant "${variant}"
      eval_variant "${variant}"
    done
    summarize
    ;;
  status)
    [[ -f "${GATE_ROOT}/SUCCESS.json" ]] && echo gate=complete || echo gate=pending
    [[ -f "${SENDER_CACHE_ROOT}/train/CACHE_SUCCESS.json" ]] && echo cache_train=complete || echo cache_train=pending
    [[ -f "${SENDER_CACHE_ROOT}/test/CACHE_SUCCESS.json" ]] && echo cache_test=complete || echo cache_test=pending
    for variant in "${VARIANTS[@]}"; do
      [[ -f "${TRAIN_ROOT}/${variant}/TRAIN_SUCCESS.json" ]] && train=complete || train=pending
      [[ -f "${EVAL_ROOT}/${variant}/SUCCESS.json" ]] && eval=complete || eval=pending
      echo "${variant}: train=${train} eval=${eval}"
    done
    [[ -f "${SUMMARY_ROOT}/SUCCESS.json" ]] && echo comparison=complete || echo comparison=pending
    ;;
  *)
    echo "Usage: bash run_all.sh {audit|runtime-smoke|gate|cache-train|cache-test|train-VARIANT|eval-VARIANT|summarize|all|status}"
    ;;
esac
