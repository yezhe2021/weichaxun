#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
RECEIVER_MODEL="${RECEIVER_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-8B}"
P2A2_ROOT="${P2A2_ROOT:-${PROJECT}/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234}"
P2C1_ROOT="${P2C1_ROOT:-${PROJECT}/runs/p2c1_structural_writer_qwen3_1_7b_to_8b_seed1234}"
P2C2_ROOT="${P2C2_ROOT:-${PROJECT}/runs/p2c2_enhanced_global_sparse_writer_qwen3_1_7b_to_8b_seed1234}"
TEACHER_CACHE_ROOT="${TEACHER_CACHE_ROOT:-${P2A2_ROOT}/cache_native_kv_pairs}"
SENDER_CACHE_ROOT="${SENDER_CACHE_ROOT:-${P2C1_ROOT}/cache_qwen3_1_7b_native_kv_pairs}"
READER_CHECKPOINT="${READER_CHECKPOINT:-${P2A2_ROOT}/train/query_only/checkpoint_latest.pt}"
TEACHER_K_RMS="${TEACHER_K_RMS:-${P2C2_ROOT}/teacher_stats/teacher_k_rms.pt}"
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
VARIANTS=(task_only shared_routing binding_relation shared_routing_relation)

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

audit() {
  require_file "${READER_CHECKPOINT}"
  require_file "${TEACHER_K_RMS}"
  require_file "${SENDER_CACHE_ROOT}/train/index.json"
  require_file "${SENDER_CACHE_ROOT}/test/index.json"
  require_file "${TEACHER_CACHE_ROOT}/train/index.json"
  require_file "${TEACHER_CACHE_ROOT}/test/index.json"
  "${PY}" -m py_compile \
    "${ROOT}/p2c3_writer.py" \
    "${ROOT}/p2c3_structure.py" \
    "${ROOT}/train_p2c3_writer.py" \
    "${ROOT}/eval_p2c3_writer.py" \
    "${ROOT}/summarize_p2c3.py"
  "${PY}" "${ROOT}/smoke_p2c3.py"
}

train_variant() {
  local variant="$1"
  local out="${TRAIN_ROOT}/${variant}"
  if [[ -f "${out}/TRAIN_SUCCESS.json" ]]; then
    echo "Skipping completed training: ${variant}"
    return
  fi
  "${PY}" "${ROOT}/train_p2c3_writer.py" \
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
  "${PY}" "${ROOT}/eval_p2c3_writer.py" \
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
  "${PY}" "${ROOT}/summarize_p2c3.py" \
    --eval-root "${EVAL_ROOT}" \
    --p2c1-reference "${P2C1_ROOT}/eval_writer_qwen3_1_7b/SUCCESS.json" \
    --p2c2-reference "${P2C2_ROOT}/eval/full_staged/SUCCESS.json" \
    --out "${SUMMARY_ROOT}"
}

case "${1:-help}" in
  audit)
    audit
    ;;
  train-*)
    train_variant "${1#train-}"
    ;;
  eval-*)
    eval_variant "${1#eval-}"
    ;;
  summarize)
    summarize
    ;;
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
    cat <<'USAGE'
P2-C3 structure-preserving Writer ablations:
  bash run_all.sh all
  bash run_all.sh status
  bash run_all.sh audit
  bash run_all.sh train-task_only
  bash run_all.sh eval-task_only
  bash run_all.sh summarize
USAGE
    ;;
esac
