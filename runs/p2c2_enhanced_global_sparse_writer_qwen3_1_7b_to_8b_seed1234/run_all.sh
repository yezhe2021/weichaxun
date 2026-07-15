#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
SENDER_MODEL="${SENDER_MODEL:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}"
RECEIVER_MODEL="${RECEIVER_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-8B}"
P2A2_ROOT="${P2A2_ROOT:-${PROJECT}/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234}"
P2C1_ROOT="${P2C1_ROOT:-${PROJECT}/runs/p2c1_structural_writer_qwen3_1_7b_to_8b_seed1234}"
DATA_ROOT="${DATA_ROOT:-${P2A2_ROOT}/data}"
TEACHER_CACHE_ROOT="${TEACHER_CACHE_ROOT:-${P2A2_ROOT}/cache_native_kv_pairs}"
SENDER_CACHE_ROOT="${SENDER_CACHE_ROOT:-${P2C1_ROOT}/cache_qwen3_1_7b_native_kv_pairs}"
READER_CHECKPOINT="${READER_CHECKPOINT:-${P2A2_ROOT}/train/query_only/checkpoint_latest.pt}"
TEACHER_STATS_ROOT="${TEACHER_STATS_ROOT:-${ROOT}/teacher_stats}"
AUDIT_ROOT="${AUDIT_ROOT:-${ROOT}/audit}"
TRAIN_ROOT="${TRAIN_ROOT:-${ROOT}/train}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT}/eval}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${ROOT}/comparison}"
TRAIN_PAIRS="${TRAIN_PAIRS:-512}"
EVAL_PAIRS="${EVAL_PAIRS:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

train_variant() {
  local variant="$1"
  require_file "${SENDER_CACHE_ROOT}/train/index.json"
  require_file "${TEACHER_CACHE_ROOT}/train/index.json"
  require_file "${TEACHER_STATS_ROOT}/teacher_k_rms.pt"
  require_file "${READER_CHECKPOINT}"
  "${PY}" "${ROOT}/train_p2c2_writer.py" \
    --receiver-model "${RECEIVER_MODEL}" \
    --reader-checkpoint "${READER_CHECKPOINT}" \
    --sender-index "${SENDER_CACHE_ROOT}/train/index.json" \
    --teacher-index "${TEACHER_CACHE_ROOT}/train/index.json" \
    --teacher-k-rms "${TEACHER_STATS_ROOT}/teacher_k_rms.pt" \
    --out "${TRAIN_ROOT}/${variant}" \
    --variant "${variant}" \
    --max-pairs "${TRAIN_PAIRS}" \
    --top-k 6 \
    --adapter-rank 32 \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

eval_variant() {
  local variant="$1"
  require_file "${TRAIN_ROOT}/${variant}/checkpoint_latest.pt"
  require_file "${SENDER_CACHE_ROOT}/test/index.json"
  require_file "${TEACHER_CACHE_ROOT}/test/index.json"
  "${PY}" "${ROOT}/eval_p2c2_writer.py" \
    --receiver-model "${RECEIVER_MODEL}" \
    --reader-checkpoint "${READER_CHECKPOINT}" \
    --writer-checkpoint "${TRAIN_ROOT}/${variant}/checkpoint_latest.pt" \
    --sender-index "${SENDER_CACHE_ROOT}/test/index.json" \
    --teacher-index "${TEACHER_CACHE_ROOT}/test/index.json" \
    --out "${EVAL_ROOT}/${variant}" \
    --max-pairs "${EVAL_PAIRS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

case "${1:-help}" in
  audit)
    require_file "${DATA_ROOT}/train.jsonl"
    require_file "${READER_CHECKPOINT}"
    "${PY}" "${ROOT}/audit_p2c2.py" \
      --sender-model "${SENDER_MODEL}" \
      --receiver-model "${RECEIVER_MODEL}" \
      --data "${DATA_ROOT}/train.jsonl" \
      --reader-checkpoint "${READER_CHECKPOINT}" \
      --out "${AUDIT_ROOT}"
    ;;
  teacher-stats)
    require_file "${TEACHER_CACHE_ROOT}/train/index.json"
    "${PY}" "${ROOT}/compute_teacher_k_stats.py" \
      --teacher-index "${TEACHER_CACHE_ROOT}/train/index.json" \
      --out "${TEACHER_STATS_ROOT}" \
      --max-pairs "${TRAIN_PAIRS}"
    ;;
  train-global-only)
    train_variant global_only
    ;;
  train-global-head)
    train_variant global_head
    ;;
  train-full-staged)
    train_variant full_staged
    ;;
  eval-global-only)
    eval_variant global_only
    ;;
  eval-global-head)
    eval_variant global_head
    ;;
  eval-full-staged)
    eval_variant full_staged
    ;;
  summarize)
    "${PY}" "${ROOT}/summarize_p2c2.py" \
      --eval-root "${EVAL_ROOT}" \
      --current-p2c1 "${P2C1_ROOT}/eval_writer_qwen3_1_7b/SUCCESS.json" \
      --out "${SUMMARY_ROOT}"
    ;;
  all)
    bash "${ROOT}/run_all.sh" audit
    bash "${ROOT}/run_all.sh" teacher-stats
    bash "${ROOT}/run_all.sh" train-full-staged
    bash "${ROOT}/run_all.sh" eval-full-staged
    bash "${ROOT}/run_all.sh" summarize
    ;;
  all-ablations)
    bash "${ROOT}/run_all.sh" audit
    bash "${ROOT}/run_all.sh" teacher-stats
    for variant in global-only global-head full-staged; do
      bash "${ROOT}/run_all.sh" "train-${variant}"
      bash "${ROOT}/run_all.sh" "eval-${variant}"
    done
    bash "${ROOT}/run_all.sh" summarize
    ;;
  *)
    cat <<'USAGE'
P2-C2 default enhanced run:
  bash run_all.sh all

Optional full ablations:
  bash run_all.sh all-ablations

Individual variants:
  bash run_all.sh train-global-only
  bash run_all.sh train-global-head
  bash run_all.sh train-full-staged
  bash run_all.sh eval-global-only
  bash run_all.sh eval-global-head
  bash run_all.sh eval-full-staged
USAGE
    ;;
esac
