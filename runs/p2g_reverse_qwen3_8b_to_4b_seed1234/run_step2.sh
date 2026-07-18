#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
SENDER_MODEL="/home/yezhe/all_models/models/Qwen/Qwen3-8B"
RECEIVER_MODEL="/home/yezhe/all_models/models/Qwen/Qwen3-4B"
P2A2_ROOT="${PROJECT}/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234"
SENDER_CACHE_ROOT="${P2A2_ROOT}/cache_native_kv_pairs"
STEP1_ROOT="${ROOT}/step1_native_reader"
TEACHER_CACHE_ROOT="${STEP1_ROOT}/cache"
READER_CHECKPOINT="${STEP1_ROOT}/train/checkpoint_latest.pt"
NATIVE_GATE="${STEP1_ROOT}/eval/NATIVE_READER_GATE_FAILED.json"
STEP2_ROOT="${ROOT}/step2_8b_to_4b_writer"
TEACHER_STATS="${STEP2_ROOT}/teacher_stats"
TRAIN_ROOT="${STEP2_ROOT}/train"
EVAL_ROOT="${STEP2_ROOT}/eval"
SUMMARY_ROOT="${STEP2_ROOT}/comparison"
LOG_DIR="${ROOT}/logs"

TRAIN_PAIRS="${TRAIN_PAIRS:-512}"
EVAL_PAIRS="${EVAL_PAIRS:-64}"
EPOCHS="${EPOCHS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
VARIANTS=(matched_task_only reader_aligned)

export TOKENIZERS_PARALLELISM=false
mkdir -p "${LOG_DIR}"
cd "${PROJECT}"

require_file() {
  [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }
}

audit() {
  require_file "${SENDER_MODEL}/config.json"
  require_file "${RECEIVER_MODEL}/config.json"
  require_file "${READER_CHECKPOINT}"
  require_file "${NATIVE_GATE}"
  require_file "${SENDER_CACHE_ROOT}/train/index.json"
  require_file "${SENDER_CACHE_ROOT}/test/index.json"
  require_file "${TEACHER_CACHE_ROOT}/train/index.json"
  require_file "${TEACHER_CACHE_ROOT}/test/index.json"
  "${PY}" -m py_compile \
    "${ROOT}/p2a_common.py" \
    "${ROOT}/p2c1_writer.py" \
    "${ROOT}/p2e_writer.py" \
    "${ROOT}/p2e_structure.py" \
    "${ROOT}/train_p2c1_writer.py" \
    "${ROOT}/train_p2g2_writer.py" \
    "${ROOT}/eval_p2g2_writer.py" \
    "${ROOT}/compute_teacher_k_stats.py" \
    "${ROOT}/audit_p2g2.py" \
    "${ROOT}/summarize_p2g2.py"
  "${PY}" "${ROOT}/smoke_p2e.py"
  "${PY}" "${ROOT}/audit_p2g2.py" \
    --sender-model "${SENDER_MODEL}" --receiver-model "${RECEIVER_MODEL}" \
    --reader-checkpoint "${READER_CHECKPOINT}" \
    --sender-train-index "${SENDER_CACHE_ROOT}/train/index.json" \
    --sender-test-index "${SENDER_CACHE_ROOT}/test/index.json" \
    --teacher-train-index "${TEACHER_CACHE_ROOT}/train/index.json" \
    --teacher-test-index "${TEACHER_CACHE_ROOT}/test/index.json" \
    --native-gate "${NATIVE_GATE}" --out "${STEP2_ROOT}/AUDIT.json"
}

teacher_stats() {
  [[ -f "${TEACHER_STATS}/SUCCESS.json" ]] && return
  "${PY}" "${ROOT}/compute_teacher_k_stats.py" \
    --teacher-index "${TEACHER_CACHE_ROOT}/train/index.json" \
    --out "${TEACHER_STATS}" --max-pairs "${TRAIN_PAIRS}"
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
  "${PY}" "${ROOT}/train_p2g2_writer.py" \
    --receiver-model "${RECEIVER_MODEL}" --reader-checkpoint "${READER_CHECKPOINT}" \
    --sender-index "${SENDER_CACHE_ROOT}/train/index.json" \
    --teacher-index "${TEACHER_CACHE_ROOT}/train/index.json" \
    --teacher-k-rms "${TEACHER_STATS}/teacher_k_rms.pt" \
    --out "${out}" --variant "${variant}" \
    --max-pairs "${TRAIN_PAIRS}" --epochs "${EPOCHS}" \
    --top-k 6 --adapter-rank 32 --lr 2e-4 \
    --route-weight "${route_weight}" --readout-weight "${readout_weight}" \
    --attention-mass-weight "${mass_weight}" --aux-every "${aux_every}" \
    --seed "${SEED}" --device "${DEVICE}" --dtype "${DTYPE}"
}

eval_variant() {
  local variant="$1" out="${EVAL_ROOT}/$1"
  require_file "${TRAIN_ROOT}/${variant}/checkpoint_latest.pt"
  [[ -f "${out}/SUCCESS.json" ]] && return
  "${PY}" "${ROOT}/eval_p2g2_writer.py" \
    --receiver-model "${RECEIVER_MODEL}" --reader-checkpoint "${READER_CHECKPOINT}" \
    --writer-checkpoint "${TRAIN_ROOT}/${variant}/checkpoint_latest.pt" \
    --sender-index "${SENDER_CACHE_ROOT}/test/index.json" \
    --teacher-index "${TEACHER_CACHE_ROOT}/test/index.json" \
    --out "${out}" --max-pairs "${EVAL_PAIRS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" --device "${DEVICE}" --dtype "${DTYPE}"
}

summarize() {
  "${PY}" "${ROOT}/summarize_p2g2.py" \
    --eval-root "${EVAL_ROOT}" --native-gate "${NATIVE_GATE}" --out "${SUMMARY_ROOT}"
}

run_all() {
  audit
  teacher_stats
  for variant in "${VARIANTS[@]}"; do
    train_variant "${variant}"
    eval_variant "${variant}"
  done
  summarize
}

wait_for_cuda() {
  printf '[%s] Waiting for CUDA before starting P2-G2.\n' "$(date -Is)"
  until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
    >/dev/null 2>&1; do
    printf '[%s] CUDA is unavailable; retrying in 60 seconds.\n' "$(date -Is)"
    sleep 60
  done
  printf '[%s] CUDA is available; starting P2-G2.\n' "$(date -Is)"
  run_all
}

status() {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  pgrep -af 'train_p2g2_writer|eval_p2g2_writer|run_step2.sh wait-cuda' || true
  [[ -f "${STEP2_ROOT}/AUDIT.json" ]] && echo audit=complete || echo audit=pending
  [[ -f "${TEACHER_STATS}/SUCCESS.json" ]] && echo teacher_stats=complete || echo teacher_stats=pending
  for variant in "${VARIANTS[@]}"; do
    [[ -f "${TRAIN_ROOT}/${variant}/TRAIN_SUCCESS.json" ]] && train=complete || train=pending
    [[ -f "${EVAL_ROOT}/${variant}/SUCCESS.json" ]] && eval=complete || eval=pending
    echo "${variant}: train=${train} eval=${eval}"
  done
  [[ -f "${SUMMARY_ROOT}/SUCCESS.json" ]] && echo comparison=complete || echo comparison=pending
}

case "${1:-help}" in
  audit) audit ;;
  teacher-stats) teacher_stats ;;
  train-*) train_variant "${1#train-}" ;;
  eval-*) eval_variant "${1#eval-}" ;;
  summarize) summarize ;;
  all) run_all ;;
  wait-cuda) wait_for_cuda ;;
  status) status ;;
  *)
    echo "Usage: bash run_step2.sh {audit|teacher-stats|train-VARIANT|eval-VARIANT|summarize|all|wait-cuda|status}"
    exit 64
    ;;
esac
