#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
SENDER_MODEL="${SENDER_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-8B}"
RECEIVER4_MODEL="${RECEIVER4_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-4B}"
RECEIVER35_MODEL="${RECEIVER35_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3___5-4B}"
P2A2_ROOT="${P2A2_ROOT:-${PROJECT}/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234}"
P2G1_ROOT="${P2G1_ROOT:-${PROJECT}/runs/p2g1_native_reader_qwen3_4b_seed1234}"
P2G2_ROOT="${P2G2_ROOT:-${PROJECT}/runs/p2g2_reverse_writer_qwen3_8b_to_4b_seed1234}"
TRAIN_DATA="${TRAIN_DATA:-${P2A2_ROOT}/data/train.jsonl}"
TEST_DATA="${TEST_DATA:-${P2A2_ROOT}/data/test.jsonl}"
SENDER_TRAIN_INDEX="${SENDER_TRAIN_INDEX:-${P2A2_ROOT}/cache_native_kv_pairs/train/index.json}"
SENDER_TEST_INDEX="${SENDER_TEST_INDEX:-${P2A2_ROOT}/cache_native_kv_pairs/test/index.json}"
NATIVE4_TRAIN_INDEX="${NATIVE4_TRAIN_INDEX:-${P2G1_ROOT}/cache/train/index.json}"
NATIVE4_READER="${NATIVE4_READER:-${P2G1_ROOT}/train/checkpoint_latest.pt}"
NATIVE_READER_CODE="${NATIVE_READER_CODE:-${P2G2_ROOT}}"

TRAIN_PAIRS="${TRAIN_PAIRS:-448}"
TEST_PAIRS="${TEST_PAIRS:-64}"
MOTHER_CYCLES="${MOTHER_CYCLES:-2}"
FORK_EPOCHS="${FORK_EPOCHS:-2}"
ORACLE_EPOCHS="${ORACLE_EPOCHS:-1}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
RECEIVER35_TRAIN_DTYPE="${RECEIVER35_TRAIN_DTYPE:-float32}"

TEACHER4="${ROOT}/cache/teacher_qwen3_4b/train"
TEACHER35="${ROOT}/cache/teacher_qwen3_5_4b/train"
CANONICAL_TRAIN="${ROOT}/cache/canonical/train"
CANONICAL_TEST="${ROOT}/cache/canonical/test"
MISMATCH_DATA="${ROOT}/data/test_true_mismatch.jsonl"
MISMATCH_NATIVE="${ROOT}/cache/true_mismatch_native/test"
MISMATCH_CANONICAL="${ROOT}/cache/true_mismatch_canonical/test"
MOTHER_ROOT="${ROOT}/mother"
FORK_ROOT="${ROOT}/forks"
ORACLE_ROOT="${ROOT}/oracle"
EVAL_ROOT="${ROOT}/eval"

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

require_file() {
  [[ -f "$1" ]] || { printf 'Required file is missing: %s\n' "$1" >&2; exit 1; }
}

audit() {
  require_file "${SENDER_MODEL}/config.json"
  require_file "${RECEIVER4_MODEL}/config.json"
  require_file "${RECEIVER35_MODEL}/config.json"
  require_file "${SENDER_TRAIN_INDEX}"
  require_file "${SENDER_TEST_INDEX}"
  require_file "${NATIVE4_TRAIN_INDEX}"
  require_file "${NATIVE4_READER}"
  "${PY}" -m py_compile "${ROOT}"/*.py
  "${PY}" "${ROOT}/audit_p2i.py" \
    --sender-model "${SENDER_MODEL}" \
    --receiver4-model "${RECEIVER4_MODEL}" \
    --receiver35-model "${RECEIVER35_MODEL}" \
    --train-data "${TRAIN_DATA}" --test-data "${TEST_DATA}" \
    --sender-train-index "${SENDER_TRAIN_INDEX}" \
    --sender-test-index "${SENDER_TEST_INDEX}" --out "${ROOT}/AUDIT.json"
}

runtime_smoke() {
  "${PY}" "${ROOT}/smoke_runtime.py" \
    --receiver-model "${RECEIVER4_MODEL}" --sender-index "${SENDER_TEST_INDEX}" \
    --device "${DEVICE}" --dtype "${DTYPE}"
  "${PY}" "${ROOT}/smoke_runtime.py" \
    --receiver-model "${RECEIVER35_MODEL}" --sender-index "${SENDER_TEST_INDEX}" \
    --device "${DEVICE}" --dtype "${DTYPE}"
}

teacher4() {
  [[ -f "${TEACHER4}/CACHE_SUCCESS.json" ]] && return
  "${PY}" "${ROOT}/cache_teacher_features.py" \
    --receiver-model "${RECEIVER4_MODEL}" --rows-index "${SENDER_TRAIN_INDEX}" \
    --teacher-kind native_reader --native-index "${NATIVE4_TRAIN_INDEX}" \
    --native-reader-checkpoint "${NATIVE4_READER}" \
    --native-reader-root "${NATIVE_READER_CODE}" --out "${TEACHER4}" \
    --max-pairs "${TRAIN_PAIRS}" --device "${DEVICE}" --dtype "${DTYPE}"
}

teacher35() {
  [[ -f "${TEACHER35}/CACHE_SUCCESS.json" ]] && return
  "${PY}" "${ROOT}/cache_teacher_features.py" \
    --receiver-model "${RECEIVER35_MODEL}" --rows-index "${SENDER_TRAIN_INDEX}" \
    --teacher-kind full_text --out "${TEACHER35}" --max-pairs "${TRAIN_PAIRS}" \
    --device "${DEVICE}" --dtype "${DTYPE}"
}

train_branch() {
  local receiver_name="$1" receiver_model="$2" teacher_index="$3" resume="$4" out="$5" train_dtype="$6"
  [[ -f "${out}/TRAIN_SUCCESS.json" ]] && return
  local resume_args=()
  [[ -n "${resume}" ]] && resume_args=(--resume "${resume}")
  "${PY}" "${ROOT}/train_canonical_branch.py" \
    --receiver-name "${receiver_name}" --receiver-model "${receiver_model}" \
    --sender-index "${SENDER_TRAIN_INDEX}" --teacher-index "${teacher_index}" \
    --out "${out}" --max-pairs "${TRAIN_PAIRS}" --epochs 1 \
    --seed "${SEED}" --device "${DEVICE}" --dtype "${train_dtype}" "${resume_args[@]}"
}

mother() {
  local resume="" cycle q4_out q35_out
  for cycle in $(seq 1 "${MOTHER_CYCLES}"); do
    q4_out="${MOTHER_ROOT}/cycle_${cycle}_qwen3_4b"
    train_branch qwen3_4b "${RECEIVER4_MODEL}" "${TEACHER4}/index.json" "${resume}" "${q4_out}" "${DTYPE}"
    resume="${q4_out}/checkpoint_latest.pt"
    q35_out="${MOTHER_ROOT}/cycle_${cycle}_qwen3_5_4b"
    train_branch qwen3_5_4b "${RECEIVER35_MODEL}" "${TEACHER35}/index.json" "${resume}" "${q35_out}" "${RECEIVER35_TRAIN_DTYPE}"
    resume="${q35_out}/checkpoint_latest.pt"
  done
  mkdir -p "${MOTHER_ROOT}"
  printf '%s\n' "${resume}" > "${MOTHER_ROOT}/FINAL_CHECKPOINT.txt"
}

mother_checkpoint() {
  require_file "${MOTHER_ROOT}/FINAL_CHECKPOINT.txt"
  local checkpoint
  checkpoint="$(cat "${MOTHER_ROOT}/FINAL_CHECKPOINT.txt")"
  require_file "${checkpoint}"
  printf '%s' "${checkpoint}"
}

cache_canonical_split() {
  local split="$1" source_index="$2" out="$3" pairs="$4" checkpoint="$5"
  [[ -f "${out}/CACHE_SUCCESS.json" ]] && return
  "${PY}" "${ROOT}/cache_canonical_evidence.py" \
    --sender-index "${source_index}" --checkpoint "${checkpoint}" --out "${out}" \
    --max-pairs "${pairs}" --device "${DEVICE}" --dtype "${DTYPE}"
}

cache_main() {
  local checkpoint
  checkpoint="$(mother_checkpoint)"
  cache_canonical_split train "${SENDER_TRAIN_INDEX}" "${CANONICAL_TRAIN}" 512 "${checkpoint}"
  cache_canonical_split test "${SENDER_TEST_INDEX}" "${CANONICAL_TEST}" "${TEST_PAIRS}" "${checkpoint}"
}

cache_mismatch() {
  local checkpoint
  checkpoint="$(mother_checkpoint)"
  mkdir -p "${ROOT}/data"
  if [[ ! -f "${MISMATCH_DATA}.manifest.json" ]]; then
    "${PY}" "${ROOT}/build_true_mismatch_data.py" \
      --data "${TEST_DATA}" --out "${MISMATCH_DATA}" --seed "${SEED}"
  fi
  if [[ ! -f "${MISMATCH_NATIVE}/CACHE_SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/cache_qwen3_native_kv.py" \
      --model "${SENDER_MODEL}" --data "${MISMATCH_DATA}" --out "${MISMATCH_NATIVE}" \
      --max-pairs "${TEST_PAIRS}" --device "${DEVICE}" --dtype "${DTYPE}"
  fi
  cache_canonical_split mismatch "${MISMATCH_NATIVE}/index.json" \
    "${MISMATCH_CANONICAL}" "${TEST_PAIRS}" "${checkpoint}"
}

train_frozen_reader() {
  local receiver_name="$1" receiver_model="$2" teacher_index="$3" out="$4" train_dtype="$5"
  [[ -f "${out}/TRAIN_SUCCESS.json" ]] && return
  local checkpoint
  checkpoint="$(mother_checkpoint)"
  "${PY}" "${ROOT}/train_canonical_branch.py" \
    --receiver-name "${receiver_name}" --receiver-model "${receiver_model}" \
    --canonical-index "${CANONICAL_TRAIN}/index.json" --teacher-index "${teacher_index}" \
    --resume "${checkpoint}" --freeze-writer --reinitialize-reader --out "${out}" \
    --max-pairs "${TRAIN_PAIRS}" --epochs "${FORK_EPOCHS}" \
    --seed "$((SEED + 100))" --device "${DEVICE}" --dtype "${train_dtype}"
}

forks() {
  train_frozen_reader qwen3_4b "${RECEIVER4_MODEL}" "${TEACHER4}/index.json" "${FORK_ROOT}/qwen3_4b" "${DTYPE}"
  train_frozen_reader qwen3_5_4b "${RECEIVER35_MODEL}" "${TEACHER35}/index.json" "${FORK_ROOT}/qwen3_5_4b" "${RECEIVER35_TRAIN_DTYPE}"
}

train_oracle() {
  local receiver_name="$1" receiver_model="$2" teacher_index="$3" out="$4" train_dtype="$5"
  [[ -f "${out}/TRAIN_SUCCESS.json" ]] && return
  local checkpoint
  checkpoint="$(mother_checkpoint)"
  "${PY}" "${ROOT}/train_canonical_branch.py" \
    --receiver-name "${receiver_name}" --receiver-model "${receiver_model}" \
    --sender-index "${SENDER_TRAIN_INDEX}" --teacher-index "${teacher_index}" \
    --resume "${checkpoint}" --reinitialize-reader --out "${out}" \
    --max-pairs "${TRAIN_PAIRS}" --epochs "${ORACLE_EPOCHS}" \
    --seed "$((SEED + 200))" --device "${DEVICE}" --dtype "${train_dtype}"
}

oracles() {
  train_oracle qwen3_4b "${RECEIVER4_MODEL}" "${TEACHER4}/index.json" "${ORACLE_ROOT}/qwen3_4b/train" "${DTYPE}"
  train_oracle qwen3_5_4b "${RECEIVER35_MODEL}" "${TEACHER35}/index.json" "${ORACLE_ROOT}/qwen3_5_4b/train" "${RECEIVER35_TRAIN_DTYPE}"
  cache_canonical_split oracle4 "${SENDER_TEST_INDEX}" "${ORACLE_ROOT}/qwen3_4b/cache_test" \
    "${TEST_PAIRS}" "${ORACLE_ROOT}/qwen3_4b/train/checkpoint_latest.pt"
  cache_canonical_split oracle35 "${SENDER_TEST_INDEX}" "${ORACLE_ROOT}/qwen3_5_4b/cache_test" \
    "${TEST_PAIRS}" "${ORACLE_ROOT}/qwen3_5_4b/train/checkpoint_latest.pt"
}

evaluate_one() {
  local receiver_name="$1" receiver_model="$2" cache_index="$3" checkpoint="$4" out="$5" mismatch_index="${6:-}"
  [[ -f "${out}/SUCCESS.json" ]] && return
  local mismatch_args=()
  [[ -n "${mismatch_index}" ]] && mismatch_args=(--mismatch-index "${mismatch_index}")
  "${PY}" "${ROOT}/eval_canonical.py" \
    --receiver-name "${receiver_name}" --receiver-model "${receiver_model}" \
    --canonical-index "${cache_index}" --checkpoint "${checkpoint}" --out "${out}" \
    --max-pairs "${TEST_PAIRS}" --seed "${SEED}" --device "${DEVICE}" --dtype "${DTYPE}" \
    "${mismatch_args[@]}"
}

evaluate() {
  local mother_ckpt mismatch
  mother_ckpt="$(mother_checkpoint)"
  mismatch="${MISMATCH_CANONICAL}/index.json"
  evaluate_one qwen3_4b "${RECEIVER4_MODEL}" "${CANONICAL_TEST}/index.json" "${mother_ckpt}" "${EVAL_ROOT}/mother_qwen3_4b" "${mismatch}"
  evaluate_one qwen3_5_4b "${RECEIVER35_MODEL}" "${CANONICAL_TEST}/index.json" "${mother_ckpt}" "${EVAL_ROOT}/mother_qwen3_5_4b" "${mismatch}"
  evaluate_one qwen3_4b "${RECEIVER4_MODEL}" "${CANONICAL_TEST}/index.json" "${FORK_ROOT}/qwen3_4b/checkpoint_latest.pt" "${EVAL_ROOT}/frozen_qwen3_4b" "${mismatch}"
  evaluate_one qwen3_5_4b "${RECEIVER35_MODEL}" "${CANONICAL_TEST}/index.json" "${FORK_ROOT}/qwen3_5_4b/checkpoint_latest.pt" "${EVAL_ROOT}/frozen_qwen3_5_4b" "${mismatch}"
  evaluate_one qwen3_4b "${RECEIVER4_MODEL}" "${ORACLE_ROOT}/qwen3_4b/cache_test/index.json" "${ORACLE_ROOT}/qwen3_4b/train/checkpoint_latest.pt" "${EVAL_ROOT}/oracle_qwen3_4b"
  evaluate_one qwen3_5_4b "${RECEIVER35_MODEL}" "${ORACLE_ROOT}/qwen3_5_4b/cache_test/index.json" "${ORACLE_ROOT}/qwen3_5_4b/train/checkpoint_latest.pt" "${EVAL_ROOT}/oracle_qwen3_5_4b"
}

summarize() {
  "${PY}" "${ROOT}/summarize_p2i.py" \
    --mother4 "${EVAL_ROOT}/mother_qwen3_4b/SUCCESS.json" \
    --mother35 "${EVAL_ROOT}/mother_qwen3_5_4b/SUCCESS.json" \
    --frozen4 "${EVAL_ROOT}/frozen_qwen3_4b/SUCCESS.json" \
    --frozen35 "${EVAL_ROOT}/frozen_qwen3_5_4b/SUCCESS.json" \
    --oracle4 "${EVAL_ROOT}/oracle_qwen3_4b/SUCCESS.json" \
    --oracle35 "${EVAL_ROOT}/oracle_qwen3_5_4b/SUCCESS.json" \
    --out "${ROOT}/comparison"
}

status() {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  pgrep -af 'cache_teacher_features|train_canonical_branch|cache_canonical_evidence|eval_canonical' || true
  find "${ROOT}" -maxdepth 5 -type f \
    \( -name 'AUDIT.json' -o -name 'CACHE_SUCCESS.json' -o -name 'TRAIN_SUCCESS.json' -o -name 'SUCCESS.json' \) \
    -print 2>/dev/null | sort
}

wait_cuda() {
  until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do
    sleep 60
  done
  "$0" all
}

case "${1:-help}" in
  audit) audit ;;
  runtime-smoke) runtime_smoke ;;
  teachers) teacher4; teacher35 ;;
  mother) mother ;;
  cache) cache_main; cache_mismatch ;;
  forks) forks ;;
  oracles) oracles ;;
  eval) evaluate ;;
  summarize) summarize ;;
  status) status ;;
  wait-cuda) wait_cuda ;;
  all)
    audit
    teacher4
    teacher35
    mother
    cache_main
    cache_mismatch
    forks
    oracles
    evaluate
    summarize
    ;;
  *)
    echo "Usage: bash run_all.sh {audit|runtime-smoke|teachers|mother|cache|forks|oracles|eval|summarize|status|wait-cuda|all}" >&2
    exit 64
    ;;
esac
