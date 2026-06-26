#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
REAL="${PROJECT}/runs/real_qwen3_0_6b_to_1_7b_seed1234"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
SENDER="${SENDER:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B}"
RECEIVER="${RECEIVER:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}"
DATA="${DATA:-/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-256}"
ATTENTION_TOPK="${ATTENTION_TOPK:-16}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
EQUIVALENCE_ATOL="${EQUIVALENCE_ATOL:-0.5}"

cd "${PROJECT}"

checkpoint_for() {
  case "$1" in
    mse_then_ce)
      printf '%s\n' "${REAL}/train/mse_then_ce/checkpoint_epoch1.pt"
      ;;
    mse_only)
      printf '%s\n' "${REAL}/train/mse_only/checkpoint_epoch1.pt"
      ;;
    *)
      echo "Unknown checkpoint label: $1" >&2
      exit 2
      ;;
  esac
}

diagnose() {
  local label="$1"
  local checkpoint
  checkpoint="$(checkpoint_for "${label}")"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  "${PY}" "${ROOT}/run_two_stage_diagnosis.py" \
    --sender-model "${SENDER}" \
    --receiver-model "${RECEIVER}" \
    --data "${DATA}" \
    --translator-checkpoint "${checkpoint}" \
    --checkpoint-label "${label}" \
    --out "${ROOT}/results/${label}" \
    --max-samples "${MAX_SAMPLES}" \
    --max-context-tokens "${MAX_CONTEXT_TOKENS}" \
    --attention-topk "${ATTENTION_TOPK}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --equivalence-atol "${EQUIVALENCE_ATOL}"
}

case "${1:-help}" in
  diagnose_mse_then_ce) diagnose mse_then_ce ;;
  diagnose_mse_only) diagnose mse_only ;;
  package) "${PY}" "${ROOT}/package_results.py" ;;
  *)
    cat <<'USAGE'
Usage:
  run_all.sh diagnose_mse_then_ce
  run_all.sh diagnose_mse_only
  run_all.sh package

Environment overrides:
  PY=/path/to/python
  DATA=/path/to/hotpot_dev_context_qa.jsonl
  MAX_SAMPLES=64
  MAX_CONTEXT_TOKENS=256
  ATTENTION_TOPK=16
  DEVICE=cuda|cpu
  DTYPE=float16|float32
  EQUIVALENCE_ATOL=0.5

No command launches both checkpoints automatically.
USAGE
    ;;
esac
