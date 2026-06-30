#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
SENDER="${SENDER:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-0___6B}"
RECEIVER="${RECEIVER:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}"
TRAIN_DATA="${TRAIN_DATA:-/home/yezhe/数据集/HotpotQA/processed/hotpot_train_context_qa.jsonl}"
VAL_DATA="${VAL_DATA:-/home/yezhe/数据集/HotpotQA/processed/hotpot_dev_context_qa.jsonl}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-512}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-64}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-256}"
ATTENTION_TOPK="${ATTENTION_TOPK:-16}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
STAGE_EPOCHS="${STAGE_EPOCHS:-1}"
EQUIVALENCE_ATOL="${EQUIVALENCE_ATOL:-0.5}"

cd "${PROJECT}"

train_one() {
  local regime="$1"
  "${PY}" "${ROOT}/train_q_aware_functional.py" \
    --sender-model "${SENDER}" \
    --receiver-model "${RECEIVER}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --out "${ROOT}/train/${regime}" \
    --regime "${regime}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --max-context-tokens "${MAX_CONTEXT_TOKENS}" \
    --stage-epochs "${STAGE_EPOCHS}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --equivalence-atol "${EQUIVALENCE_ATOL}"
}

eval_one() {
  local regime="$1"
  local checkpoint="${ROOT}/train/${regime}/checkpoint_final.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  "${PY}" "${ROOT}/eval_q_aware_comparison.py" \
    --sender-model "${SENDER}" \
    --receiver-model "${RECEIVER}" \
    --data "${VAL_DATA}" \
    --translator-checkpoint "${checkpoint}" \
    --method-label "${regime}" \
    --out "${ROOT}/eval/${regime}" \
    --max-samples "${MAX_VAL_SAMPLES}" \
    --max-context-tokens "${MAX_CONTEXT_TOKENS}" \
    --attention-topk "${ATTENTION_TOPK}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --equivalence-atol "${EQUIVALENCE_ATOL}"
}

case "${1:-help}" in
  train_mse_only) train_one mse_only ;;
  train_mse_then_ce) train_one mse_then_ce ;;
  train_q_aware_functional) train_one q_aware_functional ;;
  train)
    train_one mse_only
    train_one mse_then_ce
    train_one q_aware_functional
    ;;
  eval_mse_only) eval_one mse_only ;;
  eval_mse_then_ce) eval_one mse_then_ce ;;
  eval_q_aware_functional) eval_one q_aware_functional ;;
  eval)
    eval_one mse_only
    eval_one mse_then_ce
    eval_one q_aware_functional
    ;;
  package) "${PY}" "${ROOT}/package_results.py" ;;
  all)
    bash "$0" train
    bash "$0" eval
    bash "$0" package
    ;;
  *)
    cat <<'USAGE'
Usage:
  run_all.sh train_mse_only
  run_all.sh train_mse_then_ce
  run_all.sh train_q_aware_functional
  run_all.sh train
  run_all.sh eval_mse_only
  run_all.sh eval_mse_then_ce
  run_all.sh eval_q_aware_functional
  run_all.sh eval
  run_all.sh package
  run_all.sh all

Default budget:
  All regimes run 3 stage-passes over MAX_TRAIN_SAMPLES.
  mse_only: mse + mse + mse
  mse_then_ce: mse + ce + ce
  q_aware_functional: mse + q_aware_readout + functional

Environment overrides:
  MAX_TRAIN_SAMPLES=512
  MAX_VAL_SAMPLES=64
  STAGE_EPOCHS=1
  DEVICE=cuda|cpu
  DTYPE=float16|float32
USAGE
    ;;
esac
