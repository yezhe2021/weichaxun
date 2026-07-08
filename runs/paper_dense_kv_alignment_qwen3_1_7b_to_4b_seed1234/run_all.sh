#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
SENDER="${SENDER:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}"
RECEIVER="${RECEIVER:-/home/yezhe/all_models/models/Qwen/Qwen3-4B}"
TRAIN_DATA="${TRAIN_DATA:-/home/yezhe/数据集/gsm8k/train.jsonl}"
VAL_DATA="${VAL_DATA:-/home/yezhe/数据集/gsm8k/test.jsonl}"
GSM8K_DATA="${GSM8K_DATA:-/home/yezhe/数据集/gsm8k/test.jsonl}"
SELF_TRACE_DATA="${SELF_TRACE_DATA:-${ROOT}/self_traces/receiver_self_traces.jsonl}"
METHODS="${METHODS:-mse_only,mse_then_ce,paper_rec_then_mixed_generation,q_aware_functional}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-256}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-32}"
MAX_SOURCE_TOKENS="${MAX_SOURCE_TOKENS:-256}"
SELF_TRACE_MAX_CANDIDATES="${SELF_TRACE_MAX_CANDIDATES:-512}"
SELF_TRACE_MAX_KEPT="${SELF_TRACE_MAX_KEPT:-${MAX_TRAIN_SAMPLES}}"
SELF_TRACE_FILTER_MODE="${SELF_TRACE_FILTER_MODE:-strict}"
SELF_TRACE_MAX_NEW_TOKENS="${SELF_TRACE_MAX_NEW_TOKENS:-384}"
HIDDEN="${HIDDEN:-512}"
ANSWER_MODE="${ANSWER_MODE:-full}"
ATTENTION_TOPK="${ATTENTION_TOPK:-16}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-1}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-1}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

cd "${PROJECT}"

contains_method() {
  local method="$1"
  [[ ",${METHODS}," == *",${method},"* ]]
}

train_one() {
  local method="$1"
  local extra_args=()
  if [[ "${method}" == "paper_rec_then_mixed_generation" || "${method}" == "q_aware_functional" ]]; then
    ensure_self_traces
    extra_args+=(--phase2-data "${SELF_TRACE_DATA}" --phase2-target-field receiver_trace)
  fi
  "${PY}" "${ROOT}/train_paper_dense_adapter.py" \
    --sender-model "${SENDER}" \
    --receiver-model "${RECEIVER}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --out "${ROOT}/train/${method}" \
    --method "${method}" \
    --hidden "${HIDDEN}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --max-source-tokens "${MAX_SOURCE_TOKENS}" \
    --answer-mode "${ANSWER_MODE}" \
    --phase1-epochs "${PHASE1_EPOCHS}" \
    --phase2-epochs "${PHASE2_EPOCHS}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    "${extra_args[@]}"
}

generate_self_traces() {
  "${PY}" "${ROOT}/generate_receiver_self_traces.py" \
    --receiver-model "${RECEIVER}" \
    --data "${TRAIN_DATA}" \
    --out "${SELF_TRACE_DATA}" \
    --max-candidates "${SELF_TRACE_MAX_CANDIDATES}" \
    --max-kept "${SELF_TRACE_MAX_KEPT}" \
    --filter-mode "${SELF_TRACE_FILTER_MODE}" \
    --max-new-tokens "${SELF_TRACE_MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

ensure_self_traces() {
  if [[ ! -s "${SELF_TRACE_DATA}" ]]; then
    mkdir -p "$(dirname "${SELF_TRACE_DATA}")"
    generate_self_traces
  fi
}

checkpoint_for() {
  local method="$1"
  local checkpoint="${ROOT}/train/${method}/checkpoint_final.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  printf '%s\n' "${checkpoint}"
}

eval_one() {
  local method="$1"
  local data="${2:-${VAL_DATA}}"
  local out_dir="${3:-${ROOT}/eval/${method}}"
  "${PY}" "${ROOT}/eval_paper_dense_adapter.py" \
    --sender-model "${SENDER}" \
    --receiver-model "${RECEIVER}" \
    --data "${data}" \
    --adapter-checkpoint "$(checkpoint_for "${method}")" \
    --method-label "${method}" \
    --out "${out_dir}" \
    --max-samples "${MAX_VAL_SAMPLES}" \
    --max-source-tokens "${MAX_SOURCE_TOKENS}" \
    --answer-mode "${ANSWER_MODE}" \
    --attention-topk "${ATTENTION_TOPK}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

token_diag_one() {
  local method="$1"
  "${PY}" "${ROOT}/token_level_diagnostics.py" \
    --sender-model "${SENDER}" \
    --receiver-model "${RECEIVER}" \
    --data "${GSM8K_DATA}" \
    --adapter-checkpoint "$(checkpoint_for "${method}")" \
    --method-label "${method}" \
    --out "${ROOT}/token_diag_gsm8k/${method}" \
    --max-samples "${MAX_VAL_SAMPLES}" \
    --max-source-tokens "${MAX_SOURCE_TOKENS}" \
    --answer-mode "${ANSWER_MODE}" \
    --critical-mode numeric \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

train_selected() {
  for method in mse_only mse_then_ce paper_rec_then_mixed_generation q_aware_functional; do
    if contains_method "${method}"; then
      train_one "${method}"
    fi
  done
}

eval_selected() {
  for method in mse_only mse_then_ce paper_rec_then_mixed_generation q_aware_functional; do
    if contains_method "${method}"; then
      eval_one "${method}" "${GSM8K_DATA}" "${ROOT}/eval_gsm8k/${method}"
    fi
  done
}

case "${1:-help}" in
  self_traces) generate_self_traces ;;
  train_mse_only) train_one mse_only ;;
  train_mse_then_ce) train_one mse_then_ce ;;
  train_paper) train_one paper_rec_then_mixed_generation ;;
  train_q_aware) train_one q_aware_functional ;;
  train) train_selected ;;
  eval_mse_only) eval_one mse_only "${GSM8K_DATA}" "${ROOT}/eval_gsm8k/mse_only" ;;
  eval_mse_then_ce) eval_one mse_then_ce "${GSM8K_DATA}" "${ROOT}/eval_gsm8k/mse_then_ce" ;;
  eval_paper) eval_one paper_rec_then_mixed_generation "${GSM8K_DATA}" "${ROOT}/eval_gsm8k/paper_rec_then_mixed_generation" ;;
  eval_q_aware) eval_one q_aware_functional "${GSM8K_DATA}" "${ROOT}/eval_gsm8k/q_aware_functional" ;;
  eval) eval_selected ;;
  token_diag_mse_only) token_diag_one mse_only ;;
  token_diag_mse_then_ce) token_diag_one mse_then_ce ;;
  token_diag_paper) token_diag_one paper_rec_then_mixed_generation ;;
  token_diag_q_aware) token_diag_one q_aware_functional ;;
  all)
    bash "$0" self_traces
    bash "$0" train
    bash "$0" eval
    ;;
  *)
    cat <<'USAGE'
Usage:
  bash runs/paper_dense_kv_alignment_qwen3_1_7b_to_4b_seed1234/run_all.sh train
  bash runs/paper_dense_kv_alignment_qwen3_1_7b_to_4b_seed1234/run_all.sh eval
  bash runs/paper_dense_kv_alignment_qwen3_1_7b_to_4b_seed1234/run_all.sh all
  bash runs/paper_dense_kv_alignment_qwen3_1_7b_to_4b_seed1234/run_all.sh self_traces

Defaults:
  sender: Qwen3-1.7B
  receiver: Qwen3-4B
  train: /home/yezhe/数据集/gsm8k/train.jsonl
  eval: /home/yezhe/数据集/gsm8k/test.jsonl
  answer mode: full benchmark GSM8K answer
  paper/q-aware Phase II target: receiver self traces at self_traces/receiver_self_traces.jsonl
  methods: mse_only,mse_then_ce,paper_rec_then_mixed_generation,q_aware_functional

Useful overrides:
  METHODS=mse_only,q_aware_functional
  MAX_TRAIN_SAMPLES=128
  HIDDEN=128
  MAX_VAL_SAMPLES=16
  SELF_TRACE_FILTER_MODE=strict|flexible
  ANSWER_MODE=full|final_only
USAGE
    ;;
esac
