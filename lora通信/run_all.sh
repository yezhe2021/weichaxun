#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}
SENDER=${SENDER_MODEL:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}
RECEIVER=${RECEIVER_MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-4B}
HOTPOT=${HOTPOT_ROOT:-/home/yezhe/数据集/HotpotQA/raw}
RUN_ROOT=${LORA_RUN_ROOT:-${ROOT}/outputs/formal}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-512}
VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-64}
EPOCHS=${EPOCHS:-5}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-64}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
DEVICE=${DEVICE:-cuda}
ALLOW_NEGATIVE_FALLBACK=${ALLOW_NEGATIVE_FALLBACK:-0}

export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

prepare() {
  mkdir -p "${RUN_ROOT}/data"
  "${PY}" "${ROOT}/prepare_data.py" \
    --train "${HOTPOT}/hotpot_train_v1.1.json" \
    --validation "${HOTPOT}/hotpot_dev_distractor_v1.json" \
    --out "${RUN_ROOT}/data" \
    --train-samples "${TRAIN_SAMPLES}" \
    --validation-samples "${VALIDATION_SAMPLES}" \
    --evidence-scope gold_docs \
    --seed 1234
}

cache() {
  for split in train validation; do
    mkdir -p "${RUN_ROOT}/cache/${split}"
    "${PY}" "${ROOT}/cache_memory.py" \
      --sender "${SENDER}" \
      --data "${RUN_ROOT}/data/${split}.jsonl" \
      --out "${RUN_ROOT}/cache/${split}" \
      --device "${DEVICE}" \
      --max-memory-tokens 1024
  done
}

negatives() {
  mkdir -p "${RUN_ROOT}/negatives"
  local extra=()
  if [[ "${ALLOW_NEGATIVE_FALLBACK}" == "1" ]]; then
    extra+=(--allow-answer-type-fallback)
  fi
  for split in train validation; do
    "${PY}" "${ROOT}/build_negatives.py" \
      --memory "${RUN_ROOT}/cache/${split}/index.json" \
      --out "${RUN_ROOT}/negatives/${split}.json" \
      "${extra[@]}"
  done
}

train_variant() {
  local variant=$1
  local extra=()
  if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
    extra+=(--max-samples "${MAX_TRAIN_SAMPLES}")
  fi
  mkdir -p "${RUN_ROOT}/checkpoints/${variant}"
  "${PY}" "${ROOT}/train.py" \
    --receiver "${RECEIVER}" \
    --memory "${RUN_ROOT}/cache/train/index.json" \
    --negatives "${RUN_ROOT}/negatives/train.json" \
    --variant "${variant}" \
    --out "${RUN_ROOT}/checkpoints/${variant}" \
    --epochs "${EPOCHS}" \
    --seed 1234 \
    --device "${DEVICE}" \
    "${extra[@]}"
}

train_all() {
  train_variant reader_only
  train_variant reader_lora
  train_variant lora_only
}

evaluate() {
  mkdir -p "${RUN_ROOT}/eval"
  "${PY}" "${ROOT}/evaluate.py" \
    --receiver "${RECEIVER}" \
    --memory "${RUN_ROOT}/cache/validation/index.json" \
    --negatives "${RUN_ROOT}/negatives/validation.json" \
    --reader-only-checkpoint "${RUN_ROOT}/checkpoints/reader_only/checkpoint_best.pt" \
    --reader-lora-checkpoint "${RUN_ROOT}/checkpoints/reader_lora/checkpoint_best.pt" \
    --lora-only-checkpoint "${RUN_ROOT}/checkpoints/lora_only/checkpoint_best.pt" \
    --out "${RUN_ROOT}/eval" \
    --max-samples "${MAX_EVAL_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --seed 1234 \
    --device "${DEVICE}"
}

run_smoke() {
  LORA_RUN_ROOT="${ROOT}/outputs/smoke" \
  TRAIN_SAMPLES=32 \
  VALIDATION_SAMPLES=32 \
  EPOCHS=1 \
  MAX_TRAIN_SAMPLES=1 \
  MAX_EVAL_SAMPLES=1 \
  MAX_NEW_TOKENS=2 \
  ALLOW_NEGATIVE_FALLBACK=1 \
  bash "${ROOT}/run_all.sh" all
}

case "${1:-all}" in
  prepare) prepare ;;
  cache) cache ;;
  negatives) negatives ;;
  train-reader-only) train_variant reader_only ;;
  train-reader-lora) train_variant reader_lora ;;
  train-lora-only) train_variant lora_only ;;
  train) train_all ;;
  eval) evaluate ;;
  smoke) run_smoke ;;
  all) prepare; cache; negatives; train_all; evaluate ;;
  *) echo "Usage: $0 {prepare|cache|negatives|train-reader-only|train-reader-lora|train-lora-only|train|eval|smoke|all}"; exit 2 ;;
esac
