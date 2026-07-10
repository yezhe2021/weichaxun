#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"

PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
MODEL="${MODEL:-/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B}"
DATA="${DATA:-/home/yezhe/数据集/HotpotQA/raw/hotpot_dev_distractor_v1.json}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-4}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
SEED="${SEED:-1234}"
OUT="${OUT:-${ROOT}/results}"

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

common_args=(
  --model "${MODEL}"
  --data "${DATA}"
  --max-input-tokens "${MAX_INPUT_TOKENS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --seed "${SEED}"
)

case "${1:-help}" in
  dry-run)
    "${PY}" "${ROOT}/run_p0.py" \
      "${common_args[@]}" \
      --max-samples "${MAX_SAMPLES}" \
      --out "${OUT}" \
      --dry-run \
      --overwrite
    ;;
  smoke)
    "${PY}" "${ROOT}/run_p0.py" \
      "${common_args[@]}" \
      --max-samples "${SMOKE_SAMPLES}" \
      --out "${OUT}"
    ;;
  run)
    "${PY}" "${ROOT}/run_p0.py" \
      "${common_args[@]}" \
      --max-samples "${MAX_SAMPLES}" \
      --out "${OUT}"
    ;;
  *)
    cat <<'USAGE'
Usage:
  bash run_all.sh dry-run   # validate split, tokenizer, token budget, and prompts
  MAX_SAMPLES=4 OUT=... bash run_all.sh smoke
  MAX_SAMPLES=128 bash run_all.sh run

Conditions:
  question_only, a_only, b_only, a_plus_b

Useful overrides:
  MODEL, DATA, OUT, MAX_SAMPLES, SMOKE_SAMPLES, MAX_INPUT_TOKENS, MAX_NEW_TOKENS,
  DEVICE, DTYPE, SEED, PY
USAGE
    ;;
esac
