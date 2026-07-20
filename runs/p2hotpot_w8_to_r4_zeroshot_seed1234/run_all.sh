#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p2hotpot_w8_to_r4_zeroshot_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
RAW=/home/yezhe/数据集/HotpotQA/raw/hotpot_dev_distractor_v1.json
MODEL8=/home/yezhe/all_models/models/Qwen/Qwen3-8B
MODEL4=/home/yezhe/all_models/models/Qwen/Qwen3-4B
P2IW=/home/yezhe/伪查询/runs/p2iw_token_preserving_canonical_writer_qwen3_8b_seed1234
P2IR=/home/yezhe/伪查询/runs/p2ir_token_preserving_canonical_reader_onboarding_seed1234
WRITER=${P2IW}/writer_full/best_checkpoint.pt
PROJECTIONS=${P2IW}/projections/pca_and_random.pt
READER=${P2IR}/qwen3_4b/full/train/checkpoint_best.pt
DATA=${ROOT}/data/dev64.jsonl
CANONICAL=${ROOT}/cache/w8_canonical/index.json

export PYTHONPATH=${ROOT}:${P2IW}:${P2IR}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_cuda() {
  while ! nvidia-smi >/dev/null 2>&1; do sleep 30; done
}

run_all() {
  mkdir -p "${ROOT}/audit" "${ROOT}/data" "${ROOT}/cache" "${ROOT}/evaluation"
  if [[ ! -f "${ROOT}/audit/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/audit_hotpot.py" --root "${ROOT}" --raw "${RAW}" --writer "${WRITER}" --reader "${READER}" --out "${ROOT}/audit/SUCCESS.json"
  fi
  if [[ ! -f "${ROOT}/data/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/prepare_hotpot.py" --raw "${RAW}" --out "${ROOT}/data" --samples 64 --seed 1234
  fi
  wait_cuda
  if [[ ! -f "${ROOT}/evaluation/qwen3_4b_question_only/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/eval_full_text.py" --model-name qwen3_4b --model "${MODEL4}" --data "${DATA}" --condition question_only --out "${ROOT}/evaluation/qwen3_4b_question_only" --device cuda
  fi
  wait_cuda
  if [[ ! -f "${ROOT}/evaluation/qwen3_4b_full_text/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/eval_full_text.py" --model-name qwen3_4b --model "${MODEL4}" --data "${DATA}" --condition full_text --out "${ROOT}/evaluation/qwen3_4b_full_text" --device cuda
  fi
  wait_cuda
  if [[ ! -f "${ROOT}/evaluation/qwen3_8b_full_text/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/eval_full_text.py" --model-name qwen3_8b --model "${MODEL8}" --data "${DATA}" --condition full_text --out "${ROOT}/evaluation/qwen3_8b_full_text" --device cuda
  fi
  wait_cuda
  if [[ ! -f "${ROOT}/cache/w8_canonical/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/cache_w8_canonical.py" --model "${MODEL8}" --data "${DATA}" --writer-checkpoint "${WRITER}" --projections "${PROJECTIONS}" --out "${ROOT}/cache/w8_canonical" --device cuda
  fi
  wait_cuda
  if [[ ! -f "${ROOT}/evaluation/w8_to_r4_canonical/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/eval_canonical.py" --receiver-model "${MODEL4}" --reader-checkpoint "${READER}" --canonical-index "${CANONICAL}" --out "${ROOT}/evaluation/w8_to_r4_canonical" --device cuda --seed 1234
  fi
  "${PY}" "${ROOT}/summarize_hotpot.py" --root "${ROOT}"
}

status() {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  pgrep -af 'prepare_hotpot.py|eval_full_text.py|cache_w8_canonical.py|eval_canonical.py|p2hotpot_run.sh|run_all.sh all' || true
  find "${ROOT}" -maxdepth 4 -name 'SUCCESS.json' -print 2>/dev/null | sort
}

case "${1:-all}" in
  all) run_all ;;
  status) status ;;
  *) echo "Usage: bash run_all.sh {all|status}" >&2; exit 64 ;;
esac
