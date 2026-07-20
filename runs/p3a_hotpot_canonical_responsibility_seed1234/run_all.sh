#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3a_hotpot_canonical_responsibility_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
TRAIN_RAW=/home/yezhe/数据集/HotpotQA/raw/hotpot_train_v1.1.json
DEV_RAW=/home/yezhe/数据集/HotpotQA/raw/hotpot_dev_distractor_v1.json
MODEL8=/home/yezhe/all_models/models/Qwen/Qwen3-8B
MODEL4=/home/yezhe/all_models/models/Qwen/Qwen3-4B
P2IW=/home/yezhe/伪查询/runs/p2iw_token_preserving_canonical_writer_qwen3_8b_seed1234
P2IR=/home/yezhe/伪查询/runs/p2ir_token_preserving_canonical_reader_onboarding_seed1234
WRITER=${P2IW}/writer_full/best_checkpoint.pt
PROJECTIONS=${P2IW}/projections/pca_and_random.pt
OLD_READER=${P2IR}/qwen3_4b/full/train/checkpoint_best.pt
TRAIN_DATA=${ROOT}/data/train512.jsonl
DEV_DATA=${ROOT}/data/dev500.jsonl
TRAIN_CACHE=${ROOT}/cache/train/index.json
DEV_CACHE=${ROOT}/cache/dev/index.json

export PYTHONPATH=${ROOT}:${P2IW}:${P2IR}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

wait_cuda() { until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do sleep 60; done; }

run_eval() {
  local source="$1" checkpoint="$2" out="$3" max_samples="${4:-0}" cache="${5:-${DEV_CACHE}}"
  [[ -f "${out}/SUCCESS.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/eval_p3a_reader.py" --model "${MODEL4}" --cache "${cache}" --source "${source}" --checkpoint "${checkpoint}" --out "${out}" --max-samples "${max_samples}" --device cuda
}

run_all() {
  mkdir -p "${ROOT}/audit" "${ROOT}/data" "${ROOT}/cache" "${ROOT}/baselines" "${ROOT}/probes" "${ROOT}/readers"
  if [[ ! -f "${ROOT}/audit/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/audit_p3a.py" --root "${ROOT}" --train-raw "${TRAIN_RAW}" --dev-raw "${DEV_RAW}" --writer "${WRITER}" --old-reader "${OLD_READER}" --out "${ROOT}/audit/SUCCESS.json"
  fi
  if [[ ! -f "${ROOT}/data/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/prepare_p3a_data.py" --train-raw "${TRAIN_RAW}" --dev-raw "${DEV_RAW}" --out "${ROOT}/data" --train-samples 512 --dev-samples 500 --seed 1234
  fi
  wait_cuda
  if [[ ! -f "${ROOT}/cache/train/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/cache_p3a_memories.py" --model "${MODEL8}" --data "${TRAIN_DATA}" --writer "${WRITER}" --projections "${PROJECTIONS}" --out "${ROOT}/cache/train" --device cuda
  fi
  wait_cuda
  if [[ ! -f "${ROOT}/cache/dev/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/cache_p3a_memories.py" --model "${MODEL8}" --data "${DEV_DATA}" --writer "${WRITER}" --projections "${PROJECTIONS}" --out "${ROOT}/cache/dev" --device cuda
  fi

  if [[ ! -f "${ROOT}/baselines/q4_question_only/SUCCESS.json" ]]; then wait_cuda; "${PY}" "${ROOT}/eval_p3a_full_text.py" --model-name qwen3_4b --model "${MODEL4}" --data "${DEV_DATA}" --condition question_only --out "${ROOT}/baselines/q4_question_only" --device cuda; fi
  if [[ ! -f "${ROOT}/baselines/q4_full_text/SUCCESS.json" ]]; then wait_cuda; "${PY}" "${ROOT}/eval_p3a_full_text.py" --model-name qwen3_4b --model "${MODEL4}" --data "${DEV_DATA}" --condition full_text --out "${ROOT}/baselines/q4_full_text" --device cuda; fi
  if [[ ! -f "${ROOT}/baselines/q8_full_text/SUCCESS.json" ]]; then wait_cuda; "${PY}" "${ROOT}/eval_p3a_full_text.py" --model-name qwen3_8b --model "${MODEL8}" --data "${DEV_DATA}" --condition full_text --out "${ROOT}/baselines/q8_full_text" --device cuda; fi

  for source in hidden raw_kv pca_kv canonical; do
    out="${ROOT}/probes/${source}/train"
    if [[ ! -f "${out}/TRAIN_SUCCESS.json" ]]; then
      wait_cuda
      "${PY}" "${ROOT}/train_p3a_reader.py" --model "${MODEL4}" --cache "${TRAIN_CACHE}" --source "${source}" --profile probe --mode full --out "${out}" --epochs 2 --rank 32 --lr 2e-4 --seed 1234 --device cuda
    fi
    run_eval "${source}" "${out}/checkpoint_best.pt" "${ROOT}/probes/${source}/eval" 0
  done

  run_eval canonical "${OLD_READER}" "${ROOT}/readers/old_synthetic/eval" 0

  small="${ROOT}/readers/hotpot/small/train"
  if [[ ! -f "${small}/TRAIN_SUCCESS.json" ]]; then
    wait_cuda
    "${PY}" "${ROOT}/train_p3a_reader.py" --model "${MODEL4}" --cache "${TRAIN_CACHE}" --source canonical --profile hotpot --mode small --small-samples 32 --out "${small}" --epochs 20 --rank 64 --lr 5e-4 --seed 1234 --device cuda
  fi
  run_eval canonical "${small}/checkpoint_best.pt" "${ROOT}/readers/hotpot/small/eval" 32 "${TRAIN_CACHE}"

  full="${ROOT}/readers/hotpot/full/train"
  if [[ ! -f "${full}/TRAIN_SUCCESS.json" ]]; then
    wait_cuda
    "${PY}" "${ROOT}/train_p3a_reader.py" --model "${MODEL4}" --cache "${TRAIN_CACHE}" --source canonical --profile hotpot --mode full --init-checkpoint "${small}/checkpoint_best.pt" --out "${full}" --epochs 3 --rank 64 --lr 2e-4 --seed 1234 --device cuda
  fi
  run_eval canonical "${full}/checkpoint_best.pt" "${ROOT}/readers/hotpot/full/eval" 0
  "${PY}" "${ROOT}/summarize_p3a.py" --root "${ROOT}"
}

status() {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  pgrep -af 'prepare_p3a_data.py|cache_p3a_memories.py|train_p3a_reader.py|eval_p3a_reader.py|eval_p3a_full_text.py|run_all.sh all' || true
  find "${ROOT}" -maxdepth 5 -type f \( -name 'SUCCESS.json' -o -name 'TRAIN_SUCCESS.json' \) -print 2>/dev/null | sort
}

case "${1:-all}" in all) run_all ;; status) status ;; *) echo "Usage: bash run_all.sh {all|status}" >&2; exit 64 ;; esac
