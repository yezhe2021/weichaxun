#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
RUN=${ROOT}/runs/hidden_public_kv_mvp_seed1234
Q3=/home/yezhe/all_models/models/Qwen/Qwen3-4B
Q35=/home/yezhe/all_models/models/Qwen/Qwen3___5-4B
TRAIN_RAW=/home/yezhe/数据集/HotpotQA/raw/hotpot_train_v1.1.json
DEV_RAW=/home/yezhe/数据集/HotpotQA/raw/hotpot_dev_distractor_v1.json

export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "${RUN}/data"
if [[ ! -f "${RUN}/data/train512.jsonl" ]]; then
  "${PY}" -m hidden_public_kv.prepare_data --raw "${TRAIN_RAW}" --out "${RUN}/data/train512.jsonl" --limit 512 --seed 1234
fi
if [[ ! -f "${RUN}/data/dev500.jsonl" ]]; then
  "${PY}" -m hidden_public_kv.prepare_data --raw "${DEV_RAW}" --out "${RUN}/data/dev500.jsonl" --limit 500 --seed 1235
fi

if [[ ! -f "${RUN}/a0_overfit/checkpoint_latest.pt" ]]; then
  "${PY}" -m hidden_public_kv.train_a0 --model "${Q3}" --data "${RUN}/data/train512.jsonl" \
    --out "${RUN}/a0_overfit" --limit 32 --epochs 20 --gradient-accumulation 8 --optimizer adafactor
fi

if [[ ! -f "${RUN}/a0/checkpoint_latest.pt" ]]; then
  "${PY}" -m hidden_public_kv.train_a0 --model "${Q3}" --data "${RUN}/data/train512.jsonl" \
    --out "${RUN}/a0" --epochs 3 --gradient-accumulation 8 --optimizer adafactor
fi

if [[ ! -f "${RUN}/cache/qwen3_dev/index.json" ]]; then
  "${PY}" -m hidden_public_kv.cache_hidden --kind qwen3 --model "${Q3}" \
    --data "${RUN}/data/dev500.jsonl" --out "${RUN}/cache/qwen3_dev" --include-removed
fi

if [[ ! -f "${RUN}/eval_a0/SUCCESS.json" ]]; then
  "${PY}" -m hidden_public_kv.evaluate --receiver-model "${Q3}" \
    --cache "${RUN}/cache/qwen3_dev/index.json" --checkpoint "${RUN}/a0/checkpoint_latest.pt" \
    --out "${RUN}/eval_a0" --max-new-tokens 32
fi

if [[ ! -f "${RUN}/cache/qwen35_train/index.json" ]]; then
  "${PY}" -m hidden_public_kv.cache_hidden --kind qwen35 --model "${Q35}" \
    --data "${RUN}/data/train512.jsonl" --out "${RUN}/cache/qwen35_train"
fi
if [[ ! -f "${RUN}/cache/qwen35_dev/index.json" ]]; then
  "${PY}" -m hidden_public_kv.cache_hidden --kind qwen35 --model "${Q35}" \
    --data "${RUN}/data/dev500.jsonl" --out "${RUN}/cache/qwen35_dev" --include-removed
fi

if [[ ! -f "${RUN}/a1/checkpoint_latest.pt" ]]; then
  "${PY}" -m hidden_public_kv.train_a1 --receiver-model "${Q3}" \
    --cache "${RUN}/cache/qwen35_train/index.json" --a0-checkpoint "${RUN}/a0/checkpoint_latest.pt" \
    --out "${RUN}/a1" --epochs 3 --gradient-accumulation 8 --optimizer adafactor \
    --rank-weight 0.2 --margin 0.5
fi

if [[ ! -f "${RUN}/eval_a1/SUCCESS.json" ]]; then
  "${PY}" -m hidden_public_kv.evaluate --receiver-model "${Q3}" \
    --cache "${RUN}/cache/qwen35_dev/index.json" --checkpoint "${RUN}/a1/checkpoint_latest.pt" \
    --out "${RUN}/eval_a1" --max-new-tokens 32
fi
