#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234
P3D3=/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
MODEL=/home/yezhe/all_models/models/Qwen/Qwen3-4B

export PYTHONPATH=${ROOT}:${P3D3}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_cuda() {
  until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do sleep 60; done
}

cache() {
  mkdir -p "${ROOT}"/{cache/train,cache/validation,audit,overfit16,formal512,logs}
  wait_cuda
  "${PY}" "${ROOT}/cache_receiver_native_headwise.py" --model "${MODEL}" --data "${P3D3}/data/train.jsonl" --out "${ROOT}/cache/train" --device cuda
  "${PY}" "${ROOT}/cache_receiver_native_headwise.py" --model "${MODEL}" --data "${P3D3}/data/validation.jsonl" --out "${ROOT}/cache/validation" --device cuda
  "${PY}" "${ROOT}/audit_p3e_a.py" --model "${MODEL}" --train-cache "${ROOT}/cache/train/index.json" --validation-cache "${ROOT}/cache/validation/index.json" \
    --train-data "${P3D3}/data/train.jsonl" --validation-data "${P3D3}/data/validation.jsonl" --out "${ROOT}/audit/SUCCESS.json"
  "${PY}" "${ROOT}/smoke_p3e_a.py" --model "${MODEL}" --device cuda
}

overfit() {
  wait_cuda
  "${PY}" "${ROOT}/train_receiver_native_headwise.py" --model "${MODEL}" --memory "${ROOT}/cache/train/index.json" --out "${ROOT}/overfit16/reader" \
    --mode overfit16 --max-samples 16 --epochs "${P3E_A_OVERFIT_EPOCHS:-30}" --rank 32 --gate-init 0.01 --depend-weight 0.5 --margin 0.5 --seed 1234 --device cuda
  "${PY}" "${ROOT}/eval_receiver_native_headwise.py" --model "${MODEL}" --memory "${ROOT}/cache/train/index.json" --checkpoint "${ROOT}/overfit16/reader/checkpoint_best.pt" \
    --out "${ROOT}/overfit16/eval" --max-samples 16 --seed 1234 --device cuda
  "${PY}" "${ROOT}/check_overfit_gate.py" --result "${ROOT}/overfit16/eval/SUCCESS.json" --out "${ROOT}/overfit16/GATE.json"
}

formal() {
  wait_cuda
  "${PY}" "${ROOT}/train_receiver_native_headwise.py" --model "${MODEL}" --memory "${ROOT}/cache/train/index.json" --out "${ROOT}/formal512/reader" \
    --mode formal512 --max-samples 512 --epochs "${P3E_A_FORMAL_EPOCHS:-20}" --rank 32 --gate-init 0.01 --depend-weight 0.5 --margin 0.5 --seed 1234 --device cuda
  "${PY}" "${ROOT}/eval_receiver_native_headwise.py" --model "${MODEL}" --memory "${ROOT}/cache/validation/index.json" --checkpoint "${ROOT}/formal512/reader/checkpoint_best.pt" \
    --out "${ROOT}/formal512/eval" --max-samples 64 --seed 1234 --device cuda
}

case "${1:-all}" in
  cache) cache ;;
  overfit) overfit ;;
  formal) formal ;;
  all) cache; overfit; formal ;;
  *) echo "Usage: $0 {cache|overfit|formal|all}"; exit 2 ;;
esac
