#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_n2_reader_readout_bottleneck_diagnosis_seed1234
P3EN=/home/yezhe/伪查询/runs/p3e_n_native_oracle_bottleneck_decomposition_seed1234
P3EL=/home/yezhe/伪查询/runs/p3e_l_question_conditioned_sender_diagnosis_seed1234
P3EA=/home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234
P3D3=/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
MODEL=/home/yezhe/all_models/models/Qwen/Qwen3-4B
READER=${P3EN}/reader_c/checkpoint_epoch_005.pt

export PYTHONPATH=${ROOT}:${P3EN}:${P3EL}:${P3EA}:${P3D3}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${ROOT}"/{cache/train,cache/validation,probe,evaluation}

if [[ ! -f "${ROOT}/cache/train/SUCCESS.json" ]]; then
  "${PY}" "${ROOT}/cache_reader_readouts.py" --model "${MODEL}" \
    --memory "${P3EN}/cache/train/index.json" --reader "${READER}" \
    --out "${ROOT}/cache/train" --max-samples 512 --device cuda
fi
if [[ ! -f "${ROOT}/cache/validation/SUCCESS.json" ]]; then
  "${PY}" "${ROOT}/cache_reader_readouts.py" --model "${MODEL}" \
    --memory "${P3EN}/cache/validation/index.json" --reader "${READER}" \
    --out "${ROOT}/cache/validation" --max-samples 64 --device cuda
fi

"${PY}" "${ROOT}/train_readout_probe.py" --cache "${ROOT}/cache/train/index.json" \
  --out "${ROOT}/probe" --max-samples 512 --epochs 10 --lr 2e-4 --seed 1234 --device cuda

"${PY}" "${ROOT}/eval_readout_probe.py" --cache "${ROOT}/cache/validation/index.json" \
  --checkpoint "${ROOT}/probe/checkpoint_best.pt" --out "${ROOT}/evaluation" \
  --max-samples 64 --device cuda

"${PY}" "${ROOT}/summarize.py" --probe "${ROOT}/evaluation/SUCCESS.json" \
  --p3en "${P3EN}/evaluation/SUCCESS.json" --out "${ROOT}/SUCCESS.json"
