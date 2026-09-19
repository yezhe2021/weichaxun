#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_n3_attention_vs_readout_sufficiency_seed1234
P3EN2=/home/yezhe/伪查询/runs/p3e_n2_reader_readout_bottleneck_diagnosis_seed1234
P3D3=/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python

export PYTHONPATH="${ROOT}:${P3EN2}:${P3D3}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${ROOT}/probe" "${ROOT}/evaluation"

for MODE in attention_only readout_only attention_readout; do
  mkdir -p "${ROOT}/probe/${MODE}" "${ROOT}/evaluation/${MODE}"
  "${PY}" "${ROOT}/train_probe.py" \
    --cache "${P3EN2}/cache/train/index.json" \
    --out "${ROOT}/probe/${MODE}" --mode "${MODE}" \
    --max-samples 512 --epochs 10 --lr 2e-4 --seed 1234 --device cuda
  "${PY}" "${ROOT}/eval_probe.py" \
    --cache "${P3EN2}/cache/validation/index.json" \
    --checkpoint "${ROOT}/probe/${MODE}/checkpoint_best.pt" \
    --out "${ROOT}/evaluation/${MODE}" \
    --max-samples 64 --device cuda
done

"${PY}" "${ROOT}/summarize.py" --root "${ROOT}"

