#!/usr/bin/env bash
set -euo pipefail

ROOT=${P3D3_ROOT:-/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_seed1234}
P3B=/home/yezhe/伪查询/runs/p3b_multilayer_question_independent_kv_readability_seed1234
P3C=/home/yezhe/伪查询/runs/p3c_question_independent_multilayer_canonical_writer_seed1234
P3D=/home/yezhe/伪查询/runs/p3d_frozen_multilayer_canonical_writer_qwen3_4b_reader_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
MODEL8=/home/yezhe/all_models/models/Qwen/Qwen3-8B
MODEL4=/home/yezhe/all_models/models/Qwen/Qwen3-4B
HOTPOT=/home/yezhe/数据集/HotpotQA/raw
PROJECTIONS=${P3B}/projections/layerwise_pca_random.pt
PROTOCOL=${ROOT}/protocol/protocol.json
TRAIN_SAMPLES=${P3D3_TRAIN_SAMPLES:-64}
VALIDATION_SAMPLES=${P3D3_VALIDATION_SAMPLES:-64}

export PYTHONPATH=${ROOT}:${P3C}:${P3B}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_cuda() {
  until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do sleep 60; done
}

prepare() {
  mkdir -p "${ROOT}"/{protocol,data,cache/native,cache/canonical,readers,eval,audit,logs}
  "${PY}" "${ROOT}/select_p3d3_protocol.py" --p3d-protocol "${P3D}/protocol/protocol.json" --out "${PROTOCOL}"
  "${PY}" "${ROOT}/prepare_p3d3_data.py" --train "${HOTPOT}/hotpot_train_v1.1.json" --validation "${HOTPOT}/hotpot_dev_distractor_v1.json" --out "${ROOT}/data" --train-samples "${TRAIN_SAMPLES}" --validation-samples "${VALIDATION_SAMPLES}" --seed 1234
}

cache_split() {
  local split="$1"
  wait_cuda
  "${PY}" "${ROOT}/cache_p3d3_native.py" --model "${MODEL8}" --data "${ROOT}/data/${split}.jsonl" --out "${ROOT}/cache/native/${split}" --device cuda
  local writer
  writer=$("${PY}" -c "import json; print(json.load(open('${PROTOCOL}'))['writer_checkpoint'])")
  "${PY}" "${ROOT}/cache_p3d3_canonical.py" --native-cache "${ROOT}/cache/native/${split}/index.json" --writer "${writer}" --projections "${PROJECTIONS}" --p3c-code "${P3C}" --out "${ROOT}/cache/canonical/${split}" --device cuda
}

audit() {
  "${PY}" "${ROOT}/audit_p3d3.py" --model4 "${MODEL4}" --model8 "${MODEL8}" --protocol "${PROTOCOL}" --out "${ROOT}/audit/SUCCESS.json" \
    --native-train "${ROOT}/cache/native/train/index.json" --native-validation "${ROOT}/cache/native/validation/index.json" \
    --canonical-train "${ROOT}/cache/canonical/train/index.json" --canonical-validation "${ROOT}/cache/canonical/validation/index.json"
}

train_branch() {
  local branch="$1" memory
  if [[ "${branch}" == canonical16 ]]; then memory="${ROOT}/cache/canonical/train/index.json"; else memory="${ROOT}/cache/native/train/index.json"; fi
  wait_cuda
  "${PY}" "${ROOT}/train_p3d3_reader.py" --model "${MODEL4}" --memory "${memory}" --out "${ROOT}/readers/${branch}" --branch "${branch}" \
    --epochs "${P3D3_EPOCHS:-20}" --rank 32 --gate-init 0.01 --depend-weight 0.5 --margin 0.5 --seed 1234 --device cuda
}

evaluate() {
  wait_cuda
  "${PY}" "${ROOT}/eval_p3d3.py" --model "${MODEL4}" --native-memory "${ROOT}/cache/native/validation/index.json" \
    --canonical-memory "${ROOT}/cache/canonical/validation/index.json" --native-checkpoint "${ROOT}/readers/native_projected16/checkpoint_best.pt" \
    --canonical-checkpoint "${ROOT}/readers/canonical16/checkpoint_best.pt" --out "${ROOT}/eval" --seed 1234 --device cuda
}

case "${1:-all}" in
  prepare) prepare ;;
  cache) cache_split train; cache_split validation; audit ;;
  train-canonical) train_branch canonical16 ;;
  train-native) train_branch native_projected16 ;;
  train) train_branch canonical16; train_branch native_projected16 ;;
  eval) evaluate ;;
  all) prepare; cache_split train; cache_split validation; audit; train_branch canonical16; train_branch native_projected16; evaluate ;;
  *) echo "Usage: $0 {prepare|cache|train-canonical|train-native|train|eval|all}"; exit 2 ;;
esac
