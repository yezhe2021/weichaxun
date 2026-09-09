#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_l_question_conditioned_sender_diagnosis_seed1234
P3D3=/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234
C1=/home/yezhe/伪查询/runs/p3e_c1_learnable_overcomplete_canonical_head_reader_seed1234
C2=/home/yezhe/伪查询/runs/p3e_c2_learned_head_structured_writer_seed1234
P3EC0=/home/yezhe/伪查询/runs/p3e_c0_duplicate_overcomplete_canonical_head_bus_seed1234
P3EB=/home/yezhe/伪查询/runs/p3e_b_sender_native_headwise16_qwen3_8b_to_4b_seed1234
P3EA=/home/yezhe/伪查询/runs/p3e_a_receiver_native_headwise16_qwen3_4b_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
SENDER=/home/yezhe/all_models/models/Qwen/Qwen3-8B
RECEIVER=/home/yezhe/all_models/models/Qwen/Qwen3-4B
TRAIN_BASE=${P3D3}/cache/native/train/index.json
VAL_BASE=${P3D3}/cache/native/validation/index.json
C1_READER=${C1}/formal512/reader/checkpoint_best.pt
C2_WRITER=${C2}/writer_formal512/worker/checkpoint_best.pt
NATIVE_READER=${P3EB}/formal512/reader/checkpoint_best.pt

export PYTHONPATH=${ROOT}:${C2}:${C1}:${P3EC0}:${P3EB}:${P3EA}:${P3D3}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${ROOT}"/{cache/train,cache/validation,zero_shot,conditioned_writer,final_eval}

if [[ ! -f "${ROOT}/cache/train/SUCCESS.json" ]]; then
  "${PY}" "${ROOT}/cache_conditioned_native.py" --model "${SENDER}" --base-memory "${TRAIN_BASE}" --out "${ROOT}/cache/train" --max-samples 512 --device cuda
fi
if [[ ! -f "${ROOT}/cache/validation/SUCCESS.json" ]]; then
  "${PY}" "${ROOT}/cache_conditioned_native.py" --model "${SENDER}" --base-memory "${VAL_BASE}" --out "${ROOT}/cache/validation" --max-samples 64 --device cuda
fi

if [[ ! -f "${ROOT}/zero_shot/SUCCESS.json" ]]; then
  "${PY}" "${ROOT}/eval_p3e_l.py" --model "${RECEIVER}" --base-memory "${VAL_BASE}" --conditioned-memory "${ROOT}/cache/validation/index.json" \
    --baseline-writer "${C2_WRITER}" --conditioned-writer "${C2_WRITER}" --reader "${C1_READER}" --out "${ROOT}/zero_shot" --max-samples 64 --device cuda
fi

"${PY}" "${ROOT}/decide_writer_training.py" --zero-shot "${ROOT}/zero_shot/SUCCESS.json" --out "${ROOT}/writer_decision.json"
TRAIN_WRITER=$("${PY}" -c "import json; print('1' if json.load(open('${ROOT}/writer_decision.json'))['train_conditioned_writer'] else '0')")

FINAL_WRITER="${C2_WRITER}"
if [[ "${TRAIN_WRITER}" == "1" ]]; then
  if [[ ! -f "${ROOT}/conditioned_writer/TRAIN_SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/train_conditioned_writer.py" --model "${RECEIVER}" --memory "${ROOT}/cache/train/index.json" \
      --init-writer "${C2_WRITER}" --native-reader "${NATIVE_READER}" --canonical-reader "${C1_READER}" \
      --out "${ROOT}/conditioned_writer" --max-samples 512 --epochs 8 --lr 1e-4 --seed 1234 --device cuda
  fi
  FINAL_WRITER="${ROOT}/conditioned_writer/checkpoint_best.pt"
fi

"${PY}" "${ROOT}/eval_p3e_l.py" --model "${RECEIVER}" --base-memory "${VAL_BASE}" --conditioned-memory "${ROOT}/cache/validation/index.json" \
  --baseline-writer "${C2_WRITER}" --conditioned-writer "${FINAL_WRITER}" --reader "${C1_READER}" --out "${ROOT}/final_eval" --max-samples 64 --device cuda

"${PY}" "${ROOT}/summarize.py" --zero-shot "${ROOT}/zero_shot/SUCCESS.json" --final "${ROOT}/final_eval/SUCCESS.json" \
  --decision "${ROOT}/writer_decision.json" --out "${ROOT}/SUCCESS.json"
