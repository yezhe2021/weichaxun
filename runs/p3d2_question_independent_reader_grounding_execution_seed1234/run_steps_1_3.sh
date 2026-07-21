#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3d2_question_independent_reader_grounding_execution_seed1234
P3D=/home/yezhe/伪查询/runs/p3d_frozen_multilayer_canonical_writer_qwen3_4b_reader_seed1234
P3B=/home/yezhe/伪查询/runs/p3b_multilayer_question_independent_kv_readability_seed1234
P3C=/home/yezhe/伪查询/runs/p3c_question_independent_multilayer_canonical_writer_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
MODEL=/home/yezhe/all_models/models/Qwen/Qwen3-4B
PROTOCOL=${P3D}/protocol/protocol.json
P3D_READER=${P3D}/readers/canonical16/train/checkpoint_best.pt
NATIVE_RESULT=${P3D}/readers/native16/eval/SUCCESS.json

export PYTHONPATH=${ROOT}:${P3D}:${P3B}:${P3C}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

wait_cuda() { until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do sleep 60; done; }

cache_teacher() {
  local split="$1"
  [[ -f "${ROOT}/teacher_cache/${split}/SUCCESS.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/cache_p3d2_teachers.py" --model "${MODEL}" --protocol "${PROTOCOL}" --split "${split}" --out "${ROOT}/teacher_cache/${split}" --device cuda
}

train_configuration() {
  local configuration="$1" blocks="$2" small_epochs="$3"
  local branch="${ROOT}/step3_capacity/${configuration}"
  mkdir -p "${branch}/small/train" "${branch}/small/eval" "${branch}/train" "${branch}/eval"
  if [[ ! -f "${branch}/small/train/TRAIN_SUCCESS.json" ]]; then
    wait_cuda
    "${PY}" "${ROOT}/train_p3d2_grounded_reader.py" --model "${MODEL}" --protocol "${PROTOCOL}" \
      --teacher-train "${ROOT}/teacher_cache/train/index.json" --teacher-validation "${ROOT}/teacher_cache/validation/index.json" \
      --configuration "${configuration}" --shared-blocks "${blocks}" --mode small --small-samples 16 \
      --epochs "${small_epochs}" --lr 5e-4 --out "${branch}/small/train" --device cuda
  fi
  if [[ ! -f "${branch}/small/eval/SUCCESS.json" ]]; then
    wait_cuda
    "${PY}" "${ROOT}/eval_p3d2_grounded_reader.py" --model "${MODEL}" --protocol "${PROTOCOL}" \
      --teacher "${ROOT}/teacher_cache/train/index.json" --checkpoint "${branch}/small/train/checkpoint_best.pt" \
      --split train --max-samples 16 --out "${branch}/small/eval" --device cuda
  fi
  if [[ ! -f "${branch}/train/TRAIN_SUCCESS.json" ]]; then
    wait_cuda
    "${PY}" "${ROOT}/train_p3d2_grounded_reader.py" --model "${MODEL}" --protocol "${PROTOCOL}" \
      --teacher-train "${ROOT}/teacher_cache/train/index.json" --teacher-validation "${ROOT}/teacher_cache/validation/index.json" \
      --configuration "${configuration}" --shared-blocks "${blocks}" --mode full --epochs 3 --lr 2e-4 \
      --init-checkpoint "${branch}/small/train/checkpoint_best.pt" --out "${branch}/train" --device cuda
  fi
  if [[ ! -f "${branch}/eval/SUCCESS.json" ]]; then
    wait_cuda
    "${PY}" "${ROOT}/eval_p3d2_grounded_reader.py" --model "${MODEL}" --protocol "${PROTOCOL}" \
      --teacher "${ROOT}/teacher_cache/test/index.json" --checkpoint "${branch}/train/checkpoint_best.pt" \
      --split test --out "${branch}/eval" --device cuda
  fi
}

run_all() {
  mkdir -p "${ROOT}/audit" "${ROOT}/step1_oracle" "${ROOT}/teacher_cache" "${ROOT}/step3_capacity" "${ROOT}/logs"
  if [[ ! -f "${ROOT}/audit/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/audit_p3d2.py" --protocol "${PROTOCOL}" --p3d-reader "${P3D_READER}" --model "${MODEL}" --out "${ROOT}/audit/SUCCESS.json"
  fi

  # Step 1: no training; ordinary versus token/layer Oracle grounding.
  if [[ ! -f "${ROOT}/step1_oracle/SUCCESS.json" ]]; then
    wait_cuda
    "${PY}" "${ROOT}/eval_p3d2_oracle.py" --model "${MODEL}" --protocol "${PROTOCOL}" --reader "${P3D_READER}" \
      --out "${ROOT}/step1_oracle" --max-samples 64 --top-groups 4 --device cuda
  fi

  # Step 2: cache frozen receiver-native execution and frozen span-probe teachers once.
  cache_teacher train
  cache_teacher validation
  cache_teacher test

  # Step 3: identical grounded objective, four controlled injection-capacity variants.
  train_configuration uniform8 4 30
  train_configuration midlate8 4 30
  train_configuration key4 4 30
  train_configuration all36 6 30

  "${PY}" "${ROOT}/summarize_p3d2.py" --root "${ROOT}" --native-result "${NATIVE_RESULT}"
}

status() {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  pgrep -af 'eval_p3d2_oracle.py|cache_p3d2_teachers.py|train_p3d2_grounded_reader.py|eval_p3d2_grounded_reader.py|run_steps_1_3.sh all' || true
  find "${ROOT}" -maxdepth 6 -type f \( -name SUCCESS.json -o -name TRAIN_SUCCESS.json -o -name STEPS_1_3_SUCCESS.json \) -print 2>/dev/null | sort
  [[ -f "${ROOT}/p3d2_run.log" ]] && tail -n 5 "${ROOT}/p3d2_run.log"
}

case "${1:-all}" in
  all) run_all ;;
  status) status ;;
  *) echo "Usage: bash run_steps_1_3.sh {all|status}" >&2; exit 64 ;;
esac
