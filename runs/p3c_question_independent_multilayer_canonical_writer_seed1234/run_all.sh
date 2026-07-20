#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3c_question_independent_multilayer_canonical_writer_seed1234
P3B=/home/yezhe/伪查询/runs/p3b_multilayer_question_independent_kv_readability_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
PROJECTIONS=${P3B}/projections/layerwise_pca_random.pt
PARALLEL_JOBS=${P3C_PARALLEL_JOBS:-2}

export PYTHONPATH=${ROOT}:${P3B}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

wait_cuda() {
  until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do sleep 60; done
}

teacher_path() {
  echo "${P3B}/full/evidence_only/native_kv/$1/checkpoint_best.pt"
}

native_result_path() {
  echo "${P3B}/full/evidence_only/native_kv/$1/SUCCESS.json"
}

run_branch() {
  local layer_config="$1" seed="$2" phase="${3:-formal}"
  local branch
  if [[ "${phase}" == overfit ]]; then
    branch="${ROOT}/overfit/${layer_config}/seed${seed}"
  else
    branch="${ROOT}/branches/${layer_config}/seed${seed}"
  fi
  mkdir -p "${branch}/writer" "${branch}/cache" "${branch}/fresh_probe"
  local epochs=3 max_train=0 max_validation=0
  if [[ "${phase}" == overfit ]]; then epochs=60; max_train=16; max_validation=16; fi
  if [[ ! -f "${branch}/writer/TRAIN_SUCCESS.json" ]]; then
    wait_cuda
    local validation_cache="${P3B}/cache/validation/index.json"
    [[ "${phase}" == overfit ]] && validation_cache="${P3B}/cache/train/index.json"
    "${PY}" "${ROOT}/train_p3c_writer.py" \
      --train-cache "${P3B}/cache/train/index.json" --validation-cache "${validation_cache}" \
      --teacher-checkpoint "$(teacher_path "${layer_config}")" --projections "${PROJECTIONS}" \
      --layer-config "${layer_config}" --out "${branch}/writer" --epochs "${epochs}" \
      --max-train "${max_train}" --max-validation "${max_validation}" \
      --lr-writer "$([[ "${phase}" == overfit ]] && echo 5e-4 || echo 2e-4)" \
      --lr-probe "$([[ "${phase}" == overfit ]] && echo 1e-3 || echo 2e-4)" --seed "${seed}" --device cuda
  fi
  [[ "${phase}" == overfit ]] && return
  for split in train validation test; do
    if [[ ! -f "${branch}/cache/${split}/SUCCESS.json" ]]; then
      wait_cuda
      "${PY}" "${ROOT}/cache_p3c_canonical.py" --native-cache "${P3B}/cache/${split}/index.json" \
        --writer "${branch}/writer/writer_best.pt" --projections "${PROJECTIONS}" \
        --out "${branch}/cache/${split}" --device cuda
    fi
  done
  if [[ ! -f "${branch}/fresh_probe/SUCCESS.json" ]]; then
    wait_cuda
    "${PY}" "${ROOT}/train_eval_p3c_fresh_probe.py" \
      --train-cache "${branch}/cache/train/index.json" --validation-cache "${branch}/cache/validation/index.json" \
      --test-cache "${branch}/cache/test/index.json" --native-result "$(native_result_path "${layer_config}")" \
      --out "${branch}/fresh_probe" --epochs 3 --seed "${seed}" --device cuda
  fi
}

run_parallel() {
  local max_jobs="$1"; shift
  local active=0 command
  for command in "$@"; do
    bash -lc "${command}" &
    active=$((active + 1))
    if (( active >= max_jobs )); then wait -n; active=$((active - 1)); fi
  done
  wait
}

run_all() {
  mkdir -p "${ROOT}/audit" "${ROOT}/gate" "${ROOT}/logs"
  if [[ ! -f "${ROOT}/audit/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/audit_p3c.py" --p3b "${P3B}" --out "${ROOT}/audit/SUCCESS.json"
  fi
  if [[ ! -f "${ROOT}/gate/SUCCESS.json" ]]; then
    bash "${ROOT}/run_all.sh" branch all36 1234 overfit > "${ROOT}/logs/overfit_all36_seed1234.log" 2>&1
    "${PY}" "${ROOT}/check_p3c_gate.py" \
      --result "${ROOT}/overfit/all36/seed1234/writer/TRAIN_SUCCESS.json" --out "${ROOT}/gate/SUCCESS.json"
  fi
  commands=()
  for layer_config in all36 uniform16; do
    for seed in 1234 2345 3456; do
      commands+=("cd '${ROOT}' && bash run_all.sh branch '${layer_config}' '${seed}' formal > 'logs/${layer_config}_seed${seed}.log' 2>&1")
    done
  done
  run_parallel "${PARALLEL_JOBS}" "${commands[@]}"
  "${PY}" "${ROOT}/summarize_p3c.py" --root "${ROOT}"
}

status() {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  pgrep -af 'train_p3c_writer.py|cache_p3c_canonical.py|train_eval_p3c_fresh_probe.py|run_all.sh all' || true
  find "${ROOT}" -type f \( -name SUCCESS.json -o -name TRAIN_SUCCESS.json \) -print 2>/dev/null | sort
  for log in "${ROOT}"/logs/*.log; do [[ -f "${log}" ]] && { printf '\n== %s ==\n' "$(basename "${log}")"; tail -n 2 "${log}"; }; done
}

case "${1:-all}" in
  all) run_all ;;
  branch) shift; run_branch "$@" ;;
  status) status ;;
  *) echo "Usage: bash run_all.sh {all|status|branch LAYERS SEED [formal|overfit]}" >&2; exit 64 ;;
esac
