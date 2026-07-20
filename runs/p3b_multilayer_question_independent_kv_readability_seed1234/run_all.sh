#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3b_multilayer_question_independent_kv_readability_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
MODEL8=/home/yezhe/all_models/models/Qwen/Qwen3-8B
P3A=/home/yezhe/伪查询/runs/p3a_hotpot_canonical_responsibility_seed1234
TRAIN_SOURCE=${P3A}/data/train512.jsonl
TEST_SOURCE=${P3A}/data/dev500.jsonl
DATA=${ROOT}/data
CACHE=${ROOT}/cache
PROJECTIONS=${ROOT}/projections/layerwise_pca_random.pt
PARALLEL_JOBS=${P3B_PARALLEL_JOBS:-2}

export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

wait_cuda() {
  until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do
    sleep 60
  done
}

run_cache() {
  local split="$1"
  [[ -f "${CACHE}/${split}/SUCCESS.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/cache_p3b_states.py" \
    --model "${MODEL8}" --data "${DATA}/${split}.jsonl" --out "${CACHE}/${split}" --device cuda
}

run_probe() {
  local phase="$1" mode="$2" source="$3" layers="$4" epochs="$5" max_train="$6" max_validation="$7" max_test="$8"
  local out="${ROOT}/${phase}/${mode}/${source}"
  local validation_cache="${CACHE}/validation/index.json"
  local test_cache="${CACHE}/test/index.json"
  if [[ "${phase}" == overfit ]]; then
    validation_cache="${CACHE}/train/index.json"
    test_cache="${CACHE}/train/index.json"
  fi
  [[ -f "${out}/SUCCESS.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/train_eval_p3b_probe.py" \
    --train-cache "${CACHE}/train/index.json" \
    --validation-cache "${validation_cache}" \
    --test-cache "${test_cache}" \
    --projections "${PROJECTIONS}" \
    --sender-mode "${mode}" --source "${source}" --layer-sets "${layers}" \
    --out "${out}" --epochs "${epochs}" --max-train "${max_train}" \
    --max-validation "${max_validation}" --max-test "${max_test}" \
    --lr "$([[ "${phase}" == overfit ]] && echo 1e-3 || echo 2e-4)" \
    --gradient-accumulation "$([[ "${phase}" == overfit ]] && echo 1 || echo 4)" \
    --seed 1234 --device cuda
}

run_parallel_commands() {
  local max_jobs="$1"
  shift
  local active=0 command
  for command in "$@"; do
    bash -lc "${command}" &
    active=$((active + 1))
    if (( active >= max_jobs )); then
      wait -n
      active=$((active - 1))
    fi
  done
  wait
}

run_all() {
  mkdir -p "${ROOT}/audit" "${DATA}" "${CACHE}" "${ROOT}/projections" "${ROOT}/gate" "${ROOT}/logs"
  if [[ ! -f "${ROOT}/audit/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/audit_p3b.py" --model "${MODEL8}" --train-source "${TRAIN_SOURCE}" \
      --test-source "${TEST_SOURCE}" --root "${ROOT}" --out "${ROOT}/audit/SUCCESS.json"
  fi
  if [[ ! -f "${DATA}/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/prepare_p3b_data.py" --train-source "${TRAIN_SOURCE}" --test-source "${TEST_SOURCE}" \
      --out "${DATA}" --validation-size 64 --seed 1234
  fi

  # Qwen3-8B caching is deliberately sequential on the single 32 GiB V100.
  run_cache train
  run_cache validation
  run_cache test

  if [[ ! -f "${PROJECTIONS%.pt}.json" ]]; then
    wait_cuda
    "${PY}" "${ROOT}/fit_p3b_projections.py" --cache "${CACHE}/train/index.json" \
      --out "${PROJECTIONS}" --output-dim 256 --tokens-per-sample 4 --seed 1234 --device cuda
  fi

  overfit_commands=()
  for source in hidden native_kv trainable; do
    overfit_commands+=("cd '${ROOT}' && bash run_all.sh probe overfit evidence_only '${source}' all36 40 16 16 16 > 'logs/overfit_${source}.log' 2>&1")
  done
  run_parallel_commands "${PARALLEL_JOBS}" "${overfit_commands[@]}"
  "${PY}" "${ROOT}/check_p3b_gate.py" --root "${ROOT}" --out "${ROOT}/gate/SUCCESS.json"

  full_commands=()
  for mode in evidence_only question_evidence; do
    for source in hidden native_kv pca random trainable; do
      full_commands+=("cd '${ROOT}' && bash run_all.sh probe full '${mode}' '${source}' last1,last4,last8,uniform16,all36 3 0 0 0 > 'logs/full_${mode}_${source}.log' 2>&1")
    done
  done
  run_parallel_commands "${PARALLEL_JOBS}" "${full_commands[@]}"
  "${PY}" "${ROOT}/summarize_p3b.py" --root "${ROOT}"
}

status() {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  pgrep -af 'cache_p3b_states.py|fit_p3b_projections.py|train_eval_p3b_probe.py|run_all.sh all' || true
  find "${ROOT}" -type f \( -name SUCCESS.json -o -name TRAIN_SUCCESS.json \) -print 2>/dev/null | sort
  for log in "${ROOT}"/logs/*.log; do
    [[ -f "${log}" ]] || continue
    printf '\n== %s ==\n' "$(basename "${log}")"
    tail -n 2 "${log}"
  done
}

case "${1:-all}" in
  all)
    run_all
    ;;
  probe)
    shift
    run_probe "$@"
    ;;
  status)
    status
    ;;
  *)
    echo "Usage: bash run_all.sh {all|status|probe ...}" >&2
    exit 64
    ;;
esac
