#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
MODEL="${MODEL:-/home/yezhe/all_models/models/Qwen/Qwen3-8B}"
P2A2="${P2A2:-${PROJECT}/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234}"
NATIVE_TRAIN="${NATIVE_TRAIN:-${P2A2}/cache_native_kv_pairs/train/index.json}"
NATIVE_TEST="${NATIVE_TEST:-${P2A2}/cache_native_kv_pairs/test/index.json}"
TOKEN_TRAIN="${ROOT}/cache/token_states/train/index.json"
TOKEN_TEST="${ROOT}/cache/token_states/test/index.json"
PROJECTIONS="${ROOT}/projections/pca_and_random.pt"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
STRICT_GATES="${STRICT_GATES:-0}"

export TOKENIZERS_PARALLELISM=false
cd "${PROJECT}"

wait_cuda() {
  until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do
    sleep 60
  done
}

gate_or_warn() {
  local label="$1"
  shift
  if "$@"; then
    return 0
  fi
  if [[ "${STRICT_GATES}" == "1" ]]; then
    printf 'Validity gate failed and STRICT_GATES=1: %s\n' "${label}" >&2
    return 2
  fi
  printf 'WARNING: validity gate failed, continuing the diagnostic pipeline: %s\n' "${label}" >&2
  return 0
}

audit() {
  "${PY}" -m py_compile "${ROOT}"/*.py
  "${PY}" "${ROOT}/audit_p2iw.py" --model "${MODEL}" \
    --native-train "${NATIVE_TRAIN}" --native-test "${NATIVE_TEST}" --out "${ROOT}/AUDIT.json"
  wait_cuda
  "${PY}" "${ROOT}/smoke_runtime.py" --device "${DEVICE}"
}

cache_states() {
  wait_cuda
  if [[ ! -f "${TOKEN_TRAIN}" ]]; then
    "${PY}" "${ROOT}/cache_token_states.py" --model "${MODEL}" --native-index "${NATIVE_TRAIN}" \
      --out "$(dirname "${TOKEN_TRAIN}")" --device "${DEVICE}" --dtype float16
  fi
  wait_cuda
  if [[ ! -f "${TOKEN_TEST}" ]]; then
    "${PY}" "${ROOT}/cache_token_states.py" --model "${MODEL}" --native-index "${NATIVE_TEST}" \
      --out "$(dirname "${TOKEN_TEST}")" --device "${DEVICE}" --dtype float16
  fi
}

fit_projections() {
  [[ -f "${PROJECTIONS}" ]] && return
  wait_cuda
  mkdir -p "$(dirname "${PROJECTIONS}")"
  "${PY}" "${ROOT}/fit_projections.py" --index "${TOKEN_TRAIN}" --out "${PROJECTIONS}" \
    --train-pairs 448 --max-tokens 12000 --seed "${SEED}" --device "${DEVICE}"
}

probe() {
  local source="$1" out="$2" extra=()
  [[ -f "${out}/SUCCESS.json" ]] && return
  if [[ "${source}" == writer ]]; then
    extra+=(--writer-checkpoint "${ROOT}/writer_full/best_checkpoint.pt")
  fi
  wait_cuda
  "${PY}" "${ROOT}/train_probe.py" --train-index "${TOKEN_TRAIN}" --test-index "${TOKEN_TEST}" \
    --projections "${PROJECTIONS}" --source "${source}" --out "${out}" \
    --epochs 30 --patience 5 --seed "${SEED}" --device "${DEVICE}" "${extra[@]}"
}

baselines() {
  probe teacher "${ROOT}/baselines/teacher"
  gate_or_warn hidden_teacher_probe "${PY}" -c "import json; x=json.load(open('${ROOT}/baselines/teacher/SUCCESS.json')); raise SystemExit(0 if x['base_accuracy'] >= .95 and x['counterfactual_accuracy'] >= .95 and x['correct_paired_consistency'] >= .90 else 2)"
  probe random "${ROOT}/baselines/random"
  probe pca "${ROOT}/baselines/pca"
}

small_overfit() {
  [[ -f "${ROOT}/small_overfit/SUCCESS.json" ]] || {
    wait_cuda
    "${PY}" "${ROOT}/train_writer.py" --index "${TOKEN_TRAIN}" --projections "${PROJECTIONS}" \
      --out "${ROOT}/small_overfit" --mode small --subset 16 --epochs 120 --patience 30 \
      --batch-pairs 4 --lr 2e-3 --rank 64 --seed "${SEED}" --device "${DEVICE}"
  }
  gate_or_warn small_writer_overfit "${PY}" -c "import json; x=json.load(open('${ROOT}/small_overfit/SUCCESS.json')); raise SystemExit(0 if x['small_overfit_gate']['passed'] else 2)"
}

train_full() {
  [[ -f "${ROOT}/writer_full/SUCCESS.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/train_writer.py" --index "${TOKEN_TRAIN}" --projections "${PROJECTIONS}" \
    --out "${ROOT}/writer_full" --mode full --epochs 30 --patience 7 \
    --batch-pairs 4 --lr 5e-4 --rank 64 --seed "${SEED}" --device "${DEVICE}"
}

fresh_probe() {
  probe writer "${ROOT}/fresh_probe"
}

diagnostics() {
  [[ -f "${ROOT}/diagnostics/SUCCESS.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/diagnose_writer.py" --train-index "${TOKEN_TRAIN}" --test-index "${TOKEN_TEST}" \
    --projections "${PROJECTIONS}" --writer-checkpoint "${ROOT}/writer_full/best_checkpoint.pt" \
    --out "${ROOT}/diagnostics" --device "${DEVICE}"
}

summarize() {
  "${PY}" "${ROOT}/summarize_p2iw.py" --root "${ROOT}"
}

status() {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  pgrep -af 'cache_token_states.py|fit_projections.py|train_probe.py|train_writer.py|diagnose_writer.py' || true
  find "${ROOT}" -maxdepth 3 -type f \( -name 'AUDIT.json' -o -name 'SUCCESS.json' -o -name 'CACHE_SUCCESS.json' \) -print 2>/dev/null | sort
}

case "${1:-help}" in
  audit) audit ;;
  cache) cache_states ;;
  projections) fit_projections ;;
  baselines) baselines ;;
  small) small_overfit ;;
  train) train_full ;;
  fresh-probe) fresh_probe ;;
  diagnostics) diagnostics ;;
  summarize) summarize ;;
  status) status ;;
  all)
    audit
    cache_states
    fit_projections
    baselines
    small_overfit
    train_full
    fresh_probe
    diagnostics
    summarize
    ;;
  *) echo "Usage: bash run_all.sh {audit|cache|projections|baselines|small|train|fresh-probe|diagnostics|summarize|status|all}" >&2; exit 64 ;;
esac
