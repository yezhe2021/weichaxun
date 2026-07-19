#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${ROOT}/../.." && pwd)"
PY="${PY:-/home/yezhe/data/miniconda3/envs/attnkv/bin/python}"
P2IW_ROOT="${P2IW_ROOT:-${PROJECT}/runs/p2iw_token_preserving_canonical_writer_qwen3_8b_seed1234}"
WRITER="${WRITER:-${P2IW_ROOT}/writer_full/best_checkpoint.pt}"
PROJECTIONS="${PROJECTIONS:-${P2IW_ROOT}/projections/pca_and_random.pt}"
TOKEN_TRAIN="${TOKEN_TRAIN:-${P2IW_ROOT}/cache/token_states/train/index.json}"
TOKEN_TEST="${TOKEN_TEST:-${P2IW_ROOT}/cache/token_states/test/index.json}"
CANONICAL_TRAIN="${ROOT}/cache/canonical/train/index.json"
CANONICAL_TEST="${ROOT}/cache/canonical/test/index.json"
MODEL4="${MODEL4:-/home/yezhe/all_models/models/Qwen/Qwen3-4B}"
MODEL35="${MODEL35:-/home/yezhe/all_models/models/Qwen/Qwen3___5-4B}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"

export TOKENIZERS_PARALLELISM=false
export P2IW_ROOT
cd "${PROJECT}"

wait_cuda() {
  until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do sleep 60; done
}

audit() {
  "${PY}" -m py_compile "${ROOT}"/*.py
  "${PY}" "${ROOT}/audit_p2ir.py" --p2iw-root "${P2IW_ROOT}" --writer-checkpoint "${WRITER}" \
    --token-train "${TOKEN_TRAIN}" --token-test "${TOKEN_TEST}" --receiver4 "${MODEL4}" --receiver35 "${MODEL35}" --out "${ROOT}/AUDIT.json"
  wait_cuda; "${PY}" "${ROOT}/smoke_runtime.py" --device "${DEVICE}"
}

cache_memory() {
  wait_cuda
  [[ -f "${CANONICAL_TRAIN}" ]] || "${PY}" "${ROOT}/cache_canonical_memory.py" --token-index "${TOKEN_TRAIN}" \
    --writer-checkpoint "${WRITER}" --projections "${PROJECTIONS}" --out "$(dirname "${CANONICAL_TRAIN}")" --device "${DEVICE}"
  [[ -f "${CANONICAL_TEST}" ]] || "${PY}" "${ROOT}/cache_canonical_memory.py" --token-index "${TOKEN_TEST}" \
    --writer-checkpoint "${WRITER}" --projections "${PROJECTIONS}" --out "$(dirname "${CANONICAL_TEST}")" --device "${DEVICE}"
  "${PY}" -c "import json; a=json.load(open('${CANONICAL_TRAIN}')); b=json.load(open('${CANONICAL_TEST}')); assert a['writer_checkpoint_sha256']==b['writer_checkpoint_sha256']; assert a['writer_state_sha256']==b['writer_state_sha256']"
}

train_small() {
  local name="$1" model="$2" dtype="$3" out="${ROOT}/$1/small/train"
  [[ -f "${out}/TRAIN_SUCCESS.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/train_reader.py" --receiver-name "${name}" --receiver-model "${model}" \
    --canonical-index "${CANONICAL_TRAIN}" --out "${out}" --mode small --small-pairs 16 \
    --epochs 40 --rank 64 --lr 5e-4 --seed "${SEED}" --device "${DEVICE}" --dtype "${dtype}"
}

eval_small() {
  local name="$1" model="$2" dtype="$3" out="${ROOT}/$1/small/eval"
  [[ -f "${out}/SUCCESS.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/eval_reader.py" --receiver-name "${name}" --receiver-model "${model}" \
    --canonical-index "${CANONICAL_TRAIN}" --checkpoint "${ROOT}/$1/small/train/checkpoint_best.pt" \
    --out "${out}" --max-pairs 16 --seed "${SEED}" --device "${DEVICE}" --dtype "${dtype}"
}

train_full() {
  local name="$1" model="$2" dtype="$3" out="${ROOT}/$1/full/train"
  [[ -f "${out}/TRAIN_SUCCESS.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/train_reader.py" --receiver-name "${name}" --receiver-model "${model}" \
    --canonical-index "${CANONICAL_TRAIN}" --out "${out}" --mode full \
    --init-checkpoint "${ROOT}/$1/small/train/checkpoint_best.pt" --epochs 4 --rank 64 \
    --lr 2e-4 --seed "${SEED}" --device "${DEVICE}" --dtype "${dtype}"
}

eval_full() {
  local name="$1" model="$2" dtype="$3" out="${ROOT}/$1/full/eval"
  [[ -f "${out}/SUCCESS.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/eval_reader.py" --receiver-name "${name}" --receiver-model "${model}" \
    --canonical-index "${CANONICAL_TEST}" --checkpoint "${ROOT}/$1/full/train/checkpoint_best.pt" \
    --out "${out}" --max-pairs 64 --seed "${SEED}" --device "${DEVICE}" --dtype "${dtype}"
}

run_receiver() {
  train_small "$1" "$2" "$3"
  eval_small "$1" "$2" "$3"
  printf 'Small-overfit diagnostics completed for %s; continuing regardless of metric thresholds.\n' "$1"
  train_full "$1" "$2" "$3"
  eval_full "$1" "$2" "$3"
}

summarize() { "${PY}" "${ROOT}/summarize_p2ir.py" --root "${ROOT}"; }

status() {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  pgrep -af 'cache_canonical_memory.py|train_reader.py|eval_reader.py|run_all.sh all' || true
  find "${ROOT}" -maxdepth 5 -type f \( -name 'AUDIT.json' -o -name 'TRAIN_SUCCESS.json' -o -name 'SUCCESS.json' -o -name 'CACHE_SUCCESS.json' \) -print 2>/dev/null | sort
}

case "${1:-help}" in
  audit) audit ;; cache) cache_memory ;;
  q4) run_receiver qwen3_4b "${MODEL4}" float16 ;;
  q35) run_receiver qwen3_5_4b "${MODEL35}" float32 ;;
  summarize) summarize ;; status) status ;;
  all)
    audit
    cache_memory
    run_receiver qwen3_4b "${MODEL4}" float16
    run_receiver qwen3_5_4b "${MODEL35}" float32
    summarize
    ;;
  *) echo "Usage: bash run_all.sh {audit|cache|q4|q35|summarize|status|all}" >&2; exit 64 ;;
esac
