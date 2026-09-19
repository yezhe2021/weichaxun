#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/yezhe/伪查询/runs/p3e_n_native_oracle_bottleneck_decomposition_seed1234
nohup bash "${ROOT}/run_all.sh" > "${ROOT}/p3e_n_run.log" 2>&1 &
echo $! > "${ROOT}/p3e_n_run.pid"
echo "started pid=$(cat "${ROOT}/p3e_n_run.pid")"
