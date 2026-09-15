#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/yezhe/伪查询/runs/p3e_m_fresh_native_reader_q_conditioning_seed1234
nohup bash "${ROOT}/run_all.sh" > "${ROOT}/p3e_m_run.log" 2>&1 &
echo $! > "${ROOT}/p3e_m_run.pid"
echo "started pid=$(cat "${ROOT}/p3e_m_run.pid")"
