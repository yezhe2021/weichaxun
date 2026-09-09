#!/usr/bin/env bash
set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

"$ROOT/start_when_gpu_ready.sh" smoke
"$ROOT/start_when_gpu_ready.sh" development
