#!/usr/bin/env bash
set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"
"$ROOT/start_when_gpu_ready.sh" smoke
"$ROOT/start_when_gpu_ready.sh" development
