#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
HOST="${SPIKESSL_HOST:-0.0.0.0}"
PORT="${SPIKESSL_PORT:-7860}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

cd "$ROOT"
mkdir -p result
exec env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$GPU" \
  "$PYTHON_BIN" -u webui/app.py --host "$HOST" --port "$PORT"
