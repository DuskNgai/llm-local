#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${CONDA_PREFIX:-${VIRTUAL_ENV:-}}/bin/python"
[ -x "${PYTHON}" ] || PYTHON="$(which python3 2>/dev/null || which python)"
MODEL_ALIAS="${MLX_MODEL:?}"
MODEL_PATH="$("${PYTHON}" "${SCRIPT_DIR}/resolve_model.py" "${MODEL_ALIAS}")"
HOST="${HOST:-127.0.0.1}"
export OPENAI_PORT="${OPENAI_PORT:-8000}"

exec "${PYTHON}" -m mlx_lm server \
    --model "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${OPENAI_PORT}"
