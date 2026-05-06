#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${CONDA_PREFIX:-${VIRTUAL_ENV:-}}/bin/python"
[ -x "${PYTHON}" ] || PYTHON="$(which python3 2>/dev/null || which python)"

MODEL_PATH="$("${PYTHON}" "${SCRIPT_DIR}/../resolve_model.py" "${MODEL:?}")"
HOST="${HOST:-127.0.0.1}"
export OPENAI_PORT="${OPENAI_PORT:-8000}"
export ANTHROPIC_PORT="${ANTHROPIC_PORT:-8001}"

cleanup() {
    kill $OPENAI_PID $ANTHROPIC_PID 2>/dev/null
    wait $OPENAI_PID $ANTHROPIC_PID 2>/dev/null
}
trap cleanup EXIT INT TERM

MAX_LEN="${MAX_MODEL_LEN:-8192}"
vllm serve "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${OPENAI_PORT}" \
    --max-model-len "${MAX_LEN}" &
OPENAI_PID=$!

export MODEL="${MODEL:?}"
"${PYTHON}" "${SCRIPT_DIR}/../serve_anthropic.py" &
ANTHROPIC_PID=$!

echo "OpenAI endpoint:    http://${HOST}:${OPENAI_PORT}/v1"
echo "Anthropic endpoint: http://${HOST}:${ANTHROPIC_PORT}/v1/messages"

wait
