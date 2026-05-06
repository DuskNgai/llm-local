#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${CONDA_PREFIX:-${VIRTUAL_ENV:-}}/bin/python"
[ -x "${PYTHON}" ] || PYTHON="$(which python3 2>/dev/null || which python)"
HOST="${HOST:-127.0.0.1}"
export OPENAI_PORT="${OPENAI_PORT:-8000}"
export ANTHROPIC_PORT="${ANTHROPIC_PORT:-8001}"

cleanup() {
    kill $OPENAI_PID $ANTHROPIC_PID 2>/dev/null
    wait $OPENAI_PID $ANTHROPIC_PID 2>/dev/null
}
trap cleanup EXIT INT TERM

bash "${SCRIPT_DIR}/serve-openai.sh" &
OPENAI_PID=$!

"${PYTHON}" "${SCRIPT_DIR}/serve_anthropic.py" &
ANTHROPIC_PID=$!

echo "OpenAI endpoint:    http://${HOST}:${OPENAI_PORT}/v1"
echo "Anthropic endpoint: http://${HOST}:${ANTHROPIC_PORT}/v1/messages"

wait
