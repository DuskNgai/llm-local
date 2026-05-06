#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ALIAS="${1:-qwen3-8b}"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HF_HOME="${PROJECT_DIR}/.cache/huggingface"

case "${ALIAS}" in
    qwen3.6-27b) REPO="Qwen/Qwen3.6-27B-AWQ" ;;
    qwen3-30b)   REPO="Qwen/Qwen3-30B-A3B-AWQ" ;;
    qwen3-8b)    REPO="Qwen/Qwen3-8B-AWQ" ;;
    *)
        echo "Unknown model: ${ALIAS}"
        echo "Available: qwen3.6-27b, qwen3-30b, qwen3-8b"
        exit 1
        ;;
esac

echo "Project root: ${PROJECT_DIR}"
echo "HF cache:     ${HF_HOME}"
echo "Model:        ${ALIAS}"

hf download "${REPO}" \
    --cache-dir "${HF_HOME}"

echo "Model downloaded to Hugging Face cache at ${HF_HOME}"

MODELS_FILE="${PROJECT_DIR}/models.yaml"

if ! grep -qF "^${ALIAS}:" "${MODELS_FILE}" 2>/dev/null; then
    cat <<EOF >> "${MODELS_FILE}"

${ALIAS}:
  linux:
    repo: ${REPO}
EOF
    echo "Registered ${ALIAS} in models.yaml"
fi
