#!/bin/bash
# 国内可设置镜像: export HF_ENDPOINT=https://hf-mirror.com
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ALIAS="${1:-qwen3-8b}"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
HF_HOME="${PROJECT_DIR}/.cache/huggingface"

case "${ALIAS}" in
    qwen3.6-27b) REPO="mlx-community/Qwen3.6-27B-4bit" ;;
    qwen3-30b)   REPO="mlx-community/Qwen3-30B-A3B-4bit" ;;
    qwen3-8b)    REPO="mlx-community/Qwen3-8B-4bit" ;;
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

# Auto-register in models.yaml
MODELS_FILE="${PROJECT_DIR}/models.yaml"

if ! grep -q "^${ALIAS}:" "${MODELS_FILE}" 2>/dev/null; then
    cat <<EOF >> "${MODELS_FILE}"

${ALIAS}:
  repo: ${REPO}
EOF
    echo "Registered ${ALIAS} in models.yaml"
fi
