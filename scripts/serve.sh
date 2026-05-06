#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "$(uname -s)" in
    Darwin) BACKEND="macos" ;;
    Linux)  BACKEND="linux" ;;
    *)
        echo "Unsupported platform: $(uname -s)" >&2
        exit 1
        ;;
esac

exec bash "${SCRIPT_DIR}/${BACKEND}/serve.sh" "$@"
