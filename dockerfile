ARG CUDA_VERSION=12.8
FROM nvidia/cuda:${CUDA_VERSION}.0-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip git git-lfs build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

RUN pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple \
    && pip install --no-cache-dir --default-timeout=600 --retries 10 \
    "vllm<0.20" \
    "huggingface_hub[cli]" \
    openai \
    anthropic \
    pyyaml

WORKDIR /app

COPY scripts/ ./scripts/
COPY models.yaml .

ENV MODEL=qwen3-8b \
    HOST=0.0.0.0 \
    OPENAI_PORT=8000 \
    ANTHROPIC_PORT=8001 \
    MAX_MODEL_LEN=8192 \
    VLLM_ENFORCE_EAGER=1 \
    VLLM_COMPILATION_CONFIG='{"mode":0}'

EXPOSE 8000 8001

VOLUME ["/app/.cache"]

ENTRYPOINT []
CMD ["bash", "scripts/serve.sh"]
