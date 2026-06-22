# LLM 本地部署

跨平台 LLM 本地推理环境：macOS 使用 MLX，Linux 使用 vLLM。

## 环境要求

| | macOS | Linux |
|---|---|---|
| 硬件 | Apple Silicon (M1-M4), 24GB+ | NVIDIA GPU, 24GB+ VRAM |
| 系统 | macOS Sonoma+ | Ubuntu 22.04+ |
| 工具 | Xcode CLT (Metal) | NVIDIA 驱动 + CUDA |

> **vllm<0.20 原因**：vllm 0.20+ 依赖 PyTorch/CUDA >= 12.9，若宿主 CUDA < 13.0 则需限定 `vllm<0.20`。若宿主 CUDA >= 12.9，可放开此限制。

## 快速开始

### 1. 安装环境

```bash
# macOS
conda env create -f env/environment-macos.yaml

# Linux
conda env create -f env/environment-linux.yaml

conda activate agent-local
```

### 2. 配置

复制并编辑环境配置：

```bash
cp .env.example .env
# 编辑 .env 设置 BASE_URL, API_KEY, MODEL, OPENAI_PORT, ANTHROPIC_PORT
```

### 3. 下载模型

```bash
python main.py download qwen3.5-9b    # 9B AWQ, ~6GB
python main.py download qwen3.5-35b   # 35B AWQ, ~6GB
python main.py download qwen3.6-27b   # 27B AWQ, ~16GB

# 国内加速:
# HF_ENDPOINT=https://hf-mirror.com python main.py download qwen3.5-9b
```

### 4. 启动 API 服务

```bash
python main.py serve
```

### 5. 开始聊天

在另一个终端窗口:

```bash
conda activate agent-local
python main.py chat
```

Agent 支持工具调用（Shell 工具）和丰富的对话管理功能：
- `/save [name]` — 保存当前对话
- `/load <name>` — 加载存档
- `/list` — 列出所有存档
- `/clear` — 清空上下文
- `/branch` — 创建对话分支
- `/switch` — 切换对话分支
- `/undo` — 撤销上一步
- `/copy` — 复制对话
- `/exit` — 退出

### 6. 直接调用 API

OpenAI 兼容端点:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

### 7. 性能基准测试

```bash
python tests/benchmark.py --prompts 5 --max-tokens 256
```

## Docker 部署

### 构建

`CUDA_VERSION` 默认匹配当前宿主驱动，可通过环境变量覆盖。

```bash
# 自动检测，或手动指定
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2) \
  docker compose -f docker/docker-compose.yml build
```

### 下载模型

```bash
docker compose -f docker/docker-compose.yml run --rm agent-local python main.py download qwen3.5-9b
```

模型下载到 Docker Volume `llm-cache`，只需下载一次。

### 启动服务

```bash
docker compose -f docker/docker-compose.yml up -d
```

OpenAI 端点: `http://127.0.0.1:8000/v1`。

若需更换模型，设置 `MODEL` 环境变量后重新下载即可：

```bash
MODEL=qwen3.5-35b docker compose -f docker/docker-compose.yml run --rm agent-local python main.py download qwen3.5-35b
MODEL=qwen3.5-35b docker compose -f docker/docker-compose.yml up -d
```

## 模型管理

### 新增模型

1. 编辑 `local-models.yaml`，在对应平台下添加模型条目：
   ```yaml
   new-model:
     macos:
       repo: mlx-community/NewModel-4bit
     linux:
       repo: org/NewModel
   ```
2. 下载：`python main.py download new-model`

### 移除模型

```bash
# 1. 从 local-models.yaml 中删除对应条目
# 2. 删除 HF 缓存
rm -rf .cache/huggingface/models--<org>--<model-name>
```
