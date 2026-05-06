# LLM 本地部署

跨平台 LLM 本地推理环境：macOS 使用 MLX，Linux 使用 vLLM。

## 环境要求

| | macOS | Linux |
|---|---|---|
| 硬件 | Apple Silicon (M1-M4), 24GB+ | NVIDIA GPU, 24GB+ VRAM |
| 系统 | macOS Sonoma+ | Ubuntu 22.04+ |
| 工具 | Xcode CLT (Metal) | NVIDIA 驱动 + CUDA |

## 快速开始

### 1. 安装环境

```bash
# macOS
conda env create -f environment-macos.yaml

# Linux
conda env create -f environment-linux.yaml

conda activate llm-local
```

### 2. 下载模型

```bash
bash scripts/download-model.sh qwen3-8b      # 8B AWQ, ~5GB
bash scripts/download-model.sh qwen3-30b     # 30B MoE AWQ, ~18GB
bash scripts/download-model.sh qwen3.6-27b   # 27B AWQ, ~16GB

# 国内加速:
# HF_ENDPOINT=https://hf-mirror.com bash scripts/download-model.sh qwen3-8b
```

### 3. 启动 API 服务

```bash
MODEL=qwen3-8b bash scripts/serve.sh         # 自动检测平台启动对应后端
```

### 4. 开始聊天

在另一个终端窗口:

```bash
conda activate llm-local
python scripts/chat.py                      # 默认 OpenAI, 流式输出
python scripts/chat.py --no-stream          # 非流式输出
python scripts/chat.py --backend anthropic  # Anthropic 协议
```

### 5. 直接调用 API

OpenAI 兼容端点:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

Anthropic 兼容端点:

```bash
curl http://127.0.0.1:8001/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'
```

### 6. 性能基准测试

```bash
python scripts/benchmark.py --prompts 5 --max-tokens 256
```

## 平台架构

```
scripts/serve.sh / download-model.sh   ← 统一入口，自动检测平台
├── serve_anthropic.py                 ← Anthropic API 代理（双平台共用）
├── resolve_model.py                   ← 模型别名解析（双平台共用）
├── scripts/macos/                     ← MLX 后端 (Apple Silicon)
│   ├── serve.sh
│   └── download-model.sh
└── scripts/linux/                     ← vLLM 后端 (NVIDIA GPU)
    ├── serve.sh
    └── download-model.sh
```

## 模型管理

### 新增模型

1. 编辑 `models.yaml`，在对应平台下添加模型条目：
   ```yaml
   new-model:
     macos:
       repo: mlx-community/NewModel-4bit
     linux:
       repo: org/NewModel
   ```
2. 编辑对应平台的 `scripts/<platform>/download-model.sh`，在 `case` 语句中添加映射。
3. 下载：`bash scripts/download-model.sh new-model`

### 移除模型

```bash
# 1. 从 models.yaml 中删除对应条目
# 2. 删除 HF 缓存
rm -rf .cache/huggingface/models--<org>--<model-name>
```

## 对话管理

```
/save [name]    保存当前对话到 .cache/chats/
/load <name>    加载存档
/list           列出所有存档
/clear          清空上下文
/exit           退出
```
