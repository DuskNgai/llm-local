# MLX-LM Mac 本地 LLM 部署

Apple Silicon Mac 上的 MLX 本地 LLM 推理环境。

## 环境要求

- macOS Sonoma+
- Apple Silicon (M1-M4)，24GB+ 统一内存
- Xcode Command Line Tools（Metal 编译器）

## 快速开始

### 1. 安装环境

```bash
conda env create -f environment.yaml
conda activate mlx
```

### 2. 下载模型

```bash
bash scripts/download-model.sh qwen3-8b     # 默认 8B, 5GB
bash scripts/download-model.sh qwen3-30b    # 30B MoE, 18GB
bash scripts/download-model.sh qwen3.6-27b  # 27B, 16GB

# 国内加速:
# HF_ENDPOINT=https://hf-mirror.com bash scripts/download-model.sh qwen3-8b
```

### 3. 启动 API 服务

```bash
MLX_MODEL=qwen3-8b bash scripts/serve.sh    # 指定模型
```

### 4. 开始聊天

在另一个终端窗口:

```bash
conda activate mlx
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

## 模型管理

### 新增模型

1. 编辑 `scripts/download-model.sh`，在 `case` 语句中添加模型映射：
   ```bash
   new-model) REPO="mlx-community/NewModel-4bit" ;;
   ```
2. 下载：
   ```bash
   bash scripts/download-model.sh new-model
   # 国内: HF_ENDPOINT=https://hf-mirror.com bash scripts/download-model.sh new-model
   ```
3. 使用：
   ```bash
   MLX_MODEL=new-model bash scripts/serve.sh
   ```

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
