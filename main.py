"""agent-local CLI -- unified entry point for serve, chat, download."""

import argparse
import atexit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import traceback
import urllib.request
import uuid

from dotenv import load_dotenv
import gnureadline
from huggingface_hub import snapshot_download
import yaml

from src.agent.agent import Agent, AgentExit
from src.agent.term import RED, RESET
from src.llm.client import create_client

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def cmd_serve(args):
    """Start the API server (platform backend + Anthropic proxy)."""
    plat = _detect_platform()
    model = os.environ.get("MODEL")
    openai_port_str = os.environ.get("OPENAI_PORT")
    anthropic_port_str = os.environ.get("ANTHROPIC_PORT")

    missing = []
    if not model:
        missing.append("MODEL")
    if not openai_port_str:
        missing.append("OPENAI_PORT")
    if not anthropic_port_str:
        missing.append("ANTHROPIC_PORT")
    if missing:
        sys.exit(f"Error: missing environment variables: {', '.join(missing)}")

    model_path = _resolve_model(model)
    host = os.environ.get("HOST", "127.0.0.1")
    openai_port = int(openai_port_str)
    anthropic_port = int(anthropic_port_str)

    proc = None
    proxy = None

    def cleanup():
        if proxy:
            proxy.shutdown()
        if proc and proc.poll() is None:
            proc.terminate()

    def _on_signal(*_):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    atexit.register(cleanup)

    # Start platform backend
    if plat == "macos":
        proc = subprocess.Popen([
            sys.executable,
            "-m",
            "mlx_lm",
            "server",
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            str(openai_port),
        ])
    elif plat == "linux":
        max_model_len = os.environ.get("MAX_MODEL_LEN", "8192")
        proc = subprocess.Popen([
            sys.executable,
            "-m",
            "vllm",
            "serve",
            model_path,
            "--host",
            host,
            "--port",
            str(openai_port),
            "--max-model-len",
            max_model_len,
        ])
    else:
        sys.exit(f"Unsupported platform: {plat}")

    # Start Anthropic proxy
    proxy = _start_anthropic_proxy(openai_port, anthropic_port, model)

    print(f"OpenAI endpoint:    http://{host}:{openai_port}/v1")
    print(f"Anthropic endpoint: http://{host}:{anthropic_port}/v1/messages")

    try:
        ret = proc.wait()
        if ret != 0:
            cleanup()
            sys.exit(ret)
    except KeyboardInterrupt:
        cleanup()


def _start_anthropic_proxy(openai_port: int, listen_port: int, model_name: str):
    """Start a lightweight Anthropic Messages API proxy in a daemon thread."""
    # Bypass system proxy for localhost backend requests
    no_proxy_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    urllib.request.install_opener(no_proxy_opener)

    BACKEND_URL = f"http://127.0.0.1:{openai_port}/v1/chat/completions"
    ANTHROPIC_VERSION = "2023-06-01"

    class ProxyHandler(BaseHTTPRequestHandler):

        def do_POST(self):
            if self.path != "/v1/messages":
                self.send_error(404, "Not Found")
                return
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(content_length))
                stream = body.get("stream", False)

                # Convert Anthropic to OpenAI format
                system_msgs = []
                if body.get("system"):
                    system_msgs.append({
                        "role": "system",
                        "content": _flatten_content(body["system"])
                    })
                user_msgs = []
                for m in body.get("messages", []):
                    user_msgs.append({
                        "role": m["role"],
                        "content": _flatten_content(m["content"])
                    })

                openai_req = {
                    "messages": system_msgs + user_msgs,
                    "max_tokens": body.get("max_tokens", 4096),
                    "temperature": body.get("temperature", 0.7),
                    "stream": stream,
                }

                req = urllib.request.Request(
                    BACKEND_URL,
                    data=json.dumps(openai_req).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json"
                    },
                )

                if stream:
                    self._handle_stream(req, body)
                else:
                    self._handle_non_stream(req)
            except Exception:
                traceback.print_exc()
                self.send_error(500, "Internal Server Error")

        def _handle_non_stream(self, req):
            with urllib.request.urlopen(req, timeout=300) as resp:
                openai_resp = json.loads(resp.read().decode("utf-8"))
            choice = openai_resp["choices"][0]
            msg = choice["message"]
            # Use content when present (even if ""), fall back to reasoning
            text = msg.get("content")
            if text is None:
                r = msg.get("reasoning")
                text = r if r is not None else ""
            openai_usage = openai_resp.get("usage", {})
            anthropic_usage = {
                "input_tokens": openai_usage.get("prompt_tokens", 0),
                "output_tokens": openai_usage.get("completion_tokens", 0),
            }
            anthropic_resp = {
                "id": openai_resp.get("id", ""),
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "content": [{
                    "type": "text",
                    "text": text
                }],
                "stop_reason": "end_turn",
                "usage": anthropic_usage,
            }
            self._send_json(anthropic_resp)

        def _handle_stream(self, req, original_body):
            # Open the backend connection first -- if this fails, let the
            # exception propagate to do_POST (no headers sent yet, safe).
            resp = urllib.request.urlopen(req, timeout=600)
            with resp:
                msg_id = f"msg_{uuid.uuid4().hex[:24]}"
                call_tokens = _estimate_tokens(original_body.get("messages", []), original_body.get("system"))
                max_tokens = json.loads(req.data.decode("utf-8")).get("max_tokens", 4096)

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("anthropic-version", ANTHROPIC_VERSION)
                self.end_headers()

                # Headers committed -- errors from here on must NOT propagate
                # to do_POST, which would try send_error() and corrupt the
                # response with a second HTTP status line.
                try:
                    self._write_sse(
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": msg_id,
                                "type": "message",
                                "role": "assistant",
                                "model": model_name,
                                "content": [],
                                "usage": {
                                    "input_tokens": call_tokens,
                                    "output_tokens": max_tokens
                                },
                            },
                        }
                    )

                    reply_text = ""
                    sent_block_start = False
                    for line in resp.fp:
                        line = line.decode("utf-8").strip()
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        chunk = json.loads(line[len("data: "):])
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")

                        if content:
                            if not sent_block_start:
                                self._write_sse(
                                    "content_block_start", {
                                        "type": "content_block_start",
                                        "index": 0,
                                        "content_block": {
                                            "type": "text",
                                            "text": ""
                                        },
                                    }
                                )
                                sent_block_start = True
                            self._write_sse(
                                "content_block_delta", {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": content
                                    },
                                }
                            )
                            reply_text += content

                    if sent_block_start:
                        self._write_sse("content_block_stop", {
                            "type": "content_block_stop",
                            "index": 0
                        })

                    self._write_sse(
                        "message_delta", {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": "end_turn"
                            },
                            "usage": {
                                "output_tokens": len(reply_text.split())
                            },
                        }
                    )
                    self._write_sse("message_stop", {
                        "type": "message_stop"
                    })
                except Exception:
                    traceback.print_exc()
                    # Write an SSE error event so the client knows the
                    # stream was interrupted, rather than getting a
                    # garbled second HTTP status line.
                    try:
                        self._write_sse("error", {
                            "type": "error",
                            "error": {
                                "type": "internal_error",
                                "message": "Stream interrupted",
                            },
                        })
                    except Exception:
                        pass # nothing more we can do

        def _write_sse(self, event, data):
            payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.flush()

        def _send_json(self, data):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("anthropic-version", ANTHROPIC_VERSION)
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        def log_message(self, *args, **kwargs):
            pass

    def _flatten_content(content):
        if isinstance(content, list):
            return "\n".join(b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text")
        return str(content) if content else ""

    def _estimate_tokens(messages, system) -> int:
        total = 0
        if system:
            total += len(_flatten_content(system))
        for m in (messages if messages is not None else []):
            total += len(_flatten_content(m.get("content", "")))
        return total // 4

    server = ThreadingHTTPServer(("127.0.0.1", listen_port), ProxyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def cmd_chat(_):
    """Start interactive agent chat."""
    base_url = os.environ.get("BASE_URL")
    api_key = os.environ.get("API_KEY")
    model = os.environ.get("MODEL")

    missing = []
    if not base_url:
        missing.append("BASE_URL")
    if not api_key:
        missing.append("API_KEY")
    if not model:
        missing.append("MODEL")
    if missing:
        print(f"Error: missing environment variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in the values.")
        return

    # Resolve model alias to local path for the API server
    model_path = _resolve_model(model)

    provider = os.environ.get("PROVIDER", "openai")
    client = create_client(provider, base_url, api_key, model_path)

    print(f"Agent [{model}] in streaming mode.")
    print("'/exit' to quit, '/clear' to reset, '/save' '/load' '/list' '/branch' '/switch' '/undo' '/copy' to manage chats.\n")

    agent = Agent(client, cache_dir=PROJECT_ROOT / ".cache")

    # Load input history (up/down arrow recall)
    hist_path = PROJECT_ROOT / ".cache" / "history.txt"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        gnureadline.read_history_file(str(hist_path))
    except FileNotFoundError:
        pass
    gnureadline.set_history_length(1000)

    while True:
        try:
            user_input = input(f"{RED}User>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye.")
            break
        if not user_input:
            continue
        if user_input == "/exit":
            print("\nGoodbye.")
            break
        try:
            agent.chat(user_input)
        except AgentExit:
            print("\nGoodbye.")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

    # Save history on exit
    try:
        gnureadline.write_history_file(str(hist_path))
    except Exception:
        pass


def cmd_download(args):
    """Download a model."""
    model = args.model
    platform = _detect_platform()
    models = _load_models()

    if not model:
        print("Usage: python main.py download <model>")
        print(f"Available models: {', '.join(models.keys())}")
        sys.exit(1)

    if model not in models:
        sys.exit(f"Error: model '{model}' not found in local-models.yaml. Available: {', '.join(models.keys())}")

    entry = models[model]
    if platform not in entry or "repo" not in entry[platform]:
        sys.exit(f"Error: model '{model}' has no repo for platform '{platform}' in local-models.yaml")

    repo = entry[platform]["repo"]
    hf_cache = PROJECT_ROOT / ".cache" / "huggingface"

    print(f"Project root: {PROJECT_ROOT}")
    print(f"HF cache:     {hf_cache}")
    print(f"Downloading model '{model}' ({repo}) for {platform}...")
    snapshot_download(repo, cache_dir=str(hf_cache))
    print("Model downloaded to Hugging Face cache at", hf_cache)


def _load_models() -> dict:
    """Load and parse local-models.yaml."""
    models_yaml = PROJECT_ROOT / "local-models.yaml"
    if not models_yaml.exists():
        sys.exit(f"Error: {models_yaml} not found")
    try:
        return yaml.safe_load(models_yaml.read_text())
    except Exception:
        sys.exit(f"Error: failed to parse {models_yaml}")


def _detect_platform() -> str:
    system = platform.system()
    if system == "Darwin":
        return "macos"
    elif system == "Linux":
        return "linux"
    else:
        sys.exit(f"Unsupported platform: {system}")


def _resolve_model(alias: str) -> str:
    """Resolve model alias to local path using local-models.yaml."""
    models = _load_models()
    platform = _detect_platform()
    if alias not in models:
        sys.exit(f"Error: model '{alias}' not found in local-models.yaml. Available: {', '.join(models.keys())}")

    entry = models[alias]
    if platform not in entry:
        sys.exit(f"Model '{alias}' has no '{platform}' entry in local-models.yaml")

    plat_entry = entry[platform]
    if "repo" in plat_entry:
        repo = plat_entry["repo"]
        # Check HF cache
        hf_cache = PROJECT_ROOT / ".cache" / "huggingface"
        dirname = repo.replace("/", "--")
        snapshots_dir = hf_cache / f"models--{dirname}" / "snapshots"
        if snapshots_dir.is_dir():
            # Iterdir->getmtime is racy if a concurrent process deletes a
            # snapshot between the two calls.  Ignore vanished entries.
            def _mtime(p):
                try:
                    return os.path.getmtime(p)
                except FileNotFoundError:
                    return 0.0

            snapshots = sorted(snapshots_dir.iterdir(), key=_mtime)
            if snapshots:
                return str(snapshots[-1])
        sys.exit(f"Model '{repo}' not found in HF cache. Run: python main.py download {alias}")

    if "path" in plat_entry:
        return str(PROJECT_ROOT / plat_entry["path"])

    sys.exit(f"Model '{alias}/{platform}' has neither repo: nor path: in local-models.yaml")


def main():
    parser = argparse.ArgumentParser(description="agent-local CLI")
    subparsers = parser.add_subparsers(dest="command")

    # serve
    subparsers.add_parser("serve", help="Start API server")

    # chat
    subparsers.add_parser("chat", help="Start agent chat")

    # download
    parser_download = subparsers.add_parser("download", help="Download a model")
    parser_download.add_argument("model", nargs="?", help="Model alias to download")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "download":
        cmd_download(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
