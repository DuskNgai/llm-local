#!/usr/bin/env python3
"""Anthropic Messages API endpoint backed by MLX-LM."""

import json
import os
import urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

BACKEND_PORT = os.environ.get("OPENAI_PORT", "8000")
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}/v1/chat/completions"
LISTEN_PORT = int(os.environ.get("ANTHROPIC_PORT", "8001"))
MODEL_NAME = os.environ.get("MLX_MODEL", "qwen3-8b")


class MessagesHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path != "/v1/messages":
            self.send_error(404, "Not Found")
            return
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))
            openai_request = _to_openai(body)
            req = urllib.request.Request(
                BACKEND_URL,
                data=json.dumps(openai_request).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                openai_response = json.loads(resp.read().decode("utf-8"))
            anthropic_response = _to_anthropic(openai_response)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(anthropic_response).encode("utf-8"))
        except Exception:
            import traceback
            traceback.print_exc()
            self.send_error(500, "Internal Server Error")

    def log_message(self, format, *args):
        pass


def _flatten_content(content) -> str:
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content if b["type"] == "text")
    return content


def _to_openai(req: dict) -> dict:
    system_msgs = []
    if req.get("system"):
        system_msgs.append({"role": "system", "content": _flatten_content(req["system"])})
    user_msgs = []
    for m in req.get("messages", []):
        user_msgs.append({"role": m["role"], "content": _flatten_content(m["content"])})
    return {
        "messages": system_msgs + user_msgs,
        "max_tokens": req.get("max_tokens", 4096),
        "temperature": req.get("temperature", 0.7),
    }


def _to_anthropic(resp: dict) -> dict:
    choice = resp["choices"][0]
    msg = choice["message"]
    # MLX-LM with thinking models puts reply in reasoning field
    text = msg.get("content") or msg.get("reasoning") or ""
    return {
        "id": resp.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": MODEL_NAME,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": resp.get("usage", {}),
    }


if __name__ == "__main__":
    print(f"Anthropic endpoint listening on http://127.0.0.1:{LISTEN_PORT}/v1/messages")
    ThreadingHTTPServer(("127.0.0.1", LISTEN_PORT), MessagesHandler).serve_forever()
