#!/usr/bin/env python3
"""Anthropic Messages API proxy to OpenAI-compatible backend.

Supports both streaming (SSE) and non-streaming modes.
"""

import json
import os
import time
import uuid
import urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

BACKEND_PORT = os.environ.get("OPENAI_PORT", "8000")
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}/v1/chat/completions"
LISTEN_PORT = int(os.environ.get("ANTHROPIC_PORT", "8001"))
MODEL_NAME = os.environ.get("MODEL", "qwen3-8b")

ANTHROPIC_VERSION = "2023-06-01"


class MessagesHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path != "/v1/messages":
            self.send_error(404, "Not Found")
            return
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))
            stream = body.get("stream", False)
            openai_request = _to_openai(body)
            openai_request["stream"] = stream
            if stream:
                self._handle_stream(openai_request, body)
            else:
                self._handle_non_stream(openai_request)
        except Exception:
            import traceback
            traceback.print_exc()
            self.send_error(500, "Internal Server Error")

    def _handle_non_stream(self, openai_request):
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
        self.send_header("x-api-key", "not-needed")
        self.send_header("anthropic-version", ANTHROPIC_VERSION)
        self.end_headers()
        self.wfile.write(json.dumps(anthropic_response).encode("utf-8"))

    def _handle_stream(self, openai_request_body, original_body):
        req = urllib.request.Request(
            BACKEND_URL,
            data=json.dumps(openai_request_body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            reader = resp.fp
            msg_id = f"msg_{uuid.uuid4().hex[:24]}"
            call_tokens = _flatten_tokens(original_body.get("messages", []), original_body.get("system"))
            max_tokens = openai_request_body.get("max_tokens", 4096)
            sent_block_starts = False

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("anthropic-version", ANTHROPIC_VERSION)
            self.end_headers()

            self._write_sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": MODEL_NAME,
                    "content": [],
                    "usage": {"input_tokens": call_tokens, "output_tokens": max_tokens},
                },
            })

            reply_text = ""
            for line in reader:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[len("data: "):])
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "") or ""

                if content:
                    if not sent_block_starts:
                        self._write_sse("content_block_start", {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        })
                        sent_block_starts = True
                    self._write_sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": content},
                    })
                    reply_text += content

            if sent_block_starts:
                self._write_sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": 0,
                })

            self._write_sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": len(reply_text.split())},
            })

            self._write_sse("message_stop", {
                "type": "message_stop",
            })

    def _write_sse(self, event, data):
        payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        self.wfile.write(payload.encode("utf-8"))
        self.wfile.flush()

    def log_message(self, format, *args):
        pass


def _flatten_content(content) -> str:
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content if b["type"] == "text")
    return content


def _flatten_tokens(messages, system) -> int:
    """Rough token count from character count (4 chars ~= 1 token)."""
    total = 0
    if system:
        total += len(_flatten_content(system))
    for m in messages or []:
        total += len(_flatten_content(m.get("content", "")))
    return total // 4


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
