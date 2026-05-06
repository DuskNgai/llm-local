#!/usr/bin/env python3
"""CLI chat client backed by MLX-LM."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful assistant."}
CHATS_DIR = Path(__file__).parent.parent / ".cache" / "chats"

# ANSI color codes
C = {"red": "\033[31m", "yellow": "\033[33m", "green": "\033[32m", "r": "\033[0m"}
YOU = f"{C['red']}User>{C['r']} "
THINK = f"{C['yellow']}Think>{C['r']} "
ASSISTANT = f"{C['green']}Assistant>{C['r']} "


def format_stats(usage, elapsed: float) -> str:
    if not usage:
        return ""
    tokens = usage.get("total_tokens") or (usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
    tps = tokens / elapsed if elapsed > 0 else 0
    parts = [f"{tokens} tokens"]
    if "prompt_tokens" in usage:
        parts.append(f"({usage['prompt_tokens']} in / {usage['completion_tokens']} out)")
    else:
        parts.append(f"({usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out)")
    parts.append(f"{tps:.1f} tokens/s")
    return "  ".join(parts)


def _stream_stats(t0: float, chars: str, ttft: float | None) -> str:
    elapsed = time.time() - t0
    cps = len(chars) / elapsed if elapsed > 0 else 0
    stats = f"streamed {len(chars)} chars in {elapsed:.1f}s ({cps:.0f} chars/s)"
    if ttft:
        stats += f", TTFT: {ttft - t0:.2f}s"
    return stats


def chat_openai(client, model: str, user_input: str, stream: bool, messages: list[dict]) -> tuple[str, list[dict]]:
    messages = messages + [{"role": "user", "content": user_input}]
    print()
    t0 = time.time()
    response = client.chat.completions.create(
        model=model, messages=messages,
        temperature=0.7, max_tokens=4096, stream=stream,
    )
    if stream:
        reasoning = ""
        reply = ""
        ttft = None
        showed_think = False
        showed_reply = False
        for chunk in response:
            delta = chunk.choices[0].delta
            r = getattr(delta, 'reasoning', '') or ""
            c = getattr(delta, 'content', '') or ""
            if r:
                if ttft is None:
                    ttft = time.time()
                if not showed_think:
                    r = r.lstrip('\n')
                    if not r:
                        continue
                    print(THINK, end="", flush=True)
                    showed_think = True
                print(r, end="", flush=True)
                reasoning += r
            if c:
                if not showed_reply:
                    c = c.lstrip('\n')
                    if not c:
                        continue
                    if showed_think:
                        print(f"\n{ASSISTANT}", end="", flush=True)
                    else:
                        print(ASSISTANT, end="", flush=True)
                    showed_reply = True
                print(c, end="", flush=True)
                reply += c
        print()
        combined = f"[思考]\n{reasoning}\n\n[回答]\n{reply}" if reasoning else reply
        stats = _stream_stats(t0, reasoning + reply, ttft)
    else:
        elapsed = time.time() - t0
        msg = response.choices[0].message
        reasoning = (getattr(msg, 'reasoning', '') or "").lstrip('\n')
        reply = (getattr(msg, 'content', '') or "").lstrip('\n')
        if reasoning:
            print(f"{THINK}{reasoning}\n")
        print(f"{ASSISTANT}{reply}")
        combined = f"[思考]\n{reasoning}\n\n[回答]\n{reply}" if reasoning else reply
        usage = response.usage.model_dump() if response.usage else {}
        stats = format_stats(usage, elapsed)
    print(f"\n[{stats}]")
    return combined, messages + [{"role": "assistant", "content": combined}]


def chat_anthropic(client, model: str, user_input: str, stream: bool, messages: list[dict], system_content: str) -> tuple[str, list[dict]]:
    messages = messages + [{"role": "user", "content": user_input}]
    msgs = [m for m in messages if m["role"] != "system"]
    print()
    t0 = time.time()
    if stream:
        reply = ""
        ttft = None
        with client.messages.stream(
            model=model, messages=msgs,
            system=system_content,
            temperature=0.7, max_tokens=4096,
        ) as stream_response:
            for event in stream_response:
                if event.type == "text":
                    if ttft is None:
                        ttft = time.time()
                        print(ASSISTANT, end="", flush=True)
                    print(event.text, end="", flush=True)
                    reply += event.text
        print()
        stats = _stream_stats(t0, reply, ttft)
    else:
        response = client.messages.create(
            model=model, messages=msgs,
            system=system_content,
            temperature=0.7, max_tokens=4096,
        )
        elapsed = time.time() - t0
        reply = "".join(b.text for b in response.content if b.type == "text")
        print(f"{ASSISTANT}{reply}")
        usage = response.usage.model_dump() if response.usage else {}
        stats = format_stats(usage, elapsed)
    print(f"\n[{stats}]")
    return reply, messages + [{"role": "assistant", "content": reply}]


def _list_chats() -> list[str]:
    if not CHATS_DIR.exists():
        return []
    return sorted(p.stem for p in CHATS_DIR.glob("*.json"))


def _save_chat(name: str, msgs: list[dict]):
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    data = {"saved_at": datetime.now().isoformat(), "messages": msgs}
    (CHATS_DIR / f"{name}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2)
    )
    print(f"Saved {len(msgs)} messages to .cache/chats/{name}.json")


def _load_chat(name: str) -> list[dict] | None:
    path = CHATS_DIR / f"{name}.json"
    if not path.exists():
        print(f"Chat '{name}' not found")
        return None
    data = json.loads(path.read_text())
    msgs = data["messages"]
    print(f"Loaded {len(msgs)} messages from .cache/chats/{name}.json")
    return msgs


def main():
    parser = argparse.ArgumentParser(description="MLX-LM Chat Client")
    parser.add_argument("--backend", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-stream", action="store_false", dest="stream", default=True)
    parser.add_argument("--model", default="default_model")
    args = parser.parse_args()

    messages = [SYSTEM_PROMPT]

    if args.backend == "openai":
        from openai import OpenAI
        port = args.port or 8000
        client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="not-needed")
        model = args.model
        chat_fn = lambda u, msgs: chat_openai(client, model, u, args.stream, msgs)
    else:
        from anthropic import Anthropic
        port = args.port or 8001
        client = Anthropic(base_url=f"http://127.0.0.1:{port}")
        model = args.model
        system = SYSTEM_PROMPT["content"]
        chat_fn = lambda u, msgs: chat_anthropic(client, model, u, args.stream, msgs, system)

    stream_label = "streaming" if args.stream else "batch"
    print(f"MLX-LM Chat [{args.backend}] in [{stream_label}] mode.")
    print("`/exit` to quit, `/clear` to reset, `/save` [`/load` `/list`] to manage chats.\n")

    while True:
        try:
            user_input = input(YOU).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye.")
            break
        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/clear":
            messages = [SYSTEM_PROMPT]
            print("Context cleared.")
            continue
        if user_input.startswith("/save"):
            name = user_input[6:].strip() or datetime.now().strftime("%Y%m%d-%H%M%S")
            _save_chat(name, messages)
            continue
        if user_input.startswith("/load"):
            name = user_input[6:].strip()
            if not name:
                print("Usage: /load <name>")
                continue
            loaded = _load_chat(name)
            if loaded is not None:
                messages = loaded
            continue
        if user_input == "/list":
            names = _list_chats()
            if names:
                print("Saved chats:")
                for n in names:
                    print(f"  {n}")
            else:
                print("No saved chats.")
            continue
        try:
            _, messages = chat_fn(user_input, messages)
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
