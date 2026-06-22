import json
import logging
from typing import Callable

from anthropic import Anthropic
from anthropic.types import (
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
import httpx

from .client import LLMClient
from .types import LLMResponse, Message, ToolCall, ToolDefinition


class AnthropicClient(LLMClient):

    def __init__(self, base_url: str, api_key: str, model: str):
        super().__init__(base_url, api_key, model)
        # Anthropic client expects api_key at construction; for local MLX proxy
        # it may be empty or a placeholder.
        http_client = httpx.Client(proxy=None, trust_env=False)
        self.client = Anthropic(base_url=base_url, api_key=api_key, http_client=http_client)

    def supports_tools(self) -> bool:
        return True

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        stream: bool = True,
        on_chunk: Callable[[str, str], None] | None = None,
    ) -> LLMResponse:
        anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None

        if stream:
            return self._chat_stream(anthropic_messages, system, anthropic_tools, on_chunk)
        else:
            return self._chat_non_stream(anthropic_messages, system, anthropic_tools)

    # ------------------------------------------------------------------
    # Message / tool conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[Message]) -> list[dict]:
        """Convert our Message list to Anthropic-format messages."""
        converted: list[dict] = []
        for m in messages:
            if m.role == "tool":
                # Tool result: role must be "user" in Anthropic protocol.
                # Merge consecutive tool results into a single user message
                # so roles alternate correctly (assistant -> user -> assistant).
                prev = converted[-1] if converted else None
                prev_content = prev.get("content") if prev else None
                merge = (
                    prev is not None and \
                    prev["role"] == "user" and \
                    isinstance(prev_content, list) and \
                    any(isinstance(b, dict) and b.get("type") == "tool_result" for b in prev_content)
                )
                if merge:
                    prev_content.append({
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id if m.tool_call_id is not None else "",
                        "content": m.content,
                    })
                else:
                    converted.append({
                        "role":
                        "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id if m.tool_call_id is not None else "",
                            "content": m.content,
                        }],
                    })
            elif m.role == "system":
                # System messages must be passed via the top-level `system`
                # parameter, not in the messages list. Log a warning so the
                # caller knows the message was dropped.
                logging.warning("System message found in messages list; use the 'system' parameter instead. Message dropped.")
            else:
                # builder the content representation
                if m.tool_calls:
                    content_blocks: list[dict] = [{
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"]
                    } for tc in m.tool_calls]
                    if m.content and m.content != "[tool calls]":
                        content_blocks.insert(0, {
                            "type": "text",
                            "text": m.content
                        })
                else:
                    content_blocks = m.content
                converted.append({
                    "role": m.role,
                    "content": content_blocks,
                })
        return converted

    @staticmethod
    def _convert_tools(tools: list[ToolDefinition]) -> list[dict]:
        return [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        } for t in tools]

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def _chat_non_stream(
        self,
        messages: list[dict],
        system: str | None,
        tools: list[dict] | None,
    ) -> LLMResponse:
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)

        content: str | None = None
        reasoning: str | None = None
        tool_calls: list[ToolCall] = []
        usage = response.usage.model_dump() if response.usage else {}

        for block in response.content:
            if isinstance(block, TextBlock) or getattr(block, 'type', None) == "text":
                content = (content if content is not None else "") + (block.text if block.text is not None else "")
            elif isinstance(block, ThinkingBlock) or getattr(block, 'type', None) == "thinking":
                reasoning = (reasoning if reasoning is not None else "") + (block.thinking if block.thinking is not None else "")
            elif isinstance(block, ToolUseBlock) or getattr(block, 'type', None) == "tool_use":
                tool_input = block.input if isinstance(block.input, dict) else {}
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=tool_input,
                ))

        return LLMResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _chat_stream(
        self,
        messages: list[dict],
        system: str | None,
        tools: list[dict] | None,
        on_chunk=None,
    ) -> LLMResponse:
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage: dict = {}

        # State for accumulating streaming tool-use blocks
        current_tool: dict | None = None # {id, name, input_json}

        with self.client.messages.stream(**kwargs) as stream_response:
            for event in stream_response:
                event_type = getattr(event, 'type', None)

                # --- high-level convenience events (TextEvent, ThinkingEvent) ---
                if event_type == "text":
                    content_parts.append(event.text)
                    if on_chunk:
                        on_chunk("content", event.text)
                    continue

                if event_type == "thinking":
                    reasoning_parts.append(event.thinking)
                    if on_chunk:
                        on_chunk("reasoning", event.thinking)
                    continue

                # --- raw content-block events ---
                if event_type == "content_block_start":
                    cb = event.content_block
                    cb_type = getattr(cb, 'type', None)
                    if cb_type == "tool_use":
                        current_tool = {
                            "id": cb.id,
                            "name": cb.name,
                            "input_json": "",
                        }
                    elif cb_type == "thinking":
                        # Thinking block may start with initial text
                        if getattr(cb, 'thinking', None):
                            reasoning_parts.append(cb.thinking)
                            if on_chunk:
                                on_chunk("reasoning", cb.thinking)
                    elif cb_type == "text":
                        # Text block may start with initial text
                        if getattr(cb, 'text', None):
                            content_parts.append(cb.text)
                            if on_chunk:
                                on_chunk("content", cb.text)

                elif event_type == "content_block_delta":
                    delta = event.delta
                    delta_type = getattr(delta, 'type', None)

                    if delta_type == "text_delta":
                        content_parts.append(delta.text)
                        if on_chunk:
                            on_chunk("content", delta.text)
                    elif delta_type == "thinking_delta":
                        reasoning_parts.append(delta.thinking)
                        if on_chunk:
                            on_chunk("reasoning", delta.thinking)
                    elif delta_type == "input_json_delta" and current_tool is not None:
                        current_tool["input_json"] += delta.partial_json

                elif event_type == "content_block_stop":
                    if current_tool is not None:
                        try:
                            arguments = json.loads(current_tool["input_json"]) if current_tool["input_json"].strip() else {}
                        except json.JSONDecodeError:
                            arguments = {}
                        tool_calls.append(ToolCall(
                            id=current_tool["id"],
                            name=current_tool["name"],
                            arguments=arguments,
                        ))
                        current_tool = None

                elif event_type == "message_start":
                    # Capture input tokens from the start event
                    msg = event.message
                    if msg and getattr(msg, 'usage', None):
                        usage["input_tokens"] = msg.usage.input_tokens

                elif event_type == "message_delta":
                    # Capture output tokens from the delta event
                    if getattr(event, 'usage', None):
                        output_usage = event.usage.model_dump()
                        usage.update(output_usage)

        content = "".join(content_parts) # keep "" as-is (distinct from None)
        reasoning = "".join(reasoning_parts).lstrip('\n')
        reasoning = reasoning if reasoning else None

        return LLMResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=usage,
        )
