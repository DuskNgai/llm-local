import json
from typing import Callable

import httpx
from openai import OpenAI

from .client import LLMClient
from .types import LLMResponse, Message, ToolCall, ToolDefinition


class OpenAIClient(LLMClient):

    def __init__(self, base_url: str, api_key: str, model: str):
        super().__init__(base_url, api_key, model)
        http_client = httpx.Client(proxy=None, trust_env=False)
        self.client = OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

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
        oai_messages = self._convert_messages(messages, system)
        oai_tools = self._convert_tools(tools) if tools else None

        if stream:
            return self._chat_stream(oai_messages, oai_tools, on_chunk)
        else:
            return self._chat_non_stream(oai_messages, oai_tools)

    # ------------------------------------------------------------------
    # Message / tool conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[Message], system: str | None) -> list[dict]:
        converted: list[dict] = []
        if system:
            converted.append({
                "role": "system",
                "content": system
            })
        for m in messages:
            # Filter the sentinel placeholder used for tool-only responses
            content = m.content
            if m.tool_calls and content == "[tool calls]":
                content = ""
            entry: dict = {
                "role": m.role,
                "content": content
            }
            if m.role == "tool" and m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            if m.name:
                entry["name"] = m.name
            if m.tool_calls:
                entry["tool_calls"] = [{
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"])
                    },
                } for tc in m.tool_calls]
            converted.append(entry)
        return converted

    @staticmethod
    def _convert_tools(tools: list[ToolDefinition]) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        } for t in tools]

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def _chat_non_stream(self, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            stream=False,
        )
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        content = msg.content # keep "" as-is (distinct from None)
        reasoning = getattr(msg, 'reasoning', None)
        if reasoning is not None:
            reasoning = reasoning.lstrip('\n')
            reasoning = reasoning if reasoning else None
        tool_calls = self._parse_tool_calls_non_stream(msg)
        usage = response.usage.model_dump() if response.usage else {}

        return LLMResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=usage,
        )

    @staticmethod
    def _parse_tool_calls_non_stream(msg) -> list[ToolCall]:
        raw = getattr(msg, 'tool_calls', None)
        if raw is None:
            raw = []
        result: list[ToolCall] = []
        for tc in raw:
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                arguments = {}
            result.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=arguments,
            ))
        return result

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _chat_stream(self, messages: list[dict], tools: list[dict] | None, on_chunk=None) -> LLMResponse:
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            stream=True,
        )
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        tool_call_buffers: dict[int, dict] = {} # index -> {id, name, args_json}
        usage: dict = {}

        for chunk in response:
            delta = chunk.choices[0].delta

            # Reasoning / thinking content
            r = getattr(delta, 'reasoning', None)
            if r:
                reasoning_parts.append(r)
                if on_chunk:
                    on_chunk("reasoning", r)

            # Text content
            c = getattr(delta, 'content', None)
            if c:
                content_parts.append(c)
                if on_chunk:
                    on_chunk("content", c)

            # Tool calls (streaming chunks)
            delta_tool_calls = getattr(delta, 'tool_calls', None)
            if delta_tool_calls is None:
                delta_tool_calls = []
            for tc in delta_tool_calls:
                idx = tc.index
                if idx not in tool_call_buffers:
                    tool_call_buffers[idx] = {
                        "id": tc.id if tc.id is not None else "",
                        "name": tc.function.name if tc.function else "",
                        "args_json": "",
                    }
                buf = tool_call_buffers[idx]
                if tc.id:
                    buf["id"] = tc.id
                if tc.function and tc.function.name:
                    buf["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    buf["args_json"] += tc.function.arguments

            # Usage may arrive in the final chunk
            if chunk.usage:
                usage = chunk.usage.model_dump()

        reasoning = "".join(reasoning_parts).lstrip('\n')
        reasoning = reasoning if reasoning else None
        content = "".join(content_parts) # keep "" as-is (distinct from None)

        # Parse accumulated tool call JSON (ordered by dict insertion = chunk index order)
        tool_calls: list[ToolCall] = []
        for buf in tool_call_buffers.values():
            try:
                arguments = json.loads(buf["args_json"]) if buf["args_json"].strip() else {}
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(ToolCall(
                id=buf["id"],
                name=buf["name"],
                arguments=arguments,
            ))

        return LLMResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=usage,
        )
