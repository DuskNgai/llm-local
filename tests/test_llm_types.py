"""Tests for llm/types.py dataclasses."""

from src.llm.types import LLMResponse, Message, ToolCall, ToolDefinition


class TestMessage:

    def test_default_values(self):
        msg = Message(role="user", content="hi")
        assert msg.role == "user"
        assert msg.content == "hi"
        assert msg.tool_call_id is None
        assert msg.name is None

    def test_with_tool(self):
        msg = Message(role="tool", content="result", tool_call_id="abc", name="shell")
        assert msg.role == "tool"
        assert msg.content == "result"
        assert msg.tool_call_id == "abc"
        assert msg.name == "shell"


class TestToolDefinition:

    def test_create(self):
        td = ToolDefinition(
            name="shell",
            description="Execute a command",
            parameters={
                "type": "object",
                "properties": {}
            },
        )
        assert td.name == "shell"
        assert td.description == "Execute a command"
        assert td.parameters == {
            "type": "object",
            "properties": {}
        }


class TestToolCall:

    def test_create(self):
        tc = ToolCall(
            id="call_1", name="shell", arguments={
                "command": "ls"
            }
        )
        assert tc.id == "call_1"
        assert tc.name == "shell"
        assert tc.arguments == {
            "command": "ls"
        }


class TestLLMResponse:

    def test_defaults(self):
        resp = LLMResponse()
        assert resp.content is None
        assert resp.reasoning is None
        assert resp.tool_calls == []
        assert resp.usage == {}

    def test_with_reasoning(self):
        resp = LLMResponse(
            content="hi", reasoning="thinking...", tool_calls=[], usage={}
        )
        assert resp.content == "hi"
        assert resp.reasoning == "thinking..."
        assert resp.tool_calls == []
        assert resp.usage == {}
