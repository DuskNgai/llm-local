from dataclasses import dataclass, field


@dataclass
class Message:
    role: str                            # "user" | "assistant" | "system" | "tool"
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None # assistant messages that issue tool calls


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict # JSON Schema


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None = None   # text reply (None when only tool_calls)
    reasoning: str | None = None # model thinking/reasoning content
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
