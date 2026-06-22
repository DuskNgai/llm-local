"""Tests for the Agent class."""

import pytest

from src.agent.agent import Agent, AgentExit
from src.llm.client import LLMClient
from src.llm.types import LLMResponse, Message, ToolCall, ToolDefinition


class StubLLMClient(LLMClient):
    """Stub LLM client for testing the Agent without real API calls."""

    def __init__(self, responses=None, model="test"):
        super().__init__("http://test", "key", model)
        self.responses = responses if responses is not None else []
        self.call_count = 0

    def chat(self, messages, system=None, tools=None, stream=True, on_chunk=None):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return LLMResponse(
            content="default response", tool_calls=[], usage={}
        )

    def supports_tools(self):
        return True


@pytest.fixture
def stub_client():
    return StubLLMClient()


@pytest.fixture
def agent(stub_client):
    return Agent(client=stub_client)


class TestSlashCommands:

    def test_exit(self, agent):
        with pytest.raises(AgentExit):
            agent.chat("/exit")

    def test_clear(self, agent):
        result = agent.chat("/clear")
        assert result == "Context cleared."

    def test_save(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        client = StubLLMClient()
        agent_inst = Agent(client=client)
        agent_inst.chat("hello") # Add a message before saving
        result = agent_inst.chat("/save test-save")
        assert "test-save" in result
        assert (tmp_path / ".cache" / "chats" / "test-save.json").exists()

    def test_load(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Save a conversation first
        client1 = StubLLMClient()
        agent1 = Agent(client=client1)
        agent1.chat("hello")
        agent1.chat("/save test-load")

        # Load it into a fresh agent
        client2 = StubLLMClient()
        agent2 = Agent(client=client2)
        result = agent2.chat("/load test-load")
        assert "Loaded" in result

        # Verify the conversation was actually loaded
        messages = agent2.conversation.get_messages()
        assert len(messages) > 0
        assert any(m.content == "hello" for m in messages)

    def test_load_usage_hint(self, agent):
        result = agent.chat("/load")
        assert "Usage:" in result

    def test_load_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        client = StubLLMClient()
        agent_inst = Agent(client=client)
        result = agent_inst.chat("/load nonexistent")
        assert "not found" in result

    def test_list_empty(self, agent):
        result = agent.chat("/list")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_list_with_saved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        client = StubLLMClient()
        agent_inst = Agent(client=client)
        agent_inst.chat("hello")
        agent_inst.chat("/save test-list")
        result = agent_inst.chat("/list")
        assert "test-list" in result

    def test_branch(self, agent):
        result = agent.chat("/branch mybranch")
        assert "Created branch" in result
        assert "mybranch" in result

    def test_branch_usage_hint(self, agent):
        result = agent.chat("/branch")
        assert "Usage:" in result

    def test_switch(self, agent):
        agent.chat("/branch mybranch")
        result = agent.chat("/switch mybranch")
        assert "Switched to branch" in result
        assert "mybranch" in result

    def test_switch_usage_hint(self, agent):
        result = agent.chat("/switch")
        assert "Usage:" in result

    def test_switch_not_found(self, agent):
        result = agent.chat("/switch nonexistent")
        assert "not found" in result

    def test_undo(self, agent):
        agent.chat("hello") # Add messages to undo
        result = agent.chat("/undo")
        assert "Undid" in result

    def test_undo_with_count(self, agent):
        agent.chat("hello")
        agent.chat("world")
        result = agent.chat("/undo 2")
        assert "Undid 2" in result

    def test_copy(self, agent):
        agent.chat("hello") # Adds user + assistant messages
        result = agent.chat("/copy")

        # On macOS it copies to clipboard, otherwise returns the text
        assert result is not None
        assert len(result) > 0

    def test_copy_no_reply(self, agent):
        # No messages yet, should return "No reply to copy."
        result = agent.chat("/copy")
        assert "No reply" in result

    def test_unknown_command(self, agent):
        result = agent.chat("/unknown")
        assert "Unknown command" in result


class TestToolCalling:

    def test_normal_message(self, agent):
        result = agent.chat("hello")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_tool_calling_loop(self):
        stub = StubLLMClient(
            responses=[
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(
                        id="call_1",
                        name="shell",
                        arguments={
                            "command": "echo hello"
                        },
                    )],
                    usage={},
                ),
                LLMResponse(content="Got it!", tool_calls=[], usage={}),
            ]
        )
        agent_inst = Agent(client=stub)
        result = agent_inst.chat("run something")
        assert "Got it!" in result
