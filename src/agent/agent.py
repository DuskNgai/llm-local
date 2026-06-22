from datetime import datetime
import json
from pathlib import Path
import subprocess

from src.llm.client import LLMClient

from .conversation import Conversation
from .term import BLUE, GREEN, RESET, YELLOW
from .tools.registry import ToolRegistry
from .tools.shell import ShellTool
from .workspace import Workspace


class AgentExit(Exception):
    """Raised to signal the agent loop should exit gracefully."""
    pass


class Agent:
    """Agent core -- tool-calling loop + slash commands."""

    def __init__(
        self,
        client: LLMClient,
        workspace: Workspace | None = None,
        system_prompt: str = "You are a helpful assistant.",
        cache_dir: Path | None = None,
    ):
        self.client = client
        self.workspace = workspace if workspace is not None else Workspace()
        self.system_prompt = system_prompt
        self.conversation = Conversation(system_prompt=system_prompt)
        self.cache_dir = cache_dir if cache_dir is not None else Path(".cache")
        self.tools = ToolRegistry()
        self.tools.register(ShellTool(self.workspace.cwd))

    def chat(self, user_input: str) -> str:
        """Process user input through the agent loop."""
        if user_input.startswith("/"):
            result = self._handle_slash_command(user_input)
            print(f"\n{GREEN}Assistant>{RESET} {result}\n")
            return result

        self.conversation.add_user_message(user_input)

        max_iterations = 25
        for _ in range(max_iterations):
            messages = self.conversation.get_messages()
            tools = self.tools.get_tool_definitions() if self.client.supports_tools() else None

            # Real-time streaming display callback
            stream_state = {
                "reasoning": False,
                "content": False,
                "last_char": ""
            }

            def on_chunk(kind: str, text: str):
                if kind == "reasoning":
                    if not stream_state["reasoning"]:
                        text = text.lstrip('\n')
                        if not text:
                            return
                        print(f"\n{YELLOW}Reasoning>{RESET} ", end="", flush=True)
                        stream_state["reasoning"] = True
                    print(text, end="", flush=True)
                    if text:
                        stream_state["last_char"] = text[-1]
                elif kind == "content":
                    if not stream_state["content"]:
                        text = text.lstrip('\n')
                        if not text:
                            return
                        print(f"\n{GREEN}Assistant>{RESET} ", end="", flush=True)
                        stream_state["content"] = True
                    print(text, end="", flush=True)
                    if text:
                        stream_state["last_char"] = text[-1]

            try:
                response = self.client.chat(messages, system=self.system_prompt, tools=tools, on_chunk=on_chunk)
            except Exception as e:
                error_msg = f"Error: LLM call failed: {e}"
                print(f"\n{GREEN}Assistant>{RESET} {error_msg}")
                return error_msg

            if stream_state["reasoning"] or stream_state["content"]:
                if stream_state["last_char"] != "\n":
                    print()
                print()

            # Text content only, no tool calls -- final answer
            if response.content and not response.tool_calls:
                self.conversation.add_assistant_message(response.content)
                return response.content

            # Tool calls present -- execute them
            if response.tool_calls:
                tc_dicts = [{
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments
                } for tc in response.tool_calls]
                self.conversation.add_assistant_message(
                    response.content if response.content else "[tool calls]",
                    tool_calls=tc_dicts,
                )

                for tc in response.tool_calls:
                    try:
                        result = self.tools.execute(tc.name, tc.arguments)
                    except Exception as e:
                        result = f"Error: tool execution failed: {e}"
                    print(f"{BLUE}Tool call ({tc.name})> {tc.arguments.get('command', '')}{RESET}\n{result}")
                    self.conversation.add_tool_message(result, tc.id)

                continue

            # No content and no tool calls -- still record the turn
            self.conversation.add_assistant_message(response.content if response.content is not None else "")
            empty_msg = "(empty response)"
            print(f"\n{GREEN}Assistant>{RESET} {empty_msg}")
            return empty_msg

        error_msg = "Error: Exceeded maximum tool-calling iterations (25)."
        print(f"\n{GREEN}Assistant>{RESET} {error_msg}")
        return error_msg

    def _handle_slash_command(self, user_input: str) -> str:
        """Handle built-in slash commands."""
        parts = user_input.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/exit":
            raise AgentExit()

        elif cmd == "/clear":
            self.conversation.clear()
            return "Context cleared."

        elif cmd == "/save":
            name = arg if arg else datetime.now().strftime("%Y%m%d-%H%M%S")
            # Sanitize: extract basename only to prevent path traversal
            name = Path(name).name
            if not name:
                name = datetime.now().strftime("%Y%m%d-%H%M%S")
            chats_dir = self.cache_dir / "chats"
            chats_dir.mkdir(parents=True, exist_ok=True)
            path = chats_dir / f"{name}.json"
            path.write_text(json.dumps(self.conversation.to_dict(), ensure_ascii=False, indent=2))
            return f"Conversation saved to .cache/chats/{name}.json"

        elif cmd == "/load":
            if not arg:
                return "Usage: /load <name>"
            # Sanitize: extract basename only to prevent path traversal
            name = Path(arg).name
            path = self.cache_dir / "chats" / f"{name}.json"
            if not path.exists():
                return f"Chat '{name}' not found."
            data = json.loads(path.read_text())
            self.conversation = Conversation.from_dict(data)
            self.system_prompt = self.conversation.root.content
            return f"Loaded conversation from .cache/chats/{name}.json"

        elif cmd == "/list":
            chats_dir = self.cache_dir / "chats"
            if not chats_dir.exists():
                return "No saved chats."
            files = sorted(chats_dir.glob("*.json"))
            if not files:
                return "No saved chats."
            return "Saved chats:\n" + "\n".join(f"  {f.stem}" for f in files)

        elif cmd == "/branch":
            if not arg:
                return "Usage: /branch <name>"
            self.conversation.branch(arg)
            return f"Created branch '{arg}' at current node."

        elif cmd == "/switch":
            if not arg:
                return "Usage: /switch <name>"
            try:
                self.conversation.switch(arg)
                return f"Switched to branch '{arg}'."
            except ValueError as e:
                return str(e)

        elif cmd == "/undo":
            try:
                n = int(arg) if arg else 1
            except ValueError:
                return "Usage: /undo [n]"
            self.conversation.undo(n)
            return f"Undid {n} step(s)."

        elif cmd == "/copy":
            text = self.conversation.copy_last_reply()
            if not text:
                return "No reply to copy."
            try:
                subprocess.run(["pbcopy"], input=text, text=True)
                return "Copied last reply to clipboard."
            except Exception:
                return text

        else:
            return f"Unknown command: {cmd}"
