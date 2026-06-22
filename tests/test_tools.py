from pathlib import Path
import subprocess
from unittest.mock import patch

from src.agent.tools.registry import ToolRegistry
from src.agent.tools.shell import ShellTool


class TestShellTool:
    """Tests for the ShellTool -- command execution and safety checks."""

    def test_basic_echo(self):
        """Execute echo hello, verify output is 'hello'."""
        tool = ShellTool(Path.cwd())
        result = tool.execute("echo hello")
        assert result == "hello"

    def test_stderr_capture(self):
        """Execute a command that writes to stderr, verify stderr is captured."""
        tool = ShellTool(Path.cwd())
        result = tool.execute("python3 -c \"import sys; sys.stderr.write('error')\"")
        assert "error" in result
        assert "[stderr]" in result

    def test_forbidden_rm(self):
        """Execute rm -rf /tmp/test, verify it's rejected."""
        tool = ShellTool(Path.cwd())
        result = tool.execute("rm -rf /tmp/test")
        assert "forbidden" in result.lower()

    def test_forbidden_mv(self):
        """Verify mv is rejected (even for rename)."""
        tool = ShellTool(Path.cwd())
        result = tool.execute("mv old new")
        assert "forbidden" in result.lower()

    def test_forbidden_rmdir(self):
        """Verify rmdir is rejected."""
        tool = ShellTool(Path.cwd())
        result = tool.execute("rmdir some_dir")
        assert "forbidden" in result.lower()

    def test_workspace_cwd(self, tmp_path):
        """Set workspace cwd, execute pwd, verify output contains the cwd."""
        tool = ShellTool(tmp_path)
        result = tool.execute("pwd")
        assert str(tmp_path) in result

    def test_timeout(self):
        """Execute sleep 60, verify it returns the timeout error message."""
        tool = ShellTool(Path.cwd())
        with patch("src.agent.tools.shell.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 60", timeout=30)
            result = tool.execute("sleep 60")
            assert "timed out" in result

    def test_empty_output(self):
        """Execute a command that produces no output, verify returns '(no output)'."""
        tool = ShellTool(Path.cwd())
        result = tool.execute("true")
        assert result == "(no output)"

    def test_path_with_forbidden_word_in_args(self):
        """echo 'removing' should NOT be rejected (only command basename, not args)."""
        tool = ShellTool(Path.cwd())
        result = tool.execute("echo removing")
        assert "forbidden" not in result.lower()
        assert "removing" in result


class TestToolRegistry:
    """Tests for ToolRegistry -- registration and execution dispatching."""

    def test_register_and_get_definitions(self):
        """Register tool, get definitions, verify name/description/parameters."""
        registry = ToolRegistry()
        tool = ShellTool(Path.cwd())
        registry.register(tool)

        definitions = registry.get_tool_definitions()
        assert len(definitions) == 1

        d = definitions[0]
        assert d.name == "shell"
        assert "shell command" in d.description.lower()
        assert d.parameters["type"] == "object"
        assert "command" in d.parameters["properties"]

    def test_execute_valid_tool(self):
        """Register shell tool, execute via registry, verify result."""
        registry = ToolRegistry()
        tool = ShellTool(Path.cwd())
        registry.register(tool)

        result = registry.execute("shell", {
            "command": "echo hello"
        })
        assert result == "hello"

    def test_execute_unknown_tool(self):
        """Execute unregistered tool name, verify error message."""
        registry = ToolRegistry()
        result = registry.execute("nonexistent", {})
        assert "Unknown tool" in result
