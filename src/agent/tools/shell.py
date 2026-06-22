from pathlib import Path
import shlex
import subprocess

from .base import Tool

FORBIDDEN_COMMANDS = ["rm", "mv", "delete", "rmdir", "shred"]


class ShellTool(Tool):
    """Shell tool -- the only built-in tool. Safety: read/write only, no delete."""

    name = "shell"
    description = "Execute a read/write shell command. Deletion commands (rm, mv, rmdir, shred) are forbidden."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            }
        },
        "required": ["command"],
    }

    def __init__(self, workspace_cwd: Path):
        self.cwd = workspace_cwd

    def execute(self, command: str) -> str:
        # Parse with shell lexer to split into command + args.
        # shell=False ensures metacharacters (;, |, &, $(), ``) are literal,
        # never interpreted by a shell.
        try:
            tokens = shlex.split(command.strip())
        except ValueError:
            tokens = command.strip().split()

        if not tokens:
            return "(no command)"

        for token in tokens:
            basename = token.split("/")[-1]
            if basename in FORBIDDEN_COMMANDS:
                return f"Error: Command '{basename}' is forbidden (delete operations not allowed)."

        try:
            result = subprocess.run(
                tokens,
                capture_output=True,
                text=True,
                cwd=str(self.cwd),
                timeout=30,
            )
            output = result.stdout
            if result.stderr:
                output += "\n[stderr]\n" + result.stderr
            # Truncate long output
            max_chars = 10000
            if len(output) > max_chars:
                output = (output[: max_chars] + f"\n... (output truncated at {max_chars} chars)")
            return output.strip() if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds."
        except Exception as e:
            return f"Error executing command: {e}"
