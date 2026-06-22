from src.llm.types import ToolDefinition

from .base import Tool


class ToolRegistry:
    """Registry for managing and executing agent tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Return tool definitions for the LLM."""
        return [ToolDefinition(
            name=t.name,
            description=t.description,
            parameters=t.parameters,
        ) for t in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> str:
        """Execute a registered tool by name with the given arguments.

        Returns the tool output or an error message string.
        """
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'"
        try:
            return tool.execute(**arguments)
        except Exception as e:
            return f"Error: Tool '{name}' failed: {e}"
