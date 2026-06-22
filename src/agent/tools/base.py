from abc import ABC, abstractmethod


class Tool(ABC):
    """Abstract base class for all agent tools."""

    name: str
    description: str
    parameters: dict # JSON Schema dict

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with the given keyword arguments.

        Returns a string representation of the result (or an error message).
        """
        ...
