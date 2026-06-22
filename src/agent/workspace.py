from pathlib import Path


class Workspace:
    """Represents the agent's working directory, used for path resolution."""

    def __init__(self, cwd: Path | None = None):
        self.cwd = Path(cwd) if cwd else Path.cwd()

    def resolve(self, path: str) -> Path:
        """Resolve a path relative to the workspace root.

        Absolute paths are returned as-is; relative paths are resolved against
        the workspace cwd.
        """
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.cwd / p).resolve()
