from pathlib import Path

from src.agent.workspace import Workspace


class TestWorkspace:
    """Tests for Workspace -- default/override cwd and path resolution."""

    def test_default_cwd(self):
        """Workspace() defaults to os.getcwd()."""
        ws = Workspace()
        assert ws.cwd == Path.cwd()

    def test_custom_cwd(self):
        """Workspace(cwd='/tmp') stores the given path."""
        ws = Workspace(cwd=Path("/tmp"))
        assert ws.cwd == Path("/tmp")

    def test_resolve_relative(self):
        """With cwd='/tmp', resolve('foo') -> /tmp/foo."""
        ws = Workspace(cwd=Path("/tmp"))
        result = ws.resolve("foo")
        expected = (Path("/tmp") / "foo").resolve()
        assert result == expected
        assert result.is_absolute()

    def test_resolve_absolute(self):
        """With cwd='/tmp', resolve('/etc/passwd') -> /etc/passwd (ignores cwd)."""
        ws = Workspace(cwd=Path("/tmp"))
        result = ws.resolve("/etc/passwd")
        assert result == Path("/etc/passwd")
