import uuid

from src.llm.types import Message


class ConversationNode:
    """A single node in a conversation tree."""

    def __init__(
        self,
        id: str,
        role: str,
        content: str,
        parent: "ConversationNode | None" = None,
        branch_name: str | None = None,
        tool_call_id: str | None = None,
        tool_calls: list[dict] | None = None,
    ):
        self.id = id
        self.role = role
        self.content = content
        self.parent = parent
        self.children: list[ConversationNode] = []
        self.branch_name = branch_name
        self.tool_call_id = tool_call_id
        self.tool_calls = tool_calls


class Conversation:
    """Conversation tree data structure -- supports branching, undo, copy."""

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.root = ConversationNode(id=str(uuid.uuid4()), role="system", content=system_prompt)
        # nodes dict: id -> ConversationNode
        self.nodes: dict[str, ConversationNode] = {
            self.root.id: self.root
        }
        self.current: ConversationNode = self.root
        self.branches: dict[str, list[str]] = {} # branch_name -> ordered list of node ids
        self._branch: str = "main"

    def _add_node(self, role: str, content: str, tool_call_id: str | None = None, tool_calls: list[dict] | None = None) -> ConversationNode:
        """Add a new node as a child of the current node and advance current."""
        node_id = str(uuid.uuid4())
        node = ConversationNode(
            id=node_id,
            role=role,
            content=content,
            parent=self.current,
            branch_name=self._branch,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
        )
        self.current.children.append(node)
        self.nodes[node_id] = node
        self.current = node
        if self._branch in self.branches:
            self.branches[self._branch].append(node_id)
        else:
            self.branches[self._branch] = [node_id]
        return node

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self._add_node("user", content)

    def add_assistant_message(self, content: str, tool_calls: list[dict] | None = None):
        """Add an assistant message to the conversation."""
        self._add_node("assistant", content, tool_calls=tool_calls)

    def add_tool_message(self, content: str, tool_call_id: str):
        """Add a tool result message to the conversation."""
        self._add_node("tool", content, tool_call_id=tool_call_id)

    def undo(self, n: int = 1):
        """Undo the last n steps on the current branch."""
        for _ in range(n):
            if self.current.parent is None:
                break # reached root, nothing more to undo

            parent = self.current.parent

            # Remove undone node from parent's children list so it does
            # not accumulate across repeated undo-then-add cycles.
            if self.current in parent.children:
                parent.children.remove(self.current)

            # Remove from branch path
            branch = self.branches.get(self._branch, [])
            if branch and branch[-1] == self.current.id:
                branch.pop()

            self.current = parent

    def branch(self, name: str):
        """Mark the current node as the start of a new named branch."""
        self.current.branch_name = name
        self._branch = name
        # Initialize branch path from root to current
        path: list[str] = []
        node: ConversationNode | None = self.current
        while node:
            path.insert(0, node.id)
            node = node.parent
        self.branches[name] = path

    def switch(self, name: str):
        """Switch to an existing branch (resume from its tip)."""
        if name not in self.branches:
            raise ValueError(f"Branch '{name}' not found")
        self._branch = name
        # Set current to the tip of the branch
        tip_id = self.branches[name][-1]
        self.current = self.nodes[tip_id]

    def copy_last_reply(self) -> str:
        """Return the content of the last assistant reply."""
        node = self.current
        while node and node.role != "assistant":
            node = node.parent
        return node.content if node else ""

    def get_messages(self) -> list[Message]:
        """Get all messages on the current path from root to current."""
        path: list[ConversationNode] = []
        node = self.current
        while node and node.role != "system":
            path.append(node)
            node = node.parent
        return [Message(role=n.role, content=n.content, tool_call_id=n.tool_call_id, tool_calls=n.tool_calls) for n in reversed(path)]

    def clear(self):
        """Reset the conversation, keeping the system prompt."""
        self.__init__(self.root.content)

    def to_dict(self) -> dict:
        """Serialize the conversation to a JSON-serializable dict."""
        return {
            "root": {
                "id": self.root.id,
                "role": self.root.role,
                "content": self.root.content,
            },
            "branches": self.branches,
            "nodes": {
                nid: {
                    "role": n.role,
                    "content": n.content,
                    "parent": n.parent.id if n.parent else None,
                    "branch_name": n.branch_name,
                    "tool_call_id": n.tool_call_id,
                    "tool_calls": n.tool_calls,
                }
                for nid, n in self.nodes.items()
            },
            "current": self.current.id,
            "branch": self._branch,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        """Deserialize a conversation from a dict produced by to_dict()."""
        conv = cls(data["root"]["content"])
        # Override root id
        conv.root.id = data["root"]["id"]
        conv.nodes = {
            conv.root.id: conv.root
        }
        # Recreate all nodes
        for nid, ndata in data["nodes"].items():
            if nid == conv.root.id:
                continue
            node = ConversationNode(
                id=nid,
                role=ndata["role"],
                content=ndata["content"],
                branch_name=ndata.get("branch_name"),
                tool_call_id=ndata.get("tool_call_id"),
                tool_calls=ndata.get("tool_calls"),
            )
            conv.nodes[nid] = node
        # Reconnect parents and children
        for nid, ndata in data["nodes"].items():
            if ndata.get("parent"):
                parent = conv.nodes[ndata["parent"]]
                child = conv.nodes[nid]
                child.parent = parent
                parent.children.append(child)
        # Restore branches
        conv.branches = data.get("branches", {})
        # Restore current
        conv.current = conv.nodes[data["current"]]
        conv._branch = data.get("branch", "main")
        return conv
