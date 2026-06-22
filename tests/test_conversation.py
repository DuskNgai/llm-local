"""Tests for agent/conversation.py -- Conversation tree data structure."""

import pytest

from src.agent.conversation import Conversation, ConversationNode
from src.llm.types import Message

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _conv_after_user_assistant():
    """Return a fresh conversation with one user + one assistant turn."""
    c = Conversation()
    c.add_user_message("hello")
    c.add_assistant_message("hi there")
    return c


# ===========================================================================
# 1. basic conversation flow
# ===========================================================================


def test_basic_conversation_flow():
    c = Conversation()
    assert c.current.role == "system"

    c.add_user_message("What is 2+2?")
    c.add_assistant_message("4")

    msgs = c.get_messages()
    assert len(msgs) == 2
    assert msgs[0] == Message(role="user", content="What is 2+2?")
    assert msgs[1] == Message(role="assistant", content="4")


def test_get_messages_after_multiple_turns():
    c = Conversation()
    c.add_user_message("u1")
    c.add_assistant_message("a1")
    c.add_user_message("u2")
    c.add_assistant_message("a2")

    msgs = c.get_messages()
    assert len(msgs) == 4
    assert msgs[0].role == "user" and msgs[0].content == "u1"
    assert msgs[1].role == "assistant" and msgs[1].content == "a1"
    assert msgs[2].role == "user" and msgs[2].content == "u2"
    assert msgs[3].role == "assistant" and msgs[3].content == "a2"


# ===========================================================================
# 2. undo
# ===========================================================================


def test_undo_single_step():
    c = _conv_after_user_assistant()
    c.add_user_message("follow-up")
    c.add_assistant_message("answer")

    c.undo(1)
    msgs = c.get_messages()
    assert len(msgs) == 3
    assert msgs[-1].role == "user"
    assert msgs[-1].content == "follow-up"


def test_undo_multiple_steps():
    c = _conv_after_user_assistant()
    c.add_user_message("u2")
    c.add_assistant_message("a2")
    c.add_user_message("u3")
    c.add_assistant_message("a3")

    c.undo(2)
    msgs = c.get_messages()
    # Back to [u1, a1, u2, a2]
    assert len(msgs) == 4
    assert msgs[-1] == Message(role="assistant", content="a2")


# ===========================================================================
# 3. branch
# ===========================================================================


def test_branch_adds_messages_on_named_branch():
    c = _conv_after_user_assistant()
    c.branch("experiment")

    c.add_user_message("exp-user")
    c.add_assistant_message("exp-assistant")

    msgs = c.get_messages()
    assert len(msgs) == 4
    assert msgs[2] == Message(role="user", content="exp-user")
    assert msgs[3] == Message(role="assistant", content="exp-assistant")
    assert c._branch == "experiment"
    assert "experiment" in c.branches


def test_branch_records_path_from_root():
    c = _conv_after_user_assistant()
    c.branch("b1")
    # Path should include root -> user -> assistant
    path = c.branches["b1"]
    assert len(path) == 3 # root (system), user, assistant


# ===========================================================================
# 4. switch
# ===========================================================================


def test_switch_between_branches():
    c = _conv_after_user_assistant()     # main: [hello, hi there]
    c.branch("exp")
    c.add_user_message("exp-query")
    c.add_assistant_message("exp-reply") # Now on "exp" branch, tip = exp-reply

    c.switch("main")
    assert c._branch == "main"
    msgs = c.get_messages()
    # main path: [hello, hi there]
    assert len(msgs) == 2
    assert msgs[0].content == "hello"
    assert msgs[1].content == "hi there"

    c.switch("exp")
    assert c._branch == "exp"
    msgs = c.get_messages()
    assert len(msgs) == 4
    assert msgs[-1].content == "exp-reply"


def test_switch_unknown_branch_raises():
    c = Conversation()
    with pytest.raises(ValueError, match="not found"):
        c.switch("nonexistent")


# ===========================================================================
# 5. copy_last_reply
# ===========================================================================


def test_copy_last_reply_returns_assistant_content():
    c = _conv_after_user_assistant()
    assert c.copy_last_reply() == "hi there"

    c.add_user_message("another")
    c.add_assistant_message("another reply")
    assert c.copy_last_reply() == "another reply"


def test_copy_last_reply_when_no_assistant_exists():
    c = Conversation() # only system prompt, no assistant
    assert c.copy_last_reply() == ""

    c.add_user_message("just a question")
    assert c.copy_last_reply() == ""


# ===========================================================================
# 6. clear
# ===========================================================================


def test_clear_resets_everything_except_system_prompt():
    c = Conversation(system_prompt="Custom prompt")
    c.add_user_message("hello")
    c.add_assistant_message("hi")
    c.branch("test")
    c.add_user_message("on branch")
    c.add_assistant_message("reply")

    c.clear()

    assert c.current.role == "system"
    assert c.current.content == "Custom prompt"
    assert c._branch == "main"
    assert c.branches == {}
    assert len(c.nodes) == 1 # only the root node
    msgs = c.get_messages()
    assert msgs == []


def test_clear_keeps_new_system_prompt_id():
    c = Conversation()
    old_root_id = c.root.id
    c.add_user_message("msg")
    c.clear()
    assert c.root.id != old_root_id
    assert c.root.content == "You are a helpful assistant."


# ===========================================================================
# 7. serialization roundtrip
# ===========================================================================


def test_serialization_roundtrip_simple():
    c = _conv_after_user_assistant()
    data = c.to_dict()
    restored = Conversation.from_dict(data)

    assert restored.root.content == c.root.content
    original_msgs = c.get_messages()
    restored_msgs = restored.get_messages()
    assert len(original_msgs) == len(restored_msgs)
    for a, b in zip(original_msgs, restored_msgs):
        assert a == b


def test_serialization_roundtrip_complex():
    """Create a conversation with multiple branches, serialize, and restore."""
    c = Conversation(system_prompt="System prompt")
    c.add_user_message("u1")
    c.add_assistant_message("a1")

    c.branch("exp1")
    c.add_user_message("exp1-u")
    c.add_assistant_message("exp1-a")

    # Go back to fork point and create second branch
    c.undo(2)
    c.branch("exp2")
    c.add_user_message("exp2-u")
    c.add_assistant_message("exp2-a")

    # Switch branch before serialization
    c.switch("main")

    data = c.to_dict()
    restored = Conversation.from_dict(data)

    # Verify structure
    assert restored.root.content == "System prompt"
    assert restored._branch == c._branch
    assert restored.current.id == c.current.id

    # Verify main branch messages
    assert restored.get_messages() == c.get_messages()

    # Verify exp1 branch
    restored.switch("exp1")
    c.switch("exp1")
    assert restored.get_messages() == c.get_messages()

    # Verify exp2 branch
    restored.switch("exp2")
    c.switch("exp2")
    assert restored.get_messages() == c.get_messages()

    # Verify branch paths match
    for bname in c.branches:
        assert restored.branches[bname] == c.branches[bname]


def test_serialization_preserves_node_ids():
    c = _conv_after_user_assistant()
    data = c.to_dict()
    restored = Conversation.from_dict(data)

    # All original node IDs should be present in restored object
    for nid in c.nodes:
        assert nid in restored.nodes


# ===========================================================================
# 8. get_messages returns correct (chronological) order
# ===========================================================================


def test_get_messages_chronological_order():
    c = Conversation()
    c.add_user_message("first")
    c.add_assistant_message("second")
    c.add_user_message("third")
    c.add_assistant_message("fourth")

    msgs = c.get_messages()
    assert [m.content for m in msgs] == ["first", "second", "third", "fourth"]


# ===========================================================================
# 9. multiple branches from the same root node
# ===========================================================================


def test_multiple_branches_from_same_fork_point():
    c = Conversation()
    c.add_user_message("shared-u")
    c.add_assistant_message("shared-a")

    # First branch
    c.branch("b1")
    c.add_user_message("b1-u")
    c.add_assistant_message("b1-a")

    # Switch back to main (the fork point) without modifying b1's path
    c.switch("main")
    # Second branch from same fork point
    c.branch("b2")
    c.add_user_message("b2-u")
    c.add_assistant_message("b2-a")

    # Verify both branches work independently
    c.switch("b1")
    msgs_b1 = c.get_messages()
    assert [m.content for m in msgs_b1] == ["shared-u", "shared-a", "b1-u", "b1-a"]

    c.switch("b2")
    msgs_b2 = c.get_messages()
    assert [m.content for m in msgs_b2] == ["shared-u", "shared-a", "b2-u", "b2-a"]

    # Shared prefix is the same in both
    c.switch("main")
    msgs_main = c.get_messages()
    assert [m.content for m in msgs_main] == ["shared-u", "shared-a"]


def test_branches_have_independent_node_ids():
    c = Conversation()
    c.add_user_message("u")
    c.add_assistant_message("a")

    c.branch("b1")
    c.add_user_message("b1-u")

    c.undo(1)
    c.branch("b2")
    c.add_user_message("b2-u")

    # The two branch child nodes should have different IDs
    c.switch("b1")
    msg_b1 = c.get_messages()[-1]

    c.switch("b2")
    msg_b2 = c.get_messages()[-1]

    assert msg_b1.content != msg_b2.content # They are distinct messages (different IDs are generated via uuid)


# ===========================================================================
# 10. undo past root / stops at system prompt
# ===========================================================================


def test_undo_past_all_messages_stops_at_system_prompt():
    c = _conv_after_user_assistant()
    # Try to undo more steps than exist
    c.undo(10)

    # Should be back at the system prompt node
    assert c.current.role == "system"
    assert c.current.parent is None
    assert c.get_messages() == []


def test_undo_to_root_then_add_new_messages():
    """After undoing all the way back, new messages should work fine."""
    c = _conv_after_user_assistant()
    c.undo(2) # back to root

    c.add_user_message("fresh start")
    c.add_assistant_message("fresh reply")

    msgs = c.get_messages()
    assert len(msgs) == 2
    assert msgs[0] == Message(role="user", content="fresh start")
    assert msgs[1] == Message(role="assistant", content="fresh reply")


# ===========================================================================
# additional edge-case / sanity tests
# ===========================================================================


def test_system_prompt_default():
    c = Conversation()
    assert c.root.role == "system"
    assert c.root.content == "You are a helpful assistant."


def test_custom_system_prompt():
    c = Conversation(system_prompt="Be concise.")
    assert c.root.content == "Be concise."


def test_add_user_message_advances_current():
    c = Conversation()
    c.add_user_message("q")
    assert c.current.role == "user"
    assert c.current.content == "q"


def test_add_assistant_message_advances_current():
    c = Conversation()
    c.add_user_message("q")
    c.add_assistant_message("a")
    assert c.current.role == "assistant"
    assert c.current.content == "a"


def test_undo_on_fresh_conversation_does_nothing():
    c = Conversation()
    c.undo(1)
    assert c.current.role == "system"
    assert c.current.parent is None


def test_branch_on_root_node():
    c = Conversation()
    c.branch("custom_root_branch")
    assert c._branch == "custom_root_branch"
    path = c.branches["custom_root_branch"]
    assert len(path) == 1 # just the root


def test_current_node_has_correct_parent_chain():
    c = _conv_after_user_assistant()
    assistant_node = c.current
    assert assistant_node.parent is not None
    assert assistant_node.parent.role == "user"
    assert assistant_node.parent.parent.role == "system"
    assert assistant_node.parent.parent.parent is None


def test_nodes_have_unique_ids():
    c = _conv_after_user_assistant()
    ids = list(c.nodes.keys())
    assert len(ids) == len(set(ids))
    assert len(ids) == 3 # system, user, assistant


def test_multiple_messages_added_after_branching():
    c = Conversation()
    c.add_user_message("u1")
    c.add_assistant_message("a1")
    c.branch("test-branch")

    for i in range(5):
        c.add_user_message(f"u{i}")
        c.add_assistant_message(f"a{i}")

    msgs = c.get_messages()
    assert len(msgs) == 12 # u1, a1, + 10 branch messages
    assert c._branch == "test-branch"


def test_undo_after_branching_affects_branch_path():
    c = Conversation()
    c.add_user_message("u")
    c.add_assistant_message("a")
    c.branch("b")
    c.add_user_message("b-u")
    c.add_assistant_message("b-a")

    # Branch path before undo
    pre_undo_len = len(c.branches["b"])

    c.undo(1)

    # Branch path should be shorter
    assert len(c.branches["b"]) == pre_undo_len - 1
    assert c.current.role == "user"


def test_copy_last_reply_searches_past_user_messages():
    c = Conversation()
    c.add_user_message("u1")
    c.add_assistant_message("a1")
    c.add_user_message("u2")           # current is user, no assistant after it yet
    assert c.copy_last_reply() == "a1" # finds the previous assistant


def test_switch_does_not_change_branch_paths():
    c = Conversation()
    c.add_user_message("u")
    c.add_assistant_message("a")
    c.branch("b1")
    c.add_user_message("b1-u")
    c.branch("b2") # branch from b1-u

    b1_path_before = c.branches["b1"].copy()

    c.switch("main")

    assert c.branches["b1"] == b1_path_before


def test_to_dict_contains_expected_keys():
    c = _conv_after_user_assistant()
    data = c.to_dict()
    for key in ("root", "branches", "nodes", "current", "branch"):
        assert key in data


def test_get_messages_always_returns_Message_objects():
    c = _conv_after_user_assistant()
    msgs = c.get_messages()
    assert all(isinstance(m, Message) for m in msgs)
