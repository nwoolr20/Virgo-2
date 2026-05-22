from virgo2.conversation import ConversationMemory
from virgo2.lifecycle import FieldLifecycleManager
from virgo2.registry import FieldRegistry
from virgo2.vault import FieldVault


def test_conversation_memory(tmp_path):
    c = ConversationMemory(FieldLifecycleManager(FieldVault(tmp_path), FieldRegistry(tmp_path)))
    c.add_turn("user", "hello")
    c.add_turn("assistant", "hi")
    c.converse_memory_update("Remember that my name is Nicholas")
    ctx = c.context_for("what is my name")
    assert "Relevant memory" in ctx and "Nicholas" in ctx
