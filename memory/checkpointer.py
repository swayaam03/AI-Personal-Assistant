from langgraph.checkpoint.memory import MemorySaver

# In-memory saver singleton instance for state persistence across multi-turn thread conversations
_checkpointer_instance = None


def get_memory_checkpointer() -> MemorySaver:
    """
    Returns an instance of LangGraph MemorySaver checkpointer.
    
    The checkpointer saves state checkpoints indexed by thread_id.
    This enables seamless multi-turn conversation context, state recovery,
    and history inspection.
    """
    global _checkpointer_instance
    if _checkpointer_instance is None:
        _checkpointer_instance = MemorySaver()
    return _checkpointer_instance
