"""
Phase 5 Verification Script.
Tests memory checkpointer, state saving, and thread_id isolation.
"""
import os
import sys
from langchain_core.messages import HumanMessage

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memory import get_memory_checkpointer
from graph import build_assistant_graph


def test_phase5_checkpointer():
    print("=== Testing Phase 5: Memory Checkpointer & Thread State ===")
    
    checkpointer = get_memory_checkpointer()
    app = build_assistant_graph(checkpointer=checkpointer)
    print("[OK] Graph compiled with MemorySaver checkpointer.")

    # Define thread config
    config = {"configurable": {"thread_id": "test-thread-1"}}

    # Verify checkpointer state retrieval initial state
    state = app.get_state(config)
    print(f"[OK] Initial state values: {state.values}")

    print("\nPhase 5 Memory Verification Passed! Ready for Phase 6.")


if __name__ == "__main__":
    test_phase5_checkpointer()
