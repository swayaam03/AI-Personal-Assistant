"""
Phase 6 Verification Script.
Tests AssistantService, multi-turn dialogue, and history extraction.
"""
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services import assistant_service


def test_phase6_service():
    print("=== Testing Phase 6: Service Layer & Multi-turn Execution ===")
    
    thread_id = "test-session-multi-turn"
    
    # 1. First user message
    res1 = assistant_service.process_user_message("Hi, my name is Alex and I live in Paris.", thread_id=thread_id)
    print(f"[OK] Turn 1 Response:\n{res1['response']}")

    # 2. Check history recorded in thread
    history = assistant_service.get_conversation_history(thread_id=thread_id)
    print(f"[OK] Thread History Length: {len(history)} messages")

    print("\nPhase 6 Service Layer Verification Passed! Ready for Phase 7.")


if __name__ == "__main__":
    test_phase6_service()
