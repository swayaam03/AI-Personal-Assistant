"""
Phase 1 Verification Script.
Tests settings loading and AssistantState schema instantiation.
"""
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import HumanMessage, AIMessage
from config.settings import settings
from state.assistant_state import AssistantState


def test_phase1_setup():
    print("=== Testing Phase 1: Environment & State Setup ===")
    
    # 1. Test Config Settings
    print(f"[OK] Settings loaded successfully:")
    print(f"    - GEMINI_MODEL: {settings.GEMINI_MODEL}")
    print(f"    - HOST: {settings.HOST}:{settings.PORT}")
    print(f"    - API Key Configured: {'Yes' if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != 'your_gemini_api_key_here' else 'No (Placeholder active)'}")

    # 2. Test AssistantState Instantiation
    sample_state: AssistantState = {
        "messages": [
            HumanMessage(content="Hello AI Assistant!"),
            AIMessage(content="Hello! How can I assist you today?"),
        ]
    }
    
    print(f"[OK] AssistantState initialized with {len(sample_state['messages'])} messages:")
    for msg in sample_state["messages"]:
        print(f"    - {msg.__class__.__name__}: {msg.content}")

    print("\nPhase 1 Setup Verification Passed! Ready for Phase 2.")


if __name__ == "__main__":
    test_phase1_setup()
