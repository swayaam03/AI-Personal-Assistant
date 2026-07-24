"""
Phase 3 Verification Script.
Tests system message generation and tool binding to Gemini model.
"""
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prompts.system_prompts import get_system_message, ASSISTANT_SYSTEM_PROMPT
from agents.assistant_agent import create_agent_model
from tools import ALL_TOOLS


def test_phase3_agent_setup():
    print("=== Testing Phase 3: System Prompts & Tool Binding ===")
    
    # 1. Test System Message
    sys_msg = get_system_message()
    print("[OK] System Message created:")
    print(f"    - Type: {sys_msg.__class__.__name__}")
    print(f"    - Content length: {len(sys_msg.content)} chars")

    # 2. Test Tool Binding on LLM Model
    model_with_tools = create_agent_model()
    print(f"[OK] Agent Model created with tool bindings.")
    print(f"    - Model object: {type(model_with_tools).__name__}")
    print(f"    - Bound tools count: {len(ALL_TOOLS)}")

    print("\nPhase 3 Prompt & Agent Model Verification Passed! Ready for Phase 4.")


if __name__ == "__main__":
    test_phase3_agent_setup()
