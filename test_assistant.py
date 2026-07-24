"""
Interactive CLI Script for LangGraph AI Personal Assistant.
Allows multi-turn chat directly from the terminal.
"""
import sys
import os
import uuid
import warnings

# Suppress non-critical library warnings in CLI output
warnings.filterwarnings("ignore")

# Fix Windows console encoding for special characters
if sys.platform == "win32":
    os.system("")  # Enable ANSI/VT100 escape sequences on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from services import assistant_service
from config import settings


def main():
    print("=" * 60)
    print("  LangGraph AI Personal Assistant CLI")
    print("=" * 60)
    
    # Display model cascade info grouped by provider
    cascade = settings.get_model_cascade()
    print(f"\n  Model Cascade ({len(cascade)} models):")
    current_provider = None
    for provider, model in cascade:
        if provider != current_provider:
            provider_label = "Google Gemini" if provider == "gemini" else "OpenRouter"
            print(f"    [{provider_label}]")
            current_provider = provider
        print(f"      -> {model}")
    print()
    
    session_thread_id = f"cli-session-{str(uuid.uuid4())[:8]}"
    print(f"  [Thread ID: {session_thread_id}]")
    print("  Type your message below (or type 'exit' / 'quit' to stop).\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nExiting AI Assistant CLI. Goodbye!")
                break

            print("\nThinking and reasoning with LangGraph...")
            result = assistant_service.process_user_message(
                message=user_input,
                thread_id=session_thread_id
            )

            # Display tool calls if invoked
            if result.get("tool_calls"):
                print("[Tools Executed]:")
                for tc in result["tool_calls"]:
                    print(f"   - {tc['name']} with args {tc['args']}")

            print(f"\nA: {result['response']}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
