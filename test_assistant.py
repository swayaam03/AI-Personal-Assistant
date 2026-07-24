"""
Interactive CLI Script for LangGraph AI Personal Assistant.
Allows multi-turn chat directly from the terminal.
"""
import sys
import uuid
import warnings

# Suppress non-critical library warnings in CLI output
warnings.filterwarnings("ignore")

from services import assistant_service


def main():
    print("=" * 60)
    print("🤖 LangGraph AI Personal Assistant CLI Interactive Shell")
    print("=" * 60)
    
    session_thread_id = f"cli-session-{str(uuid.uuid4())[:8]}"
    print(f"[Thread ID: {session_thread_id}]")
    print("Type your message below (or type 'exit' / 'quit' to stop).\n")

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
                print(f"🛠️ [Tools Executed]:")
                for tc in result["tool_calls"]:
                    print(f"   - {tc['name']} with args {tc['args']}")

            print(f"\nAssistant: {result['response']}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
