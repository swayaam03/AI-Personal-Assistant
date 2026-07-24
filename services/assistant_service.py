from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from graph import build_assistant_graph
from memory import get_memory_checkpointer


class AssistantService:
    """
    Service Layer encapsulating LangGraph workflow execution,
    thread management, and response formatting.
    """

    def __init__(self):
        self.checkpointer = get_memory_checkpointer()
        self.app = build_assistant_graph(checkpointer=self.checkpointer)

    def process_user_message(self, message: str, thread_id: str = "default_thread") -> Dict[str, Any]:
        """
        Executes a user message through the LangGraph agent workflow.
        
        Args:
            message: Raw text prompt from the user.
            thread_id: Unique identifier for maintaining conversation memory.
            
        Returns:
            Dictionary containing:
            - response: Final text answer from the assistant
            - tool_calls_executed: Summary of tools called during reasoning loop
            - thread_id: Active thread id
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # Stream or invoke graph state with new user input
        initial_input = {"messages": [HumanMessage(content=message)]}
        final_state = self.app.invoke(initial_input, config=config)

        messages = final_state.get("messages", [])
        
        # Extract last AIMessage content as response
        final_response = ""
        tool_calls_executed = []

        for msg in messages:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tool_calls_executed.append({
                        "name": tc.get("name"),
                        "args": tc.get("args")
                    })
            if isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content

        return {
            "response": final_response or "I have processed your request.",
            "tool_calls": tool_calls_executed,
            "thread_id": thread_id,
        }

    def get_conversation_history(self, thread_id: str = "default_thread") -> List[Dict[str, str]]:
        """
        Retrieves the full message history for a given thread_id.
        """
        config = {"configurable": {"thread_id": thread_id}}
        state = self.app.get_state(config)
        messages = state.values.get("messages", []) if state.values else []

        history = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else ("tool" if isinstance(msg, ToolMessage) else "assistant")
            history.append({
                "role": role,
                "content": str(msg.content)
            })
        return history


# Global singleton instance of AssistantService
assistant_service = AssistantService()
