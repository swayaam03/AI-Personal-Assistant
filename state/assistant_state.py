from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AssistantState(TypedDict):
    """
    Represents the complete state schema for our LangGraph AI Personal Assistant.
    
    Attributes:
        messages: Sequence of chat messages (HumanMessage, AIMessage, ToolMessage).
                  Annotated with `add_messages` reducer to automatically handle:
                  - Appending new messages to the message history list
                  - Replacing messages if matching IDs are passed (updates)
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
