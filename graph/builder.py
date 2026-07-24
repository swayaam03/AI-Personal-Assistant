import logging
from typing import Literal
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state.assistant_state import AssistantState
from agents.assistant_agent import invoke_with_model_cascade, SimulatedAgentModel
from prompts.system_prompts import get_system_message
from tools import ALL_TOOLS

logger = logging.getLogger(__name__)


# 1. Agent Node Function
def call_agent_node(state: AssistantState) -> dict:
    """
    Agent Node: Receives graph state, prepends system instructions,
    and invokes the Gemini LLM using the model cascade.
    
    Model Cascade Strategy:
        1. Try each model in GEMINI_MODELS list (e.g. gemini-2.0-flash, gemini-2.0-flash-lite, ...)
        2. If a model returns 429/404/403, skip it and try the next one.
        3. If ALL models fail, fall back to the local SimulatedAgentModel.
    """
    messages = list(state["messages"])
    
    # Prepend System Prompt if not already present in message history
    if not messages or not getattr(messages[0], "type", "") == "system":
        messages = [get_system_message()] + messages

    # Try the model cascade first
    response = invoke_with_model_cascade(messages)
    if response is not None:
        return {"messages": [response]}

    # All live models exhausted -> fall back to local agentic reasoning
    logger.debug("All models in cascade exhausted. Using local Agentic Simulator.")
    simulated_model = SimulatedAgentModel()
    response = simulated_model.invoke(messages)
    return {"messages": [response]}


# 2. Conditional Routing Edge Function
def should_continue(state: AssistantState) -> Literal["tools", "__end__"]:
    """
    Conditional Routing Edge: Inspects the agent's last response.
    
    - If the agent requested one or more tool calls (response.tool_calls is not empty),
      route state to the 'tools' Node.
    - If no tools were requested, route state to END (finish conversation turn).
    """
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# 3. StateGraph Construction Function
def build_assistant_graph(checkpointer=None):
    """
    Constructs and compiles the complete LangGraph Workflow.
    
    Graph Topology:
        [START] --> agent_node --> (should_continue?)
                                        |---> "tools": tool_node --> agent_node (Loop back)
                                        |---> END: Finish turn
    """
    graph_builder = StateGraph(AssistantState)

    # Add Nodes
    graph_builder.add_node("agent", call_agent_node)
    graph_builder.add_node("tools", ToolNode(ALL_TOOLS))

    # Set Entry Point
    graph_builder.set_entry_point("agent")

    # Add Conditional Edge from Agent Node
    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )

    # Add Edge from Tool Node back to Agent Node (Agentic Loop)
    graph_builder.add_edge("tools", "agent")

    # Compile with optional checkpointer memory
    compiled_graph = graph_builder.compile(checkpointer=checkpointer)
    return compiled_graph
