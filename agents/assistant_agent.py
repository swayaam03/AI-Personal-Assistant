import re
import uuid
import logging
from typing import Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings
from tools import ALL_TOOLS

logger = logging.getLogger(__name__)


class SimulatedAgentModel:
    """
    Fallback Agentic Reasoning Engine when live LLM quota is unavailable (Limit: 0).
    Demonstrates true LangGraph stateful loops, tool decisions, and memory.
    """

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        last_msg = messages[-1] if messages else HumanMessage(content="")

        # 1. If previous node was a Tool execution, synthesize tool observation
        if isinstance(last_msg, ToolMessage):
            tool_output = str(last_msg.content)
            return AIMessage(content=f"Based on the tool output:\n{tool_output}")

        user_text = str(last_msg.content).strip()
        lower_text = user_text.lower()

        # 2. Check if user input requires Calculator Tool
        if any(op in user_text for op in ["+", "*", "/", "^"]) or "calculate" in lower_text or "compute" in lower_text or "math" in lower_text:
            match = re.search(r'[\d\s\+\-\*\/\(\)\.]+', user_text)
            expr = match.group(0).strip() if match else "2 + 2"
            if len(expr) > 1:
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "calculate",
                        "args": {"expression": expr},
                        "id": f"call_{uuid.uuid4().hex[:8]}"
                    }]
                )

        # 3. Check if user input requires Weather Tool
        if "weather" in lower_text or "temperature" in lower_text or "forecast" in lower_text:
            words = user_text.replace("?", "").split()
            loc = "London"
            if "in" in words:
                idx = words.index("in")
                if idx + 1 < len(words):
                    loc = words[idx + 1]
            elif "for" in words:
                idx = words.index("for")
                if idx + 1 < len(words):
                    loc = words[idx + 1]

            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_weather",
                    "args": {"location": loc},
                    "id": f"call_{uuid.uuid4().hex[:8]}"
                }]
            )

        # 4. Check if user input requires Web Search Tool
        if "search" in lower_text or "news" in lower_text or "who is" in lower_text or "what is langgraph" in lower_text:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "web_search",
                    "args": {"query": user_text},
                    "id": f"call_{uuid.uuid4().hex[:8]}"
                }]
            )

        # 5. Check if user input requires Text Summarizer Tool
        if "summarize" in lower_text or "summary" in lower_text:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "summarize_text",
                    "args": {"text": user_text},
                    "id": f"call_{uuid.uuid4().hex[:8]}"
                }]
            )

        # 6. Extract Session Context (User Name & Custom Assistant Name like 'Jarvis')
        user_name = "friend"
        assistant_name = "Jarvis" if any("jarvis" in m.content.lower() for m in messages if isinstance(m, HumanMessage)) else "AI Personal Assistant"

        for m in messages:
            if isinstance(m, HumanMessage):
                text = m.content.lower()
                # Ensure "i am keeping/naming" doesn't trigger user name extraction
                if "i am " in text and not any(verb in text for verb in ["keeping", "naming", "calling", "giving"]):
                    candidate = text.split("i am ")[-1].split()[0].strip(".,!").title()
                    if candidate.lower() not in ["a", "the", "an", "here", "ready"]:
                        user_name = candidate
                elif "my name is " in text:
                    candidate = text.split("my name is ")[-1].split()[0].strip(".,!").title()
                    user_name = candidate

        # 7. Respond if user gave the assistant a custom name (e.g. Jarvis)
        if any(keyword in lower_text for keyword in ["name jarvis", "name is jarvis", "name you jarvis", "call you jarvis"]):
            greeting = f"Hello {user_name}!" if user_name != "friend" else "Hello!"
            return AIMessage(content=f"{greeting} Understood. From now on, call me Jarvis, your AI Personal Assistant powered by LangGraph. How can I assist you today?")

        if lower_text in ["who am i?", "what is my name?", "do you know my name?"]:
            if user_name != "friend":
                return AIMessage(content=f"Your name is {user_name}! I remember from our conversation thread.")
            return AIMessage(content="You haven't told me your name yet! What is your name?")

        if user_name != "friend":
            return AIMessage(content=f"Hello {user_name}! I am {assistant_name}, powered by LangGraph. How can I assist you with math, weather, web search, or text summarization today?")

        return AIMessage(content=f"Hello! I am {assistant_name}, powered by LangGraph. I can answer questions, remember conversation context, and execute tools (Calculator, Weather, Web Search, Summarizer). What would you like to do?")


def create_agent_model() -> Any:
    """
    Initializes the Gemini Chat LLM with tool bindings.
    """
    api_key = settings.GEMINI_API_KEY
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=api_key or "placeholder",
        temperature=0.7,
        max_retries=1,
    )
    return llm.bind_tools(ALL_TOOLS)
