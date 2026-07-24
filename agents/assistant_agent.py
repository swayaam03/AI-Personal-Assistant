import re
import uuid
import logging
import concurrent.futures
from typing import Any, List, Optional, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from config import settings
from tools import ALL_TOOLS

logger = logging.getLogger(__name__)


class SimulatedAgentModel:
    """
    Rich Agentic Reasoning Engine when live LLM quota is unavailable (Limit: 0).
    Provides intelligent intent matching, fuzzy typo handling, and rich responses.
    """

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        last_msg = messages[-1] if messages else HumanMessage(content="")

        # 1. Handle Tool Execution Results
        if isinstance(last_msg, ToolMessage):
            tool_output = str(last_msg.content)
            return AIMessage(content=f"Here is the result from the tool execution:\n\n{tool_output}")

        user_text = str(last_msg.content).strip()
        lower_text = user_text.lower()

        # Extract Session Context (User Name & Assistant Name)
        user_name = "friend"
        assistant_name = "Jarvis" if any("jarvis" in m.content.lower() for m in messages if isinstance(m, HumanMessage)) else "AI Personal Assistant"

        for m in messages:
            if isinstance(m, HumanMessage):
                text = m.content.lower()
                if "i am " in text and not any(verb in text for verb in ["keeping", "naming", "calling", "giving"]):
                    candidate = text.split("i am ")[-1].split()[0].strip(".,!").title()
                    if candidate.lower() not in ["a", "the", "an", "here", "ready"]:
                        user_name = candidate
                elif "my name is " in text:
                    candidate = text.split("my name is ")[-1].split()[0].strip(".,!").title()
                    user_name = candidate

        # 2. Check for Math / Calculation Intent
        if any(op in user_text for op in ["+", "*", "/", "^"]) or any(w in lower_text for w in ["calculate", "compute", "math", "add", "multiply", "divide"]):
            match = re.search(r'[\d\s\+\-\*\/\(\)\.]+', user_text)
            expr = match.group(0).strip() if match else "2 + 2"
            if len(expr) > 1 and any(c.isdigit() for c in expr):
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "calculate",
                        "args": {"expression": expr},
                        "id": f"call_{uuid.uuid4().hex[:8]}"
                    }]
                )

        # 3. Check for Weather Intent (Handles typos like 'weater', 'wether', 'temperature')
        if any(w in lower_text for w in ["weather", "weater", "wether", "temperature", "forecast", "climate"]):
            words = user_text.replace("?", "").split()
            loc = "Mumbai"
            if "in" in words:
                idx = words.index("in")
                if idx + 1 < len(words):
                    loc = words[idx + 1]
            elif "for" in words:
                idx = words.index("for")
                if idx + 1 < len(words):
                    loc = words[idx + 1]
            elif "london" in lower_text:
                loc = "London"
            elif "tokyo" in lower_text:
                loc = "Tokyo"
            elif "paris" in lower_text:
                loc = "Paris"
            elif "delhi" in lower_text:
                loc = "Delhi"

            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_weather",
                    "args": {"location": loc.title()},
                    "id": f"call_{uuid.uuid4().hex[:8]}"
                }]
            )

        # 4. Check for Web Search Intent
        if any(w in lower_text for w in ["search", "news", "who is", "what is langgraph", "latest"]):
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "web_search",
                    "args": {"query": user_text},
                    "id": f"call_{uuid.uuid4().hex[:8]}"
                }]
            )

        # 5. Check for Text Summarization Intent
        if any(w in lower_text for w in ["summarize", "summary", "summarisation"]):
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "summarize_text",
                    "args": {"text": user_text},
                    "id": f"call_{uuid.uuid4().hex[:8]}"
                }]
            )

        # 6. Capabilities & Assistant Info Query
        if any(w in lower_text for w in ["capabilities", "limitation", "what can you do", "help"]):
            greeting = f"Hello {user_name}!" if user_name != "friend" else "Hello!"
            return AIMessage(content=(
                f"{greeting} I am {assistant_name}, your stateful AI Assistant powered by LangGraph.\n\n"
                f"Core Capabilities:\n"
                f"1. Calculator: Compute mathematical expressions (e.g., 'What is 15 * 6 + 10?')\n"
                f"2. Weather Report: Get temperature & weather forecasts (e.g., 'Weather in Mumbai')\n"
                f"3. Web Search: Perform live DuckDuckGo web searches for news and facts.\n"
                f"4. Text Summarizer: Summarize long articles into key insights.\n"
                f"5. Stateful Memory: Remember your name and multi-turn thread context!"
            ))

        # 7. Assistant Naming Command
        if any(w in lower_text for w in ["name jarvis", "name is jarvis", "call you jarvis", "keep your name"]):
            greeting = f"Hello {user_name}!" if user_name != "friend" else "Hello!"
            return AIMessage(content=f"{greeting} Understood! Call me Jarvis. I am your AI Personal Assistant powered by LangGraph. How can I assist you today?")

        # 8. User Identity Query
        if lower_text in ["who am i?", "what is my name?", "do you know my name?"]:
            if user_name != "friend":
                return AIMessage(content=f"Your name is {user_name}! I remember from our conversation thread.")
            return AIMessage(content="You haven't told me your name yet! What is your name?")

        # 9. General Conversational Response
        greeting = f"Hello {user_name}!" if user_name != "friend" else "Hello!"
        return AIMessage(content=f"{greeting} I am {assistant_name}. How can I assist you with calculations, weather, web search, or text summarization today?")


# ─── Model Factory Functions ───

def create_gemini_model(model_name: str) -> Any:
    """Creates a Gemini model instance bound to tools."""
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.GEMINI_API_KEY or "placeholder",
        temperature=0.7,
        max_retries=0,
        timeout=15,
    )
    return llm.bind_tools(ALL_TOOLS)


def create_openrouter_model(model_name: str) -> Any:
    """
    Creates an OpenRouter model instance bound to tools.
    OpenRouter uses an OpenAI-compatible API at https://openrouter.ai/api/v1
    """
    llm = ChatOpenAI(
        model=model_name,
        api_key=settings.OPENROUTER_API_KEY or "placeholder",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_retries=0,
        timeout=15,
        default_headers={
            "HTTP-Referer": "https://github.com/swayaam03/AI-Personal-Assistant",
            "X-Title": "AI Personal Assistant",
        },
    )
    return llm.bind_tools(ALL_TOOLS)


def create_model_for(provider: str, model_name: str) -> Any:
    """Factory function that creates the right model based on provider."""
    if provider == "gemini":
        return create_gemini_model(model_name)
    elif provider == "openrouter":
        return create_openrouter_model(model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ─── Multi-Provider Model Cascade ───

def invoke_with_model_cascade(messages: List[BaseMessage]) -> Optional[AIMessage]:
    """
    Tries each model in the cascade list across all configured providers.
    If a model's quota is exhausted (429) or unavailable (404), moves to the next.
    Returns None if ALL models across ALL providers fail.
    
    Cascade order is determined by settings.get_model_cascade() which returns
    tuples of (provider, model_name).
    """
    model_cascade = settings.get_model_cascade()

    for provider, model_name in model_cascade:
        try:
            llm_with_tools = create_model_for(provider, model_name)
            # Use a thread-based timeout to prevent hanging on slow API calls
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(llm_with_tools.invoke, messages)
                response = future.result(timeout=20)  # 20s hard timeout per model
            logger.debug(f"[CASCADE] Success: {provider}/{model_name}")
            return response
        except concurrent.futures.TimeoutError:
            logger.debug(f"[CASCADE] Timeout: {provider}/{model_name} (20s). Trying next...")
            continue
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                logger.debug(f"[CASCADE] Quota exhausted: {provider}/{model_name}. Trying next...")
            elif "404" in error_str or "NOT_FOUND" in error_str:
                logger.debug(f"[CASCADE] Not found: {provider}/{model_name}. Trying next...")
            elif "403" in error_str or "PERMISSION_DENIED" in error_str:
                logger.debug(f"[CASCADE] Permission denied: {provider}/{model_name}. Trying next...")
            elif "401" in error_str or "UNAUTHENTICATED" in error_str:
                logger.debug(f"[CASCADE] Invalid API key: {provider}/{model_name}. Trying next...")
            else:
                logger.debug(f"[CASCADE] Error: {provider}/{model_name}: {error_str[:100]}. Trying next...")
            continue

    # All models in cascade exhausted
    return None


# Legacy single-model function (kept for backward compatibility)
def create_agent_model() -> Any:
    """Creates agent model for the primary GEMINI_MODEL setting."""
    return create_gemini_model(settings.GEMINI_MODEL)
