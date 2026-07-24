"""
Phase 2 Verification Script.
Tests all 4 custom agentic tools: Calculator, Weather, Search, Summarizer.
"""
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools import calculate, get_weather, web_search, summarize_text, ALL_TOOLS


def test_phase2_tools():
    print("=== Testing Phase 2: Custom Tools ===")
    
    print(f"[OK] Total tools registered: {len(ALL_TOOLS)}")
    for t in ALL_TOOLS:
        print(f"    - Tool: '{t.name}' -> {t.description.splitlines()[0]}")

    print("\n--- 1. Testing Calculator Tool ---")
    calc_res = calculate.invoke({"expression": "12 * 8 + 4"})
    print(f"Result: {calc_res}")
    assert "100" in calc_res, "Calculator verification failed!"

    print("\n--- 2. Testing Weather Tool ---")
    weather_res = get_weather.invoke({"location": "London"})
    print(f"Result:\n{weather_res}")
    assert "Weather Report" in weather_res, "Weather verification failed!"

    print("\n--- 3. Testing Web Search Tool ---")
    search_res = web_search.invoke({"query": "LangGraph tutorial"})
    print(f"Result:\n{search_res[:150]}...")
    assert "Search" in search_res, "Search verification failed!"

    print("\n--- 4. Testing Text Summarizer Tool ---")
    long_text = (
        "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
        "It extends LangChain by allowing cycle-based flows, which are essential for complex agentic loops. "
        "With LangGraph, developers can build graph structures containing nodes, edges, state reducers, and memory checkpointers."
    )
    summarizer_res = summarize_text.invoke({"text": long_text})
    print(f"Result:\n{summarizer_res}")
    assert "Summary" in summarizer_res, "Summarizer verification failed!"

    print("\nPhase 2 Tools Verification Passed! Ready for Phase 3.")


if __name__ == "__main__":
    test_phase2_tools()
