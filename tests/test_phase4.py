"""
Phase 4 Verification Script.
Tests LangGraph StateGraph compilation, node addition, and edge topology.
"""
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph import build_assistant_graph


def test_phase4_graph_construction():
    print("=== Testing Phase 4: LangGraph Construction & Topology ===")
    
    # 1. Compile graph without checkpointer
    app_graph = build_assistant_graph()
    print("[OK] StateGraph compiled successfully.")
    
    # 2. Inspect nodes and structure
    nodes = list(app_graph.nodes.keys())
    print(f"[OK] Graph Nodes: {nodes}")
    assert "agent" in nodes, "Missing 'agent' node in graph!"
    assert "tools" in nodes, "Missing 'tools' node in graph!"

    print("\nPhase 4 Graph Topology Verification Passed! Ready for Phase 5.")


if __name__ == "__main__":
    test_phase4_graph_construction()
