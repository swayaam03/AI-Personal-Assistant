"""
Phase 7 Verification Script.
Tests FastAPI web app endpoints (/api/health, /api/chat, /api/history).
"""
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_phase7_api_endpoints():
    print("=== Testing Phase 7: FastAPI Endpoints ===")
    
    # 1. Test Root /
    root_res = client.get("/")
    print(f"[OK] Root Endpoint status: {root_res.status_code}")
    assert root_res.status_code == 200, "Root endpoint failed!"

    # 2. Test Health Endpoint /api/health
    health_res = client.get("/api/health")
    print(f"[OK] Health Endpoint status: {health_res.status_code}")
    print(f"     Payload: {health_res.json()}")
    assert health_res.status_code == 200, "Health endpoint failed!"

    # 3. Test Chat Endpoint /api/chat
    chat_payload = {
        "message": "What tools do you have access to?",
        "thread_id": "test-api-thread"
    }
    chat_res = client.post("/api/chat", json=chat_payload)
    print(f"[OK] Chat Endpoint status: {chat_res.status_code}")
    print(f"     Response text: {chat_res.json()['response'][:120]}...")
    assert chat_res.status_code == 200, "Chat endpoint failed!"

    # 4. Test History Endpoint /api/history/{thread_id}
    hist_res = client.get("/api/history/test-api-thread")
    print(f"[OK] History Endpoint status: {hist_res.status_code}")
    print(f"     Recorded messages: {len(hist_res.json()['history'])}")
    assert hist_res.status_code == 200, "History endpoint failed!"

    print("\nPhase 7 FastAPI API Layer Verification Passed! Project fully built.")


if __name__ == "__main__":
    test_phase7_api_endpoints()
