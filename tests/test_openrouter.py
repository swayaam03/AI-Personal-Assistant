"""Test the best free OpenRouter models for our assistant."""
import warnings
warnings.filterwarnings("ignore")
import os
from dotenv import load_dotenv
load_dotenv()

import httpx
import json

api_key = os.getenv("OPENROUTER_API_KEY", "")
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/swayaam03/AI-Personal-Assistant",
}

# Best free models to test (strong reasoning + tool calling support)
test_models = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
]

for model in test_models:
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello and tell me your name in one sentence."}],
        "max_tokens": 100,
    }
    print(f"Testing: {model}")
    try:
        resp = httpx.post(url, headers=headers, json=data, timeout=30.0)
        result = resp.json()
        if "choices" in result:
            content = result["choices"][0]["message"]["content"]
            print(f"  OK: {content[:120]}")
        elif "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  RAW: {json.dumps(result)[:200]}")
    except Exception as e:
        print(f"  EXCEPTION: {e}")
    print()

print("Done.")
