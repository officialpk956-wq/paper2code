import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"


def classify_section(text_chunk: str, max_retries=5):
    prompt = f"""
You are classifying parts of a research paper.

Possible sections:
abstract, introduction, related_work, method, experiments,
results, discussion, conclusion, other

Return valid JSON ONLY:
{{"section": "<section>", "content": "<original_text>"}}

Text:
\"\"\"{text_chunk}\"\"\"
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512,
    }

    for attempt in range(1, max_retries + 1):
        response = requests.post(GROQ_URL, headers=headers, json=payload)

        # ✅ Success
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]

        # ⏳ Rate limited → wait & retry
        if response.status_code == 429:
            wait_time = 2 * attempt  # exponential backoff
            print(f"  ⏳ Rate limited. Sleeping {wait_time}s (retry {attempt}/{max_retries})")
            time.sleep(wait_time)
            continue

        # ❌ Other error → fail fast
        print("Groq error:", response.status_code, response.text)
        response.raise_for_status()

    raise RuntimeError("Groq API failed after retries")

# src/llm_client.py

def llm_complete(prompt: str) -> str:
    """
    Generic LLM completion.
    Returns raw text output.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
