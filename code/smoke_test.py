"""Lesson 1 · smoke test — if this runs, your install + API key work.

    python smoke_test.py
"""
from google import genai

# Client() reads your key from the GEMINI_API_KEY (or GOOGLE_API_KEY) env var.
client = genai.Client()

resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="In one sentence, what is cloud file sync?",
)
print(resp.text)
