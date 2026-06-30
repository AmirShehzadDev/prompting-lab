"""Lesson 1 · measuring tokens before and after a call.

    python measure.py
"""
from google import genai
from google.genai import types

client = genai.Client()
MODEL = "gemini-2.5-flash"

question = "How do I share a Nimbus folder with someone who doesn't have an account?"

# Count input tokens BEFORE sending — useful for budgeting / truncation checks.
# NOTE the attribute name: count_tokens returns .total_tokens (singular).
ct = client.models.count_tokens(model=MODEL, contents=question)
print("estimated input tokens:", ct.total_tokens)

resp = client.models.generate_content(
    model=MODEL,
    contents=question,
    config=types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=300,
        # Turn off the model's internal "thinking" so total == prompt + candidate
        # and the answer isn't truncated by hidden reasoning. (More in Lesson 4.)
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)
print(resp.text)

# Exact accounting AFTER the call lives on usage_metadata (different attribute names!).
u = resp.usage_metadata
print("prompt   :", u.prompt_token_count)       # input tokens actually billed
print("candidate:", u.candidates_token_count)   # output tokens generated
print("total    :", u.total_token_count)
