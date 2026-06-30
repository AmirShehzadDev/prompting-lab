"""Lesson 1 · the same zero-shot baseline, built with LangChain.

    python zero_shot_lc.py

Requires: pip install langchain langchain-google-genai
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# LangChain reads GOOGLE_API_KEY (falls back to GEMINI_API_KEY).
# max_retries gives you backoff for free — the framework's version of our retry loop.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_output_tokens=350,
    max_retries=5,
    thinking_budget=0,   # disable internal "thinking" so it doesn't eat the output budget
)

question = "My Nimbus desktop app says 'sync paused' and won't resume. What do I do?"
resp = llm.invoke([HumanMessage(question)])

print(resp.content)                 # the answer text
print(resp.usage_metadata)          # {'input_tokens': .., 'output_tokens': .., 'total_tokens': ..}
