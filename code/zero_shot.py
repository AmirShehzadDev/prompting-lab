"""Lesson 1 · zero-shot prompting via the shared wrapper.

Run from the repo root so the `nimbus` package is importable:
    python zero_shot.py
"""
from nimbus.llm import LLM

llm = LLM()

# The entire "prompt" is just the user's question. No examples, no persona, no context.
question = "My Nimbus desktop app says 'sync paused' and won't resume. What do I do?"

answer = llm.generate(question, temperature=0.3, max_output_tokens=350)
print(answer)
print("\n---\n" + llm.usage.report())
