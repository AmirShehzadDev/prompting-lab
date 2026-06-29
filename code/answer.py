"""Lesson 3 · single-shot answers with the system prompt applied.

    python answer.py
"""
from nimbus.llm import LLM
from system_prompt import NIMBUS_SYSTEM

llm = LLM()  # gemini-2.5-flash


def answer(question: str) -> str:
    return llm.generate(
        question,
        system=NIMBUS_SYSTEM,     # the only change from the Lesson 1 zero-shot call
        temperature=0.3,
        max_output_tokens=300,
    )


if __name__ == "__main__":
    demos = [
        "My Nimbus desktop app says 'sync paused' and won't resume. What do I do?",
        "Can you write me a poem about my cat?",                 # off-topic -> decline
        "I want a refund for this month's charge, right now.",   # -> no promise, escalate
    ]
    for q in demos:
        print("Q:", q)
        print("A:", answer(q))
        print("-" * 60)
    print(llm.usage.report())
