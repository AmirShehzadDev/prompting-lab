"""Lesson 4 · chain-of-thought variants on a reasoning-heavy Nimbus task.

    python reasoning.py
"""
import re
from google.genai import types
from nimbus.llm import LLM

NIMBUS_FACTS = """Nimbus storage plans (billed monthly):
- Free: 15 GB, $0/mo
- Plus: 2 TB (2048 GB), $10/mo
- Pro:  10 TB, $25/mo

Quota rule: a shared folder counts against the storage quota of the person who
OWNS it. If you are only a MEMBER of a folder someone else owns, it does NOT
count against your quota. (1 TB = 1024 GB.)"""

QUESTION = (
    "It's late June. My own Nimbus files total 1,600 GB. I'm also a member of a "
    "1590 GB team folder that my colleague owns. I want the cheapest plan that fits, "
    "billed monthly for the 6 remaining months (July-December). "
    "Which plan do I need, and what's my total cost through December?"
)

llm = LLM()  # gemini-2.5-flash

# Disable the model's built-in 'thinking' so we can demonstrate the CLASSIC
# prompting technique in isolation. (Native thinking is shown in answer_thinking.)
NO_THINKING = types.ThinkingConfig(thinking_budget=0)


def answer_direct() -> str:
    """No room to reason -> often falls for the trap."""
    prompt = (f"{NIMBUS_FACTS}\n\n{QUESTION}\n\n"
              "Answer with ONLY this line: FINAL: plan=<name>, total=$<amount>")
    return llm.generate(prompt, temperature=0, max_output_tokens=40,
                        thinking_config=NO_THINKING)


def answer_cot(temperature: float = 0.0) -> str:
    """Zero-shot CoT: reason first, then give the FINAL line."""
    prompt = (f"{NIMBUS_FACTS}\n\n{QUESTION}\n\n"
              "Think step by step. First apply the quota rule, then do the math. "
              "End with a line exactly like: FINAL: plan=<name>, total=$<amount>")
    return llm.generate(prompt, temperature=temperature, max_output_tokens=400,
                        thinking_config=NO_THINKING)


FEWSHOT_COT = """Example 1
Q: I use 30 GB of my own files and I'm a member of a 5 TB folder my manager owns. Cheapest plan for 3 months?
A: The 5 TB folder is owned by my manager; member-only, so it doesn't count. My usage = 30 GB. Free (15 GB) is too small; Plus (2 TB) fits. 3 x $10 = $30.
FINAL: plan=Plus, total=$30

Example 2
Q: I own a 4 TB shared folder and have no other files. Cheapest plan for 2 months?
A: I OWN the folder, so its 4 TB counts against me. Plus (2 TB) is too small; Pro (10 TB) fits. 2 x $25 = $50.
FINAL: plan=Pro, total=$50
"""


def answer_fewshot_cot(temperature: float = 0.0) -> str:
    """Few-shot CoT: worked examples teach the reasoning style + format."""
    prompt = (f"{NIMBUS_FACTS}\n\nHere are worked examples:\n\n{FEWSHOT_COT}\n"
              f"Now solve this one the same way.\nQ: {QUESTION}\nA:")
    return llm.generate(prompt, temperature=temperature, max_output_tokens=400,
                        thinking_config=NO_THINKING)


def answer_thinking(budget: int = 1024) -> str:
    """Native thinking: reason internally (hidden), return a clean final line."""
    prompt = (f"{NIMBUS_FACTS}\n\n{QUESTION}\n\n"
              "Reply with ONLY: FINAL: plan=<name>, total=$<amount>")
    return llm.generate(
        prompt, temperature=0, max_output_tokens=200,
        # budget = token allowance for hidden reasoning; 0 disables, -1 = dynamic.
        thinking_config=types.ThinkingConfig(thinking_budget=budget),
    )


def parse_final(text: str):
    """Pull the structured answer out of the reasoning. Returns (plan, total) or None."""
    m = re.search(r"FINAL:\s*plan=([A-Za-z]+),\s*total=\$?([\d.]+)", text)
    if not m:
        return None
    return (m.group(1).lower(), float(m.group(2)))


if __name__ == "__main__":
    print("direct      :", parse_final(answer_direct()))
    # print("zero-shot CoT:", parse_final(answer_cot()))
    # print("few-shot CoT :", parse_final(answer_fewshot_cot()))
    # print("thinking     :", parse_final(answer_thinking()))
    # print("---")
    # print(llm.usage.report())
