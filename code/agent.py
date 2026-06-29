"""Lesson 3 · a multi-turn support agent that keeps the persona across turns.

    python agent.py
"""
from google.genai import types
from nimbus.llm import LLM
from system_prompt import NIMBUS_SYSTEM


class SupportAgent:
    """A multi-turn support agent. The system prompt is re-sent every turn, so the
    persona + rules persist for the whole conversation; `history` carries context."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.llm = LLM(model=model)
        self.history: list = []  # alternating user / model Content turns

    def ask(self, message: str) -> str:
        self.history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=message)]))
        reply = self.llm.generate(
            self.history,             # the full conversation so far
            system=NIMBUS_SYSTEM,     # standing orders, applied on every turn
            temperature=0.3,
            max_output_tokens=300,
        )
        self.history.append(
            types.Content(role="model", parts=[types.Part.from_text(text=reply)]))
        return reply


if __name__ == "__main__":
    agent = SupportAgent()
    print("A:", agent.ask("My sync is paused and won't resume."))
    print()
    print("A:", agent.ask("Windows 11, app version 3.2."))   # no repeated context needed
    print("---")
    print(agent.llm.usage.report())
