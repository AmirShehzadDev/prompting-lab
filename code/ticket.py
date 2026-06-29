"""Lesson 6 · schema-validated support tickets via structured output.

    python ticket.py
"""
from typing import Literal
from pydantic import BaseModel, Field
from nimbus.llm import LLM


class Ticket(BaseModel):
    intent: Literal["billing", "bug", "how_to", "cancellation", "other"]
    severity: Literal["low", "medium", "high", "urgent"]
    summary: str = Field(description="One-sentence summary of the customer's issue")
    sentiment: Literal["angry", "frustrated", "neutral", "happy"]
    needs_human: bool = Field(description="True if this must be escalated to a human agent")
    suggested_next_step: str


llm = LLM()


def build_ticket(message: str) -> Ticket:
    return llm.generate_json(
        f"Create a support ticket for this Nimbus message:\n\n{message}",
        schema=Ticket,            # pass the Pydantic class directly
        temperature=0,
    )


if __name__ == "__main__":
    t = build_ticket("I was charged $10 TWICE this month and nobody is helping me!!!")
    print(type(t).__name__, "->", t.intent, t.severity, t.needs_human)
    print(t.summary)
    print("next:", t.suggested_next_step)
    print("---")
    print(llm.usage.report())
