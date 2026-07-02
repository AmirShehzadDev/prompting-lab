"""Lesson 6 · structured output + tools in LangChain.

    python structured_lc.py

Requires: pip install langchain langchain-google-genai
"""
import json
import pathlib
from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool


class Ticket(BaseModel):
    intent: Literal["billing", "bug", "how_to", "cancellation", "other"]
    severity: Literal["low", "medium", "high", "urgent"]
    summary: str = Field(description="One-sentence summary of the issue")
    sentiment: Literal["angry", "frustrated", "neutral", "happy"]
    needs_human: bool
    suggested_next_step: str


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- structured output: returns a populated Ticket instance ---
ticket_llm = llm.with_structured_output(Ticket)

# --- tools ---
_DATA = json.loads((pathlib.Path(__file__).parent / "data" / "accounts.json").read_text())


@tool
def get_recent_orders(email: str) -> dict:
    """List a Nimbus customer's recent charges/orders by email."""
    acct = _DATA.get(email)
    return {"orders": acct["orders"]} if acct else {"error": "no account found"}


llm_tools = llm.bind_tools([get_recent_orders])


if __name__ == "__main__":
    t = ticket_llm.invoke("I was charged $10 twice this month and nobody is helping!!!")
    print("ticket:", t.intent, t.severity, t.needs_human)

    ai = llm_tools.invoke("Was alice@example.com charged twice this month?")
    print("tool calls:", ai.tool_calls)   # the requested call(s) — you execute them yourself
