"""Lesson 7 · the ReAct agent in LangChain via create_agent (runs on LangGraph).

    python agent_react_lc.py

Requires: pip install langchain langchain-google-genai langgraph
"""
import json
import pathlib
from langchain.agents import create_agent          # v1: replaces langgraph.prebuilt.create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from system_prompt import NIMBUS_SYSTEM

_DATA = json.loads((pathlib.Path(__file__).parent / "data" / "accounts.json").read_text())


@tool
def get_subscription(email: str) -> dict:
    """Look up a Nimbus customer's current subscription by email."""
    a = _DATA.get(email)
    return {k: a[k] for k in ("name", "plan", "status", "renews")} if a else {"error": "no account"}


@tool
def get_recent_orders(email: str) -> dict:
    """List a Nimbus customer's recent charges/orders by email."""
    a = _DATA.get(email)
    return {"orders": a["orders"]} if a else {"error": "no account"}


agent = create_agent(
    model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3),
    tools=[get_subscription, get_recent_orders],
    system_prompt=NIMBUS_SYSTEM,
)


if __name__ == "__main__":
    result = agent.invoke({"messages": [
        {"role": "user",
         "content": "I'm alice@example.com. Was I double-charged, and what plan am I on?"}
    ]})
    print(result["messages"][-1].content)      # the final answer
