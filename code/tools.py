"""Lesson 6 · tool use (function calling) grounded in a fake account datastore.

    python tools.py

Shows automatic function calling, then the manual call-execute-respond loop.
"""
import json
import pathlib
from google import genai
from google.genai import types
from system_prompt import NIMBUS_SYSTEM   # Lesson 3 persona

_DATA = json.loads((pathlib.Path(__file__).parent / "data" / "accounts.json").read_text())


def get_subscription(email: str) -> dict:
    """Look up a Nimbus customer's current subscription by email.

    Args:
        email: the customer's account email address.
    """
    acct = _DATA.get(email)
    if not acct:
        return {"error": "no account found for that email"}
    return {"name": acct["name"], "plan": acct["plan"],
            "status": acct["status"], "renews": acct["renews"]}


def get_recent_orders(email: str) -> dict:
    """List a Nimbus customer's recent charges/orders by email.

    Args:
        email: the customer's account email address.
    """
    acct = _DATA.get(email)
    if not acct:
        return {"error": "no account found for that email"}
    return {"orders": acct["orders"]}


client = genai.Client()
MODEL = "gemini-2.5-flash"
TOOLS = [get_subscription, get_recent_orders]
REGISTRY = {"get_subscription": get_subscription, "get_recent_orders": get_recent_orders}


def automatic(user_msg: str) -> str:
    """The SDK runs the whole call-execute-respond round trip for you."""
    resp = client.models.generate_content(
        model=MODEL,
        contents=user_msg,
        config=types.GenerateContentConfig(tools=TOOLS, system_instruction=NIMBUS_SYSTEM),
    )
    return resp.text


def manual(user_msg: str) -> str:
    """The same thing, but we drive the loop (the control point for agents)."""
    cfg = types.GenerateContentConfig(
        tools=TOOLS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    first = client.models.generate_content(model=MODEL, contents=user_msg, config=cfg)
    fc = first.candidates[0].content.parts[0].function_call   # .name and .args
    print("model wants:", fc.name, dict(fc.args))

    result = REGISTRY[fc.name](**fc.args)                      # WE execute (validate/log here)

    final = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text=user_msg)]),
            first.candidates[0].content,                       # the model's function-call turn
            types.Content(role="user",
                          parts=[types.Part.from_function_response(
                              name=fc.name, response={"result": result})]),
        ],
        config=cfg,
    )
    return final.text


if __name__ == "__main__":
    print("AUTOMATIC:")
    print(automatic("Hi, I'm alice@example.com - was I charged twice this month?"))
    print("\nMANUAL:")
    print(manual("Is carol@example.com's account in good standing?"))
