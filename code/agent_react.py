"""Lesson 7 · a raw ReAct agent: reason -> act -> observe, looped, with hard guards.

Reuses the Lesson 6 tools (client, TOOLS, REGISTRY) and the Lesson 3 persona.

    python agent_react.py
"""
from google.genai import types
from tools import client, TOOLS, REGISTRY     # Lesson 6: client + the two account tools
from system_prompt import NIMBUS_SYSTEM       # Lesson 3 persona

MODEL = "gemini-2.5-flash"
CFG = types.GenerateContentConfig(
    tools=TOOLS,
    system_instruction=NIMBUS_SYSTEM,
    # Turn OFF auto-calling: WE drive the loop, so we can log + enforce guards.
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def run_agent(user_msg: str, max_steps: int = 6, token_budget: int = 20_000) -> dict:
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_msg)])]
    tokens = 0
    for step in range(1, max_steps + 1):                       # LOOP GUARD: bounded steps
        resp = client.models.generate_content(model=MODEL, contents=contents, config=CFG)
        tokens += resp.usage_metadata.total_token_count or 0
        turn = resp.candidates[0].content
        contents.append(turn)                                  # remember the model's move

        calls = [p.function_call for p in turn.parts if p.function_call]
        if not calls:                                          # no tool wanted -> we're done
            return {"answer": resp.text, "steps": step, "tokens": tokens}
        if tokens > token_budget:                              # BUDGET GUARD
            return {"answer": "Stopped: token budget exceeded.", "steps": step, "tokens": tokens}

        results = []
        for fc in calls:                                       # ACT + OBSERVE
            print(f"  step {step}: {fc.name}({dict(fc.args)})")
            try:
                out = REGISTRY[fc.name](**fc.args)
            except Exception as e:                             # tool errors don't crash the agent
                out = {"error": str(e)}
            results.append(types.Part.from_function_response(
                name=fc.name, response={"result": out}))
        contents.append(types.Content(role="user", parts=results))

    return {"answer": "Stopped: step limit reached.", "steps": max_steps, "tokens": tokens}


if __name__ == "__main__":
    out = run_agent("I'm alice@example.com. Was I double-charged, and what plan am I on?")
    print(f"\n[{out['steps']} steps, {out['tokens']} tokens]\n{out['answer']}")
