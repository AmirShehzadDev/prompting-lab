"""Lesson 5 · the Nimbus support pipeline: classify -> route -> draft -> critique -> refine.

Reuses the Lesson 2 router (classify) and the Lesson 3 persona (NIMBUS_SYSTEM).

    python pipeline.py
"""
from nimbus.llm import LLM
from classify import classify              # Lesson 2 router (uses gemini-2.5-flash-lite)
from system_prompt import NIMBUS_SYSTEM    # Lesson 3 persona

drafter = LLM(model="gemini-2.5-flash")    # capable model for the customer-facing reply
critic = LLM(model="gemini-2.5-flash")

# Stage 2 is just a lookup: each intent maps to handling guidance for the drafter.
ROUTES = {
    "billing":      "Acknowledge warmly. Do NOT promise refunds; offer to escalate to a human billing agent.",
    "cancellation": "Acknowledge. Mention they can pause or downgrade instead; offer to escalate to a human.",
    "bug":          "Give concrete troubleshooting steps. Ask for OS, app version, and exact error if missing.",
    "how_to":       "Give clear numbered steps.",
    "other":        "Answer if it's in scope for Nimbus; otherwise politely decline.",
}

RUBRIC = """Check the draft reply against these:
1) Directly addresses the customer's message.
2) Follows the handling guidance for its category.
3) On-brand: friendly, concise, invents NO Nimbus specifics.
4) Ends with a clear next step.
Reply with exactly "PASS" if all hold, otherwise "FAIL: <one line on what to fix>"."""


def draft(message: str, intent: str) -> str:
    """Stage 3: write the reply, using the route's guidance + the Nimbus persona."""
    guidance = ROUTES.get(intent, ROUTES["other"])
    prompt = (f"Customer message (category: {intent}): {message}\n\n"
              f"Handling guidance: {guidance}\n\n"
              "Write the support reply now.")
    return drafter.generate(prompt, system=NIMBUS_SYSTEM,
                            temperature=0.4, max_output_tokens=250)


def critique(message: str, intent: str, reply: str) -> str:
    """Stage 4: judge the draft. temperature=0 so the verdict is stable."""
    prompt = (f"Customer (category: {intent}): {message}\n\n"
              f"Draft reply:\n{reply}\n\n{RUBRIC}")
    return critic.generate(prompt, temperature=0, max_output_tokens=80)


def refine(message: str, intent: str, reply: str, feedback: str) -> str:
    """Stage 5: rewrite the draft to address the critic's feedback."""
    prompt = (f"Customer (category: {intent}): {message}\n\n"
              f"Your previous draft:\n{reply}\n\n"
              f"A reviewer said: {feedback}\n\n"
              "Rewrite the reply to fix that. Reply with ONLY the improved message.")
    return drafter.generate(prompt, system=NIMBUS_SYSTEM,
                            temperature=0.4, max_output_tokens=250)


def handle(message: str, max_rounds: int = 2) -> dict:
    """The full pipeline: classify -> route -> draft -> (critique -> refine)*"""
    intent = classify(message)                 # stage 1
    reply = draft(message, intent)             # stages 2-3

    for r in range(max_rounds):                # stages 4-5, bounded loop
        verdict = critique(message, intent, reply)
        if verdict.strip().upper().startswith("PASS"):
            return {"intent": intent, "reply": reply, "refines": r, "status": "passed"}
        reply = refine(message, intent, reply, verdict)

    return {"intent": intent, "reply": reply, "refines": max_rounds, "status": "max_rounds"}


if __name__ == "__main__":
    out = handle("Sync keeps stopping and I don't know why. Fix it.")
    print(f"[{out['intent']}] refines={out['refines']} ({out['status']})\n")
    print(out["reply"])
    print("---")
    print(drafter.usage.report(), "(drafter)")
    print(critic.usage.report(), "(critic)")
