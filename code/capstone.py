"""Lesson 10 · the complete Nimbus Support Agent, wired from every prior lesson.

    python capstone.py

NOTE: importing this builds the RAG index (embeds kb/) and constructs clients, so it
makes real API calls and needs GEMINI_API_KEY set.
"""
from ticket import build_ticket          # Lesson 6: structured intake
from agent_react import run_agent        # Lesson 7: tool-using agent
from rag import answer as rag_answer      # Lesson 8: grounded help-center answers


def handle(message: str, email: str | None = None) -> dict:
    """The full Nimbus Support Agent: one message in, ticket + reply out."""
    ticket = build_ticket(message)                       # L6: intent, severity, needs_human...

    if ticket.intent in ("billing", "cancellation") and email:
        # Account-specific -> the tool-using agent looks up real data (L7).
        reply = run_agent(f"Customer {email} says: {message}")["answer"]
    else:
        # Knowledge question -> ground the answer in the help center (L8).
        reply = rag_answer(message)

    return {"ticket": ticket, "reply": reply}


if __name__ == "__main__":
    out = handle("I'm alice@example.com and I think I was charged twice!",
                 email="alice@example.com")
    t = out["ticket"]
    print(f"[{t.intent}/{t.severity}] needs_human={t.needs_human}")
    print(out["reply"])

    print("\n" + "=" * 60 + "\n")

    out2 = handle("How do I share a folder with someone who has no account?")
    t2 = out2["ticket"]
    print(f"[{t2.intent}/{t2.severity}] needs_human={t2.needs_human}")
    print(out2["reply"])
