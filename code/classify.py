"""Lesson 2 · the Nimbus intent router.

Contains BOTH routers so the eval can compare them:
  * classify()      -> few-shot (the technique this lesson teaches)
  * classify_zero() -> zero-shot baseline (Lesson 1 style), kept for comparison

    python classify.py
"""
from google.genai import types
from nimbus.llm import LLM

LABELS = ["billing", "bug", "how_to", "cancellation", "other"]

SYSTEM = (
    "You are an intent classifier for Nimbus support messages.\n"
    "Classify each message into exactly one of: "
    "billing, bug, how_to, cancellation, other.\n"
    "Reply with ONLY the label — no punctuation, no explanation."
)

# A few labelled demonstrations. These teach the task AND pin the output format.
EXAMPLES = [
    ("I was charged twice this month, can I get a refund?", "billing"),
    ("The desktop app crashes every time I open a shared folder.", "bug"),
    ("How do I move a file into a shared folder?", "how_to"),
    ("I want to close my account and stop being billed.", "cancellation"),
    ("Do you have an office in Berlin?", "other"),
    ("Upgrade me to the Pro plan and bill my card.", "billing"),
]


def build_contents(message: str) -> list:
    """Message-style few-shot: alternating user/model turns, then the real message.

    The API is stateless, so these example turns are re-sent on EVERY call — they are
    how the model 'sees' the pattern, not a saved conversation.
    """
    contents = []
    for msg, label in EXAMPLES:
        contents.append(types.Content(role="user",  parts=[types.Part.from_text(text=msg)]))
        contents.append(types.Content(role="model", parts=[types.Part.from_text(text=label)]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))
    return contents


def normalize(raw: str) -> str:
    """Never trust raw model text. Coerce to a known label or fall back to 'other'."""
    label = raw.strip().lower()
    return label if label in LABELS else "other"


# flash-lite: simple, high-volume task -> use the cheapest model that works.
llm = LLM(model="gemini-2.5-flash-lite")


def classify(message: str) -> str:
    """Few-shot router."""
    raw = llm.generate(
        build_contents(message),
        system=SYSTEM,
        temperature=0,        # deterministic: same message -> same label
        max_output_tokens=8,  # a label is 1-3 tokens; cap hard to save cost + time
    )
    return normalize(raw)


def classify_zero(message: str) -> str:
    """Zero-shot baseline (no examples) — for comparison in eval_router.py."""
    return llm.generate(
        "Classify this Nimbus support message as billing, bug, how_to, "
        f"cancellation, or other:\n\n{message}",
        temperature=0,
        max_output_tokens=30,
    ).strip()


if __name__ == "__main__":
    for m in ["My card got declined but I still see the charge.",
              "Sync keeps failing with error code 0x83.",
              "Where's the button to invite a teammate?"]:
        print(f"{classify(m):13} <- {m}")
    print("---")
    print(llm.usage.report())
