"""Lesson 2 · a 60-second eval: does few-shot beat zero-shot?

    python eval_router.py
"""
from classify import classify, classify_zero, normalize  # the two routers + the coercer

# A small golden set: messages with the label a human says is correct.
TESTSET = [
    ("Refund me for the double charge please",        "billing"),
    ("App freezes whenever I upload a big file",       "bug"),
    ("How can I restore a file I deleted?",            "how_to"),
    ("Cancel my subscription effective today",         "cancellation"),
    ("Are you hiring backend engineers?",              "other"),
    ("Upgrade me to Pro and charge my card",           "billing"),
]


def accuracy(router) -> float:
    # normalize BOTH so the comparison is fair (zero-shot needs the help to even parse).
    hits = sum(normalize(router(msg)) == gold for msg, gold in TESTSET)
    return hits / len(TESTSET)


if __name__ == "__main__":
    print(f"zero-shot accuracy: {accuracy(classify_zero):.0%}")
    print(f"few-shot  accuracy: {accuracy(classify):.0%}")
