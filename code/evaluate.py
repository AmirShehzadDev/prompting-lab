"""Lesson 9 · the Nimbus eval harness: accuracy, LLM-as-judge, groundedness, A/B.

    python evaluate.py
"""
import json
import pathlib
from pydantic import BaseModel, Field
from nimbus.llm import LLM
from classify import classify, classify_zero, normalize   # Lesson 2 routers + coercer


def load_evalset():
    path = pathlib.Path(__file__).parent / "data" / "evalset.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


EVAL = load_evalset()


def accuracy(router) -> float:
    """Metric 1: exact-match accuracy for the router."""
    hits = sum(normalize(router(row["message"])) == row["intent"] for row in EVAL)
    return hits / len(EVAL)


# --- Metric 2: LLM-as-judge ---------------------------------------------------
class Judgment(BaseModel):
    score: int = Field(ge=1, le=5, description="1=poor, 5=excellent")
    passed: bool                       # score >= 4
    reason: str


judge_llm = LLM()
JUDGE_RUBRIC = (
    "You are a STRICT QA reviewer for Nimbus support replies. Score 1-5 on whether the reply: "
    "addresses the question, is on-brand and concise, invents NO Nimbus specifics, and ends with "
    "a clear next step. passed = score >= 4. Be critical."
)


def judge(question: str, reply: str) -> Judgment:
    return judge_llm.generate_json(
        f"Customer: {question}\n\nAgent reply:\n{reply}",
        schema=Judgment, system=JUDGE_RUBRIC, temperature=0,
    )


# --- Metric 3: groundedness (pair with rag.retrieve to get `context`) ---------
class Grounded(BaseModel):
    supported: bool
    unsupported_claims: list[str]


def check_grounded(answer: str, context: str) -> Grounded:
    return judge_llm.generate_json(
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
        "List any claim in the ANSWER that is NOT supported by the CONTEXT.",
        schema=Grounded,
        system="You verify faithfulness. Only context-supported claims count as supported.",
        temperature=0,
    )


if __name__ == "__main__":
    # A/B: prove Lesson 2's claim that few-shot beats zero-shot.
    print(f"zero-shot router: {accuracy(classify_zero):.0%}")
    print(f"few-shot router : {accuracy(classify):.0%}")

    j = judge("Was I double charged?",
              "Yes, I see two $10 charges. I'll escalate the duplicate to billing.")
    print("judge:", j.score, j.passed, "-", j.reason)
