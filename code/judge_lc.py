"""Lesson 9 · LLM-as-judge in LangChain via with_structured_output.

    python judge_lc.py

Requires: pip install langchain langchain-google-genai
"""
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI


class Judgment(BaseModel):
    score: int = Field(ge=1, le=5)
    passed: bool
    reason: str


judge = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).with_structured_output(Judgment)


def evaluate_reply(question: str, reply: str) -> Judgment:
    return judge.invoke(
        "You are a STRICT QA reviewer for Nimbus support replies. Score 1-5 (passed = score >= 4) on: "
        "addresses the question, on-brand and concise, no invented specifics, has a next step.\n\n"
        f"Customer: {question}\nReply: {reply}"
    )


if __name__ == "__main__":
    j = evaluate_reply("Was I double charged?",
                       "Yes, two $10 charges. I'll escalate the duplicate.")
    print(j.score, j.passed, "-", j.reason)
