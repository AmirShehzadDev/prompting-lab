"""Lesson 4 · chain-of-thought + self-consistency, built with LangChain.

    python cot_lc.py

Requires: pip install langchain langchain-google-genai
"""
from collections import Counter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from reasoning import NIMBUS_FACTS, QUESTION, parse_final   # reuse facts + the parser

# temperature > 0 so repeated runs differ (required for self-consistency)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8, max_output_tokens=400, thinking_budget=0)

prompt = ChatPromptTemplate.from_messages([
    ("human", "{facts}\n\n{question}\n\nThink step by step, then end with a line "
              "exactly like: FINAL: plan=<name>, total=$<amount>"),
])
chain = prompt | llm | StrOutputParser()        # prompt -> model -> text


def self_consistent(n: int = 5):
    runs = (chain.invoke({"facts": NIMBUS_FACTS, "question": QUESTION}) for _ in range(n))
    finals = [p for p in (parse_final(t) for t in runs) if p]
    return Counter(finals).most_common(1)[0] if finals else None


if __name__ == "__main__":
    once = parse_final(chain.invoke({"facts": NIMBUS_FACTS, "question": QUESTION}))
    print("single CoT     :", once)
    print("self-consistent:", self_consistent())   # ((plan, total), votes)
