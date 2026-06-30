"""Lesson 5 · the linear pipeline spine in LangChain (LCEL).

The classify -> route -> draft chain composes cleanly with the | pipe.
(The critique -> refine LOOP is a conditional cycle -> that's LangGraph territory,
 built in Lesson 7. Here we keep the linear part in LCEL.)

    python pipeline_lc.py

Requires: pip install langchain langchain-google-genai
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from system_prompt import NIMBUS_SYSTEM
from classify_lc import classify          # Lesson 2 LangChain router

ROUTES = {
    "billing":      "Acknowledge warmly. Do NOT promise refunds; offer to escalate to a human billing agent.",
    "cancellation": "Acknowledge. Mention pause/downgrade; offer to escalate to a human.",
    "bug":          "Give troubleshooting steps. Ask for OS, app version, and exact error if missing.",
    "how_to":       "Give clear numbered steps.",
    "other":        "Answer if in scope; otherwise politely decline.",
}

draft_prompt = ChatPromptTemplate.from_messages([
    ("system", NIMBUS_SYSTEM),
    ("human", "Customer message (category: {intent}): {message}\n\n"
              "Handling guidance: {guidance}\n\nWrite the support reply now."),
])
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, max_output_tokens=250, thinking_budget=0)
draft_chain = draft_prompt | llm | StrOutputParser()

# Each .assign() adds a key to the dict flowing down the chain.
pipeline = (
    RunnablePassthrough.assign(intent=lambda x: classify(x["message"]))
    | RunnablePassthrough.assign(guidance=lambda x: ROUTES.get(x["intent"], ROUTES["other"]))
    | RunnablePassthrough.assign(reply=draft_chain)
)

if __name__ == "__main__":
    out = pipeline.invoke({"message": "I was charged $10 twice and I'm annoyed."})
    print(out["intent"], "\n\n", out["reply"])
