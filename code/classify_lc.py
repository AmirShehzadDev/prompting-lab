"""Lesson 2 · the intent router, built with LangChain's FewShotChatMessagePromptTemplate.

    python classify_lc.py

Requires: pip install langchain langchain-google-genai
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

LABELS = {"billing", "bug", "how_to", "cancellation", "other"}

examples = [
    {"input": "I was charged twice this month, can I get a refund?", "output": "billing"},
    {"input": "The desktop app crashes every time I open a shared folder.", "output": "bug"},
    {"input": "How do I move a file into a shared folder?", "output": "how_to"},
    {"input": "I want to close my account and stop being billed.", "output": "cancellation"},
    {"input": "Do you have an office in Berlin?", "output": "other"},
    {"input": "Upgrade me to the Pro plan and bill my card.", "output": "billing"},
]

# Shape of ONE example: a human turn followed by the ai (model) answer.
example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])

# This expands `examples` into alternating human/ai turns automatically.
few_shot = FewShotChatMessagePromptTemplate(examples=examples, example_prompt=example_prompt)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the Nimbus support message into exactly one label: "
               "billing, bug, how_to, cancellation, or other. Reply with ONLY the label."),
    few_shot,
    ("human", "{input}"),
])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, max_output_tokens=8)
chain = final_prompt | llm | StrOutputParser()      # prompt -> model -> plain string


def classify(message: str) -> str:
    raw = chain.invoke({"input": message}).strip().lower()
    return raw if raw in LABELS else "other"


if __name__ == "__main__":
    for m in ["My card got declined but I still see the charge.",
              "Sync keeps failing with error code 0x83."]:
        print(classify(m), "<-", m)
