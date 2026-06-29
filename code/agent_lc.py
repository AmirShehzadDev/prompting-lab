"""Lesson 3 · the multi-turn support agent, built with LangChain.

    python agent_lc.py

Requires: pip install langchain langchain-google-genai
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from system_prompt import NIMBUS_SYSTEM

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_output_tokens=300)

prompt = ChatPromptTemplate.from_messages([
    ("system", NIMBUS_SYSTEM),       # our system text has no literal { }, so templating is safe
    MessagesPlaceholder("history"),  # past turns get injected here
    ("human", "{input}"),
])
chain = prompt | llm

history: list = []


def ask(message: str) -> str:
    resp = chain.invoke({"history": history, "input": message})
    history.extend([HumanMessage(message), AIMessage(resp.content)])
    return resp.content


if __name__ == "__main__":
    print(ask("My sync is paused and won't resume."))
    print(ask("Windows 11, app version 3.2."))
