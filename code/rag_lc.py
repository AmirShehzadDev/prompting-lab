"""Lesson 8 · RAG over the Nimbus help center, in LangChain.

    python rag_lc.py

Requires: pip install langchain langchain-google-genai
"""
import pathlib
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

KB_DIR = pathlib.Path(__file__).parent / "kb"
docs = []
for path in sorted(KB_DIR.glob("*.md")):
    for i, para in enumerate(p.strip() for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()):
        docs.append(Document(page_content=para, metadata={"id": f"{path.stem}#{i}"}))

# The framework handles dimensionality + normalization for you.
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
store = InMemoryVectorStore.from_documents(docs, embedding=embeddings)
retriever = store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Nimbus Support. Answer ONLY from the context and cite [id]. "
               "If the answer isn't in the context, say you don't know and offer to escalate."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)


def format_docs(ds):
    return "\n\n".join(f"[{d.metadata['id']}] {d.page_content}" for d in ds)


def rag(question: str) -> str:
    ctx = format_docs(retriever.invoke(question))
    return (prompt | llm | StrOutputParser()).invoke({"context": ctx, "question": question})


if __name__ == "__main__":
    print(rag("My Nimbus app says 'sync paused' and won't resume. What do I do?"))
