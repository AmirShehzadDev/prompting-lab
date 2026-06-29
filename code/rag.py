"""Lesson 8 · retrieval-augmented generation over the Nimbus help center (raw).

    python rag.py
"""
import math
import pathlib
from google import genai
from google.genai import types
from nimbus.llm import LLM
from system_prompt import NIMBUS_SYSTEM

client = genai.Client()
EMBED_MODEL = "gemini-embedding-001"
DIM = 768   # 128-3072 supported; smaller = cheaper to store/compare


def l2_normalize(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def embed(texts, task_type):
    """task_type: 'RETRIEVAL_DOCUMENT' for the KB, 'RETRIEVAL_QUERY' for questions."""
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=DIM, task_type=task_type),
    )
    # gemini-embedding-001 needs manual L2 normalization when dim != 3072.
    return [l2_normalize(e.values) for e in resp.embeddings]


def cosine(a, b):                       # both already normalized -> cosine == dot product
    return sum(x * y for x, y in zip(a, b))


KB_DIR = pathlib.Path(__file__).parent / "kb"


def load_chunks():
    """Split each .md article into paragraph chunks with a citable id."""
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")):
        paras = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]
        for i, para in enumerate(paras):
            chunks.append({"id": f"{path.stem}#{i}", "text": para})
    return chunks


CHUNKS = load_chunks()                                              # ~12 small passages
DOC_VECS = embed([c["text"] for c in CHUNKS], task_type="RETRIEVAL_DOCUMENT")  # embed ONCE


def retrieve(query, k=3):
    qv = embed([query], task_type="RETRIEVAL_QUERY")[0]
    ranked = sorted(zip(CHUNKS, DOC_VECS),
                    key=lambda cv: cosine(qv, cv[1]), reverse=True)
    return [c for c, _ in ranked[:k]]


llm = LLM()

RAG_RULES = (
    "\n\nGROUNDING:\n"
    "- Answer ONLY using the provided context passages.\n"
    "- Cite the passage id(s) you used in brackets, e.g. [sync-errors#0].\n"
    "- If the answer is not in the context, say you don't have that information "
    "and offer to escalate to a human. Do NOT guess."
)


def answer(query, k=3):
    hits = retrieve(query, k)
    context = "\n\n".join(f"[{c['id']}] {c['text']}" for c in hits)
    prompt = f"Context passages:\n{context}\n\nCustomer question: {query}"
    return llm.generate(prompt, system=NIMBUS_SYSTEM + RAG_RULES,
                        temperature=0.2, max_output_tokens=300)


if __name__ == "__main__":
    print("Q1 (in KB):")
    print(answer("My Nimbus app says 'sync paused' and won't resume. What do I do?"))
    print("\nQ2 (not in KB):")
    print(answer("Does Nimbus have an app for smartwatches?"))
    print("\n---")
    print(llm.usage.report())
