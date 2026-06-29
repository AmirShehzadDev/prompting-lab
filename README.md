# Prompting Lab — Nimbus Support Agent

> **By Amir Shehzad** · a portfolio project for learning production LLM engineering.

A build-it-twice course in modern prompting techniques. Every technique is
implemented two ways — **raw Google Gemini SDK** and **LangChain / LangGraph** —
anchored to one project: an AI support assistant for *Nimbus*, a fictional
cloud file-sync SaaS.


## Lessons

Each lesson covers when to use a technique, why it works, its failure modes, and the cost/latency/reliability
trade-offs — then implements it **twice**: once with the raw Gemini SDK, once with LangChain.

| # | Lesson | Technique | Code (raw · LangChain) |
|--:|--------|-----------|------------------------|
| 1 | Foundations & your first prompt | Zero-shot + a reusable client wrapper (retries, timeout, cost, graceful failure) | [`nimbus/llm.py`](code/nimbus/llm.py) · [`zero_shot.py`](code/zero_shot.py) · [`zero_shot_lc.py`](code/zero_shot_lc.py) |
| 2 | Few-shot prompting | Few-shot in-context learning → an intent **router** | [`classify.py`](code/classify.py) · [`classify_lc.py`](code/classify_lc.py) |
| 3 | Role & system prompting | System prompt: persona, scope, guardrails, refusals | [`system_prompt.py`](code/system_prompt.py) · [`agent.py`](code/agent.py) · [`agent_lc.py`](code/agent_lc.py) |
| 4 | Chain-of-thought & variants | Zero-/few-shot CoT, self-consistency, native "thinking" | [`reasoning.py`](code/reasoning.py) · [`self_consistency.py`](code/self_consistency.py) · [`cot_lc.py`](code/cot_lc.py) |
| 5 | Prompt chaining & self-refine | Pipeline: classify → route → draft → critique → refine (LCEL) | [`pipeline.py`](code/pipeline.py) · [`pipeline_lc.py`](code/pipeline_lc.py) |
| 6 | Structured output & tool use | JSON-schema output (Pydantic) + function calling | [`ticket.py`](code/ticket.py) · [`tools.py`](code/tools.py) · [`structured_lc.py`](code/structured_lc.py) |
| 7 | ReAct-style agents | Reason → act → observe loop with step + budget guards | [`agent_react.py`](code/agent_react.py) · [`agent_react_lc.py`](code/agent_react_lc.py) |
| 8 | Retrieval-augmented prompting (RAG) | Embeddings, retrieval, grounded answers + citations | [`rag.py`](code/rag.py) · [`rag_lc.py`](code/rag_lc.py) |
| 9 | Evaluation & optimization | Accuracy, LLM-as-judge, groundedness, prompt A/B | [`evaluate.py`](code/evaluate.py) · [`judge_lc.py`](code/judge_lc.py) |
| 10 | Capstone + outdated tricks | Full agent assembly + honest myth-busting | [`capstone.py`](code/capstone.py) |

The rendered lessons live in [`lessons/`](lessons/) (open them via the live site above, or locally through `index.html`).

## Read the lessons

Open **`index.html`** in a browser (the lessons are self-contained HTML — no
build step, no external dependencies). It links every lesson in order.

## Run the code

The runnable Python from each lesson also lives in this repo so you can execute
it directly instead of copy-pasting.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# Get a key at https://aistudio.google.com/apikey, then:
export GEMINI_API_KEY="your-key-here"        # macOS/Linux
# $env:GEMINI_API_KEY = "your-key-here"      # Windows PowerShell

cd code   # all runnable scripts (and the data/ + kb/ they use) live here
```

### Lesson 1 — Foundations & zero-shot

```bash
python smoke_test.py        # 5-line "does it work?" check
python measure.py           # token counting + cost accounting
python -m nimbus.llm        # self-test the shared wrapper
python zero_shot.py         # zero-shot baseline via the wrapper (run from repo root)
python zero_shot_lc.py      # the same thing in LangChain
```

### Lesson 2 — Few-shot prompting (the intent router)

```bash
python classify.py          # few-shot router (+ zero-shot baseline), run from repo root
python eval_router.py       # 60-second eval: few-shot vs zero-shot accuracy
python classify_lc.py       # the same router in LangChain (FewShotChatMessagePromptTemplate)
```

### Lesson 3 — Role & system prompting

```bash
python answer.py            # single-shot answers with the Nimbus system prompt + guardrails
python agent.py             # multi-turn SupportAgent (persona persists across turns)
python agent_lc.py          # the same agent in LangChain (system + MessagesPlaceholder)
```

### Lesson 4 — Chain-of-thought & variants

```bash
python reasoning.py         # direct vs zero-shot CoT vs few-shot CoT vs native thinking
python self_consistency.py  # sample N reasoning paths, majority-vote the answer
python cot_lc.py            # CoT chain + self-consistency in LangChain
```

### Lesson 5 — Prompt chaining & self-refine

```bash
python pipeline.py          # classify -> route -> draft -> critique -> refine (raw)
python pipeline_lc.py       # the linear spine in LangChain (LCEL)
```

### Lesson 6 — Structured output & tool use

```bash
python ticket.py            # schema-validated Ticket via Pydantic response_schema
python tools.py             # function calling: automatic + manual loop (account lookup)
python structured_lc.py     # with_structured_output + bind_tools in LangChain
```

### Lesson 7 — ReAct-style agents

```bash
python agent_react.py       # raw reason->act->observe loop with step + budget guards
python agent_react_lc.py    # the same agent via LangChain create_agent (on LangGraph)
```

### Lesson 8 — Retrieval-augmented prompting (RAG)

```bash
python rag.py               # embed the kb/ help center, retrieve, ground answers + cite
python rag_lc.py            # the same with GoogleGenerativeAIEmbeddings + InMemoryVectorStore
```

### Lesson 9 — Evaluation & optimization

```bash
python evaluate.py          # accuracy + LLM-as-judge + groundedness; few-shot vs zero-shot A/B
python judge_lc.py          # LLM-as-judge via LangChain with_structured_output
```

### Lesson 10 — Capstone

```bash
python capstone.py          # the full agent: ticket -> route -> (agent tools | RAG) -> reply
```

## Author

**Amir Shehzad** — I built this course to learn production LLM engineering hands-on, implementing each
technique twice (raw SDK + framework).

## License

[MIT](LICENSE) © 2026 Amir Shehzad


## Layout

```
index.html                 # course hub (the GitHub Pages site)
lessons/                   # one self-contained HTML file per lesson (10 lessons)
code/                      # all runnable Python
  ├─ nimbus/llm.py         #   shared client: retries, timeout, cost tracking, graceful failure
  ├─ data/                 #   fake account store (accounts.json) + eval set (evalset.jsonl)
  ├─ kb/                   #   help-center knowledge base (markdown)
  └─ *.py                  #   the runnable script(s) for each lesson
requirements.txt
LICENSE
```