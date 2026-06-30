# AI Knowledge Agent — Intégration et Évaluation d'Agents IA

A full-stack autonomous AI research agent that can ingest documents, search the web, reason over a knowledge base, and evaluate its own performance. Built in Python with LangChain, ChromaDB, and a Streamlit UI.

---

## What this project does

The agent:
1. **Ingests** documents (PDF, TXT, web URLs) into a local vector knowledge base
2. **Retrieves** relevant context using semantic search (RAG)
3. **Searches the web** autonomously via Tavily when local knowledge is insufficient
4. **Reasons** using a ReAct loop (Reason → Act → Observe → Repeat)
5. **Evaluates** answer quality using RAGAS metrics (faithfulness, relevance, recall)
6. **Serves** through a Streamlit chat UI and a FastAPI REST endpoint

---

## Project Structure

```
ai-knowledge-agent/
├── README.md                    ← you are here
├── requirements.txt             ← all dependencies
├── .env.example                 ← copy to .env and fill in keys
│
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── agent.py             ← main ReAct agent loop
│   │   └── memory.py            ← conversation memory management
│   │
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── ingestion.py         ← load & chunk documents
│   │   ├── vectorstore.py       ← ChromaDB setup & retrieval
│   │   └── embeddings.py        ← embedding model wrapper
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── retrieval_tool.py    ← RAG tool for the agent
│   │   ├── web_search_tool.py   ← Tavily web search tool
│   │   └── file_reader_tool.py  ← parse PDF/TXT files on-the-fly
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py         ← RAGAS evaluation pipeline
│       └── metrics.py           ← custom metric helpers
│
├── data/
│   └── docs/                    ← put your documents here
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_agent.py
│
├── scripts/
│   ├── ingest.py                ← CLI: add docs to knowledge base
│   └── evaluate.py              ← CLI: run evaluation suite
│
├── app.py                       ← Streamlit chat UI
└── api.py                       ← FastAPI REST API
```

---

## Stack — 100% Free

| Layer | Tool | Cost |
|---|---|---|
| LLM | Google Gemini 1.5 Flash | FREE — 15 req/min, 1M tokens/day |
| Agent Framework | LangChain | Free, open-source |
| Vector Store | ChromaDB | Free, runs locally |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, runs on CPU |
| Web Search | DuckDuckGo Search | Free, no API key needed |
| Evaluation | RAGAS | Free, open-source |
| UI | Streamlit | Free, open-source |
| API | FastAPI | Free, open-source |

The only key you need is a **Google Gemini API key** — get it free (no credit card) at https://aistudio.google.com/app/apikey

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo>
cd ai-knowledge-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required keys in `.env`:
```
GOOGLE_API_KEY=AIza...    # FREE at https://aistudio.google.com/app/apikey
```

That's it — one key, no credit card, nothing else needed.

### 3. Add documents to your knowledge base

Drop any `.pdf`, `.txt`, or `.md` files into `data/docs/`, then run:

```bash
python scripts/ingest.py
```

You can also ingest a web URL directly:

```bash
python scripts/ingest.py --url https://example.com/article
```

### 4. Run the Streamlit UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 5. Run the REST API

```bash
uvicorn api:app --reload
```

Endpoints:
- `POST /chat` — send a message, get an agent response
- `POST /ingest` — add a document URL to the knowledge base
- `GET /health` — check the service is running
- `GET /docs` — auto-generated Swagger UI

---

## Evaluation

Run the full evaluation suite against a set of question/answer pairs:

```bash
python scripts/evaluate.py
```

This produces a report with RAGAS metrics:
- **Faithfulness** — does the answer only use information from the retrieved context?
- **Answer relevancy** — how well does the answer address the question?
- **Context recall** — did retrieval find the right chunks?
- **Context precision** — were the retrieved chunks actually useful?

Sample output:
```
┌─────────────────────┬────────┐
│ Metric              │ Score  │
├─────────────────────┼────────┤
│ Faithfulness        │ 0.87   │
│ Answer relevancy    │ 0.91   │
│ Context recall      │ 0.83   │
│ Context precision   │ 0.79   │
└─────────────────────┴────────┘
```

---

## How the agent works (ReAct loop)

```
User question
     │
     ▼
 [Think] What do I need to answer this?
     │
     ├── Search knowledge base (RAG tool)
     │       └── Embed query → find top-k chunks → return context
     │
     ├── Search the web (Tavily tool)
     │       └── Query → top web results → summarize
     │
     └── [Answer] Synthesize context + web results into final response
```

The agent decides autonomously which tools to use and in what order. It can call multiple tools before answering.

---

## Adding your own tools

Create a new file in `src/tools/`, define a function, and decorate it with `@tool` from LangChain:

```python
from langchain.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """Describe what this tool does. The agent uses this description to decide when to call it."""
    # your logic here
    return result
```

Then add it to the tools list in `src/agent/agent.py`.

---

## Configuration

All tuneable parameters live in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 512 | Token size per document chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `TOP_K` | 5 | Number of chunks to retrieve |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Local embedding model |
| `LLM_MODEL` | claude-sonnet-4-6 | LLM for the agent |
| `MAX_ITERATIONS` | 10 | Max agent reasoning steps |

---

## Running tests

```bash
pytest tests/ -v
```

---

## Extending the project

Some ideas to take this further:

- **Graph memory** — replace simple vector search with a knowledge graph (Neo4j + LangGraph)
- **Multi-agent** — add a planner agent that delegates to specialist sub-agents
- **Streaming** — stream agent reasoning steps to the UI in real-time
- **Fine-tuning** — use evaluation results to build a fine-tuning dataset
- **Authentication** — add JWT auth to the FastAPI layer for multi-user support

---

## License

MIT
