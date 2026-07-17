# Aunt Flo Assistant 🌸

A retrieval-augmented chatbot for menstrual and reproductive health education —
rebuilt on a modern, production-ready RAG stack.

## What changed from the original

| | Before | Now |
|---|---|---|
| Model format | GGML (deprecated) | GGUF via **Ollama** (or any hosted LLM API) |
| Retrieval | FAISS, dense only | **Qdrant** + **BM25 hybrid search** + cross-encoder **reranker** |
| Embeddings | implicit/basic | **BAAI/bge-small-en-v1.5** |
| Architecture | single Streamlit script | **FastAPI backend** + Streamlit frontend, talking over SSE |
| Memory | in-process Streamlit state | **SQLite**-backed session history |
| Citations | none | source + page shown per answer |
| Safety | none | emergency-keyword short-circuit + disclaimer system prompt |
| Deployment | manual `streamlit run` | Docker + docker-compose + CI |
| Quality checks | none | lightweight keyword-based eval set |

## Detailed changelog vs. the original repo

**Model & inference**
- Removed `ctransformers` + GGML weights (`llama-2-7b-chat.ggmlv3.q4_0.bin`) — GGML has been deprecated in favor of GGUF since 2023 and ctransformers is unmaintained.
- Inference now goes through **Ollama's OpenAI-compatible API**, defaulting to `llama3.1:8b`. Swappable to any GGUF model or a hosted LLM API purely via `.env` — no code changes.

**Retrieval (previously: raw FAISS, no metadata, no reranking)**
- `ingest.py` rewritten: chunks now carry `source` and `page` metadata (needed for citations), embeddings come from a real sentence-transformers model (`bge-small-en-v1.5`) instead of whatever FAISS's default was.
- Vector store moved from FAISS to **Qdrant** (adds metadata filtering, easier to swap to a networked instance later).
- New: a parallel **BM25 sparse index** (`rank_bm25`), merged with dense results for **hybrid search** — meaningfully better recall on medical/domain terms.
- New: a **cross-encoder reranker** (`bge-reranker-base`) reorders hybrid candidates before they reach the LLM.

**Generation**
- New: **source citations** — every answer is grounded in numbered context passages `[1] [2]`, and the frontend shows the source PDF + page per answer. The original had no citation mechanism at all.
- New: **emergency-keyword safety check** (`config.py`) that short-circuits generation and returns a "see a doctor" message for concerning queries, plus a disclaimer baked into the system prompt. Not present before.

**Architecture (previously: a single `app.py` Streamlit script doing everything)**
- Split into a **FastAPI backend** (`main.py`) that owns retrieval + generation + streaming, and a thin **Streamlit frontend** (`app.py`) that just calls the API over SSE. This is what makes independent scaling/deployment and a future non-Streamlit frontend possible.
- Conversation memory moved from Streamlit's in-process session state to a **SQLite-backed store** (`memory.py`) — survives restarts, works across multiple frontend instances.
- Responses now **stream token-by-token** to the frontend instead of arriving all at once.

**Deployment (previously: none — `streamlit run app.py` locally only)**
- New: `Dockerfile` (API), `Dockerfile.frontend` (Streamlit), and `docker-compose.yml` wiring up Ollama + API + frontend as separate services.
- New: GitHub Actions CI (`.github/workflows/ci.yml`) running tests on every push, with a commented-out deploy job as a starting point.

**Quality & testing (previously: none)**
- New: `eval/qa_set.json` + `eval/run_eval.py` — a lightweight keyword-based eval loop to catch retrieval/prompt regressions.
- New: `tests/test_api.py` — smoke tests for the safety-check logic, run in CI.

## Architecture

```
PDFs → ingest.py → Qdrant (dense) + BM25 (sparse)
                          │
                    retrieval.py (hybrid search + rerank)
                          │
Streamlit (app.py) ──SSE──▶ FastAPI (main.py) ──▶ LLM (Ollama or hosted API)
                          │
                     memory.py (SQLite session history)
```

The LLM backend is swappable via three env vars (`LLM_BASE_URL`, `LLM_API_KEY`,
`LLM_MODEL`) — no code changes needed to move from local Ollama to a hosted API.

## Setup

### 1. Install Ollama and pull a model (skip if using a hosted API instead)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama serve
```

### 2. Install Python dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# edit .env if you want a hosted LLM API, different models, etc.
```

### 4. Add your knowledge base and ingest it

Drop PDFs into `data/`, then:

```bash
python ingest.py
```

This builds the Qdrant vector store and BM25 index under `vectorstore/`.

### 5. Run the backend

```bash
uvicorn main:app --reload --port 8000
```

### 6. Run the frontend

```bash
streamlit run app.py
```

Open http://localhost:8501.

## Running with Docker

```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.1:8b   # first run only
```

Then run ingestion once (either locally against the mounted `vectorstore/`
volume, or by exec-ing into the `api` container) before using the app.

## Evaluation

After changing the prompt, retrieval settings, or embedding model, sanity-check
quality with the small eval set:

```bash
python eval/run_eval.py
```

Edit `eval/qa_set.json` to add real questions from your own knowledge base —
the shipped set is just a starting example.

## Deployment notes

- The `api` and `frontend` services are independent — deploy them as separate
  services/containers so you can scale or restart them independently.
- Qdrant here runs in embedded/file mode for simplicity. For multi-instance
  deployments, switch to a standalone Qdrant server (`qdrant/qdrant` image)
  and point `QdrantClient` at it via URL instead of a local path.
- Ollama needs real CPU/RAM (4 vCPU / 8GB is a reasonable floor for an 8B
  quantized model). If cost or latency is a concern, swap in a hosted LLM API
  via `.env` instead of self-hosting inference.
- Tighten the CORS origin in `main.py` before exposing this publicly.
- This is a health-information assistant, not a medical device — the system
  prompt and emergency-keyword check in `config.py` are a starting point, not
  a complete safety solution. Review and expand `EMERGENCY_KEYWORDS` deliberately.

## License

MIT — see `LICENSE`.
