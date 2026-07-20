# Aunt Flo Assistant 🌸

A retrieval-augmented chatbot for menstrual and reproductive health education — rebuilt on a
modern, production-ready RAG stack and deployed as three independent services on Fly.io.

**Live app:** https://aunt-flo-frontend.fly.dev

## What changed from the original

| | Before | Now |
|---|---|---|
| Model format | GGML (deprecated) | GGUF via **Ollama** |
| Retrieval | FAISS, dense only | **Qdrant** + **BM25 hybrid search** + cross-encoder **reranker** (toggleable) |
| Embeddings | implicit/basic | **BAAI/bge-small-en-v1.5** |
| Architecture | single Streamlit script | **FastAPI backend** + Streamlit frontend, talking over SSE |
| Memory | in-process Streamlit state | **SQLite**-backed session history |
| Citations | none | source + page shown per answer |
| Safety | none | emergency-keyword short-circuit + disclaimer system prompt |
| Deployment | manual `streamlit run` | Three **Docker** containers, deployed independently on **Fly.io** |
| Quality checks | none | lightweight keyword-based eval set |

## Architecture

The app is split into **three separate Fly.io apps**, each in its own container, communicating
over public HTTPS:

```
                    ┌─────────────────────┐
   User's browser → │  aunt-flo-frontend   │  (Streamlit)
                    └──────────┬──────────┘
                               │ HTTPS
                               ▼
                    ┌─────────────────────┐
                    │ auntfloassistant-    │  (FastAPI + Qdrant + BM25)
                    │ hybridragondevice    │
                    └──────────┬──────────┘
                               │ HTTPS
                               ▼
                    ┌─────────────────────┐
                    │   aunt-flo-ollama    │  (Ollama, qwen2.5:1.5b)
                    └─────────────────────┘
```

**Why public HTTPS between the apps, not Fly's private networking?** Fly's private network
(`.internal` / `.flycast`) requires dual-stack IPv6/IPv4 binding and DNS provisioning that this
setup didn't have configured correctly, causing intermittent connection failures. Every app talks
to the others over its normal public `https://*.fly.dev` URL instead — proven reliable in testing,
still fully encrypted, and simpler to reason about for a project this size.

Ingestion flow (run once, offline, whenever the source PDFs change):
```
PDFs → ingest.py → Qdrant (dense vectors) + BM25 (sparse index)
```

Query flow (every chat message):
```
User question → Streamlit → FastAPI → hybrid retrieval + rerank → Ollama → streamed, cited answer
```

## Repo structure

```
.
├── main.py              FastAPI backend: retrieval + generation + streaming + safety check
├── retrieval.py          Hybrid search (Qdrant + BM25) with optional cross-encoder reranking
├── ingest.py             One-off script: PDFs → chunks → embeddings → Qdrant + BM25
├── memory.py             SQLite-backed conversation history per session
├── config.py             Central config, all overridable via env vars / Fly secrets
├── app.py                 Streamlit frontend, calls the API over SSE
├── Dockerfile             API container image
├── Dockerfile.frontend    Frontend container image
├── docker-compose.yml     Local multi-container setup (API + frontend; Ollama runs natively)
├── fly.toml               Fly config for the API app
├── eval/                  Lightweight keyword-based answer-quality checks
├── tests/                 Smoke tests (safety-check logic), run in CI
└── .github/workflows/     CI: runs tests on every push
```

The Ollama and frontend Fly apps live in sibling directories (`aunt-flo-ollama/`,
`aunt-flo-frontend/`) with their own minimal `Dockerfile` + `fly.toml`, since Fly deploys one app
per directory rather than orchestrating multi-service Compose files the way Render does.

## Local development

### 1. Install Ollama and pull a model

```bash
curl -fsSL https://ollama.com/install.sh | sh   # or `brew install ollama` on Mac
ollama serve &
ollama pull qwen2.5:1.5b
```

### 2. Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

### 4. Add your knowledge base and ingest it

Drop PDFs into `data/`, then:

```bash
python ingest.py
```

### 5. Run the backend and frontend (two terminals)

```bash
uvicorn main:app --reload --port 8000
```
```bash
streamlit run app.py
```

Open http://localhost:8501.

### Running with Docker locally

```bash
docker compose up --build
```

Ollama runs natively on the host (for GPU/Metal acceleration on Mac); the API container reaches
it via `host.docker.internal`. Run `python ingest.py` once against the mounted `vectorstore/`
volume before using the app.

## Production deployment (Fly.io)

Each service is its own Fly app:

| App | Purpose | Key config |
|---|---|---|
| `auntfloassistanthybridragondevice` | FastAPI backend | 2GB+ RAM, persistent volume at `/data` for vector store + sessions |
| `aunt-flo-ollama` | LLM serving | 2GB+ RAM, persistent volume at `/root/.ollama` for model weights |
| `aunt-flo-frontend` | Streamlit UI | Lightweight, no persistent storage needed |

All three set `min_machines_running = 1` to stay warm (no cold-start delay).

**Key secrets** (set via `fly secrets set`):
```
# On the API app
LLM_BASE_URL=https://aunt-flo-ollama.fly.dev/v1
LLM_MODEL=qwen2.5:1.5b
USE_RERANKER=false
QDRANT_PATH=/data/vectorstore/qdrant_db
BM25_INDEX_PATH=/data/vectorstore/bm25.pkl
SESSION_DB_PATH=/data/sessions.db

# On the frontend app
API_URL=https://auntfloassistanthybridragondevice.fly.dev
```

**One-time setup on the deployed API app** (PDFs and the vector store live on the persistent
volume, not in the Docker image):
```bash
fly ssh sftp shell -a auntfloassistanthybridragondevice
# put your PDFs into /data/pdfs/

fly ssh console -a auntfloassistanthybridragondevice
python ingest.py
```

**Pull the model onto the Ollama app once:**
```bash
fly ssh console -a aunt-flo-ollama
ollama pull qwen2.5:1.5b
```

## Evaluation

```bash
python eval/run_eval.py
```

Runs a small set of test questions (`eval/qa_set.json`) against the running API and checks
whether expected keywords appear in the answers — catches obvious retrieval/prompt regressions
before they reach users.

## Notes on cost, performance, and safety

- Ollama on Fly runs **CPU-only** (no GPU on standard tiers) — expect several seconds per response,
  and only one request processed at a time (`OLLAMA_NUM_PARALLEL=1`). For heavier traffic, swap
  `LLM_BASE_URL` to a hosted LLM API instead — no other code changes needed.
- This is a health-information assistant, not a medical device. `config.py`'s
  `EMERGENCY_KEYWORDS` list and system prompt disclaimer are a starting point, not a complete
  safety solution — review and expand deliberately before wider release.
- Running three always-on Fly machines has an ongoing cost; scale `min_machines_running` back to
  `0` on the frontend/API if idle-time cost matters more than instant response for early users.

## License

MIT — see `LICENSE`.