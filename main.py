"""
FastAPI backend for Aunt Flo Assistant.

Endpoints:
    POST /chat   - streams a response (SSE) for a query + session_id
    GET  /health - basic liveness check
    POST /reset  - clears a session's history

Run:
    uvicorn main:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import config
import memory
from retrieval import get_retriever

llm_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    memory.init_db()
    #get_retriever()  # warm up embedding/reranker models once at startup
    yield


app = FastAPI(title="Aunt Flo Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


def check_emergency(query: str) -> bool:
    lowered = query.lower()
    return any(kw in lowered for kw in config.EMERGENCY_KEYWORDS)


def build_messages(query: str, session_id: str, context: str) -> list[dict]:
    history = memory.get_history(session_id)
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        }
    )
    return messages


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(session_id: str = "default"):
    memory.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/chat")
async def chat(req: ChatRequest):
    if check_emergency(req.query):
        async def emergency_stream():
            yield {"data": config.EMERGENCY_RESPONSE}
            yield {"event": "done", "data": ""}
        return EventSourceResponse(emergency_stream())

    retriever = get_retriever()
    chunks = retriever.retrieve(req.query)

    context = "\n\n".join(
        f"[{i+1}] {c.text} (Source: {c.source}, p.{c.page})" for i, c in enumerate(chunks)
    )
    sources = [{"source": c.source, "page": c.page} for c in chunks]

    messages = build_messages(req.query, req.session_id, context)
    memory.add_message(req.session_id, "user", req.query)

    async def token_stream():
        full_response = ""
        stream = llm_client.chat.completions.create(
            model=config.LLM_MODEL, messages=messages, stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                yield {"data": delta}

        memory.add_message(req.session_id, "assistant", full_response)
        yield {"event": "sources", "data": str(sources)}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(token_stream())
