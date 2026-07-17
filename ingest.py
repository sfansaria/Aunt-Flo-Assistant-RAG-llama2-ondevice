"""
Ingest PDFs into the RAG knowledge base.

- Splits each PDF into overlapping chunks (with page-level metadata for citations)
- Embeds chunks with a sentence-transformers model, stores them in a local Qdrant collection
- Builds a parallel BM25 sparse index for hybrid retrieval
- Pickles chunk text + BM25 index so the API can load them at startup

Run:
    python ingest.py
"""
import os
import pickle
import uuid
from pathlib import Path

import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config


def load_pdf_chunks() -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = []
    pdf_paths = list(Path(config.PDF_DIR).glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(
            f"No PDFs found in {config.PDF_DIR}/. Add your knowledge base PDFs there first."
        )

    for pdf_path in pdf_paths:
        reader = pypdf.PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            for piece in splitter.split_text(text):
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": piece,
                        "source": pdf_path.name,
                        "page": page_num,
                    }
                )
    return chunks


def build_vector_store(chunks: list[dict]) -> None:
    print(f"Embedding {len(chunks)} chunks with {config.EMBEDDING_MODEL} ...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    vectors = embedder.encode(
        [c["text"] for c in chunks], show_progress_bar=True, batch_size=32
    )

    os.makedirs(Path(config.QDRANT_PATH).parent, exist_ok=True)
    client = QdrantClient(path=config.QDRANT_PATH)
    client.recreate_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=c["id"],
            vector=vec.tolist(),
            payload={"text": c["text"], "source": c["source"], "page": c["page"]},
        )
        for c, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
    print(f"Stored {len(points)} vectors in Qdrant at {config.QDRANT_PATH}")


def build_bm25_index(chunks: list[dict]) -> None:
    print("Building BM25 sparse index ...")
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    os.makedirs(Path(config.BM25_INDEX_PATH).parent, exist_ok=True)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"Stored BM25 index at {config.BM25_INDEX_PATH}")


if __name__ == "__main__":
    chunks = load_pdf_chunks()
    build_vector_store(chunks)
    build_bm25_index(chunks)
    print("Ingestion complete.")
