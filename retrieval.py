"""
Hybrid retrieval: dense (Qdrant) + sparse (BM25) candidates, merged and
reranked with a cross-encoder. Loaded once at API startup and reused.
"""
import pickle
from dataclasses import dataclass

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

import config


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    score: float = 0.0


class Retriever:
    def __init__(self):
        print("Loading embedding model ...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        print("Loading Qdrant collection ...")
        self.qdrant = QdrantClient(path=config.QDRANT_PATH)

        print("Loading BM25 index ...")
        with open(config.BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.bm25_chunks = data["chunks"]

        self.reranker = None
        if config.USE_RERANKER:
            print("Loading reranker ...")
            self.reranker = CrossEncoder(config.RERANKER_MODEL)

    def _dense_search(self, query: str, k: int) -> list[Chunk]:
        vec = self.embedder.encode(query).tolist()
        hits = self.qdrant.search(
            collection_name=config.QDRANT_COLLECTION, query_vector=vec, limit=k
        )
        return [
            Chunk(text=h.payload["text"], source=h.payload["source"], page=h.payload["page"], score=h.score)
            for h in hits
        ]

    def _sparse_search(self, query: str, k: int) -> list[Chunk]:
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            Chunk(
                text=self.bm25_chunks[i]["text"],
                source=self.bm25_chunks[i]["source"],
                page=self.bm25_chunks[i]["page"],
                score=scores[i],
            )
            for i in top_idx
        ]

    @staticmethod
    def _dedupe(chunks: list[Chunk]) -> list[Chunk]:
        seen, out = set(), []
        for c in chunks:
            key = (c.source, c.page, c.text[:50])
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out

    def retrieve(self, query: str) -> list[Chunk]:
        dense = self._dense_search(query, config.RETRIEVAL_TOP_K)
        sparse = self._sparse_search(query, config.RETRIEVAL_TOP_K)
        candidates = self._dedupe(dense + sparse)

        if self.reranker and candidates:
            pairs = [(query, c.text) for c in candidates]
            scores = self.reranker.predict(pairs)
            for c, s in zip(candidates, scores):
                c.score = float(s)
            candidates.sort(key=lambda c: c.score, reverse=True)

        return candidates[: config.FINAL_TOP_K]


# Singleton, loaded once when the API process starts.
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
