from __future__ import annotations

import hashlib
import logging
import threading
from typing import Optional

import httpx

from evolve.settings import settings

logger = logging.getLogger(__name__)

_COLLECTION = "evolve_candidates"
_EMBED_DIM = 1536          # text-embedding-3-small output size
_EMBED_MODEL = "text-embedding-3-small"
_MAX_INPUT_CHARS = 8000    # OpenAI embedding input limit (chars)

# Module-level embedding cache: code_hash → vector
# Shared across all MemoryStore instances so the same code is never re-embedded.
_embed_cache: dict[str, list[float]] = {}


def _embed(text: str) -> Optional[list[float]]:
    """
    Embed text via OpenAI text-embedding-3-small.
    Returns None if the API key is missing or the call fails (safe fallback).
    """
    if not settings.openai_api_key:
        return None
    key = hashlib.sha256(text.encode()).hexdigest()
    if key in _embed_cache:
        return _embed_cache[key]
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post(
                f"{settings.openai_base_url.rstrip('/')}/embeddings",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={"model": _EMBED_MODEL, "input": text[:_MAX_INPUT_CHARS]},
            )
            r.raise_for_status()
            vec: list[float] = r.json()["data"][0]["embedding"]
            _embed_cache[key] = vec
            return vec
    except Exception as exc:
        logger.warning("Embedding call failed: %s", exc)
        return None


class QdrantMemory:
    """
    Optional semantic memory backed by Qdrant.

    Each MemoryStore run gets its own ``run_id`` so candidates from different
    runs are isolated within the shared collection.

    Connection is lazy: the first call to upsert() or search() triggers the
    actual network connection so that MemoryStore.__init__ is always fast.

    Retrieval strategy (best_n_semantic):
      1. Embed the query code (current population's best solution).
      2. Search Qdrant for cosine-similar candidates in this run.
      3. Over-fetch (n × 3) then re-rank by fitness → return top-n.

    Falls back silently to an empty list if Qdrant is unreachable or
    embeddings are unavailable; callers must fall back to linear retrieval.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._client = None
        self._ok: bool = False
        self._lock = threading.Lock()

        if settings.qdrant_url:
            # Connect in a daemon thread — never blocks the evolution loop.
            t = threading.Thread(target=self._connect_bg, daemon=True)
            t.start()

    def _connect_bg(self) -> None:
        """Background connection attempt — called once from a daemon thread."""
        if not settings.qdrant_url:
            return
        try:
            from qdrant_client import QdrantClient

            kwargs: dict = {"url": settings.qdrant_url, "timeout": 5.0}
            if settings.qdrant_api_key:
                kwargs["api_key"] = settings.qdrant_api_key

            client = QdrantClient(**kwargs)
            self._ensure_collection(client)
            with self._lock:
                self._client = client
                self._ok = True
            logger.info("Qdrant connected at %s (run=%s)", settings.qdrant_url, self.run_id)
        except Exception as exc:
            logger.warning("Qdrant unavailable, using linear retrieval: %s", exc)

    def _ensure_collection(self, client) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in client.get_collections().collections}
        if _COLLECTION not in existing:
            client.create_collection(
                collection_name=_COLLECTION,
                vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s'", _COLLECTION)

    @property
    def available(self) -> bool:
        """True only after a successful background connection."""
        return self._ok

    def upsert(
        self,
        record_id: str,
        code: str,
        fitness: float,
        generation: int,
    ) -> None:
        """Embed code and store the vector in Qdrant (no-op if not yet connected)."""
        with self._lock:
            if not self._ok:
                return
            client = self._client
        vec = _embed(code)
        if vec is None:
            return
        from qdrant_client.models import PointStruct

        try:
            client.upsert(
                collection_name=_COLLECTION,
                points=[
                    PointStruct(
                        id=record_id,
                        vector=vec,
                        payload={
                            "run_id":     self.run_id,
                            "fitness":    fitness,
                            "generation": generation,
                        },
                    )
                ],
            )
        except Exception as exc:
            logger.warning("Qdrant upsert failed: %s", exc)

    def search(self, query_code: str, n: int) -> list[tuple[str, float]]:
        """
        Return (record_id, fitness) pairs for the top-n most relevant candidates.

        Algorithm:
          - Fetch n*3 candidates by cosine similarity.
          - Re-rank by fitness (primary) so the LLM sees the best-performing
            semantically-similar solutions.
        """
        with self._lock:
            if not self._ok:
                return []
            client = self._client
        vec = _embed(query_code)
        if vec is None:
            return []
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        try:
            hits = client.search(
                collection_name=_COLLECTION,
                query_vector=vec,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="run_id",
                            match=MatchValue(value=self.run_id),
                        )
                    ]
                ),
                limit=n * 3,
                with_payload=True,
            )
        except Exception as exc:
            logger.warning("Qdrant search failed: %s", exc)
            return []

        # Re-rank: primary = fitness, secondary = cosine similarity
        ranked = sorted(
            hits,
            key=lambda h: (h.payload.get("fitness", 0.0), h.score),
            reverse=True,
        )
        return [(str(h.id), h.payload.get("fitness", 0.0)) for h in ranked[:n]]
