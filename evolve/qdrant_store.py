from __future__ import annotations

"""
Qdrant persistence layer — write-only background store.

Purpose
-------
Every evaluated candidate is persisted to Qdrant so results survive server
restarts and can be inspected externally (dashboards, analytics, etc.).

Design constraints
------------------
* The evolution execution process is NOT changed: selection, LLM calls,
  fitness evaluation, and context building all work exactly as before.
* ``best_n()`` in MemoryStore still uses in-memory linear fitness ranking.
* No embeddings, no semantic search — Qdrant is purely a log/audit store.
* All writes are fire-and-forget (daemon threads) so the main loop is
  never blocked by network I/O.

Qdrant collection
-----------------
Name   : evolve_log
Vector : 1-dimensional float [fitness / 1000.0]  (placeholder — not used for search)
Payload: run_id, task, generation, fitness, strategy_tag, mutation_notes, code
"""

import logging
import threading
import warnings
from typing import Optional

from evolve.settings import settings

logger = logging.getLogger(__name__)

_COLLECTION = "evolve_log"
_VEC_DIM = 1          # dummy single-dim vector — Qdrant requires at least 1


class QdrantLogger:
    """
    Background Qdrant writer.

    * __init__   — launches a daemon thread to connect; returns immediately.
    * log()      — fire-and-forget upsert in a daemon thread.
    * No reads, no embeddings.
    """

    def __init__(self) -> None:
        self._client = None
        self._ok: bool = False
        self._lock = threading.Lock()

        if settings.qdrant_url:
            threading.Thread(target=self._connect_bg, daemon=True).start()

    # ── connection ─────────────────────────────────────────────────────────

    def _connect_bg(self) -> None:
        try:
            from qdrant_client import QdrantClient

            kw: dict = {"url": settings.qdrant_url, "timeout": 5.0}
            if settings.qdrant_api_key:
                kw["api_key"] = settings.qdrant_api_key

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*insecure connection.*")
                client = QdrantClient(**kw)

            self._ensure_collection(client)

            with self._lock:
                self._client = client
                self._ok = True
            logger.info("Qdrant logger connected at %s", settings.qdrant_url)
        except Exception as exc:
            logger.warning("Qdrant unavailable (candidates will not be persisted): %s", exc)

    def _ensure_collection(self, client) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in client.get_collections().collections}
        if _COLLECTION in existing:
            return
        try:
            client.create_collection(
                collection_name=_COLLECTION,
                vectors_config=VectorParams(size=_VEC_DIM, distance=Distance.DOT),
            )
            logger.info("Qdrant: created collection '%s'", _COLLECTION)
        except Exception as exc:
            if "already exists" in str(exc) or "409" in str(exc):
                pass  # concurrent creation — fine
            else:
                raise

    # ── write ──────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._ok

    def log(
        self,
        *,
        record_id: str,
        run_id: str,
        task: str,
        generation: int,
        fitness: float,
        strategy_tag: str,
        mutation_notes: str,
        code: str,
    ) -> None:
        """
        Persist a candidate to Qdrant in a background daemon thread.
        Returns immediately — never blocks the caller.
        """
        with self._lock:
            if not self._ok:
                return
            client = self._client

        def _write() -> None:
            from qdrant_client.models import PointStruct
            # Use normalised fitness as the dummy vector value.
            vec = [max(-1.0, min(1.0, fitness / 1000.0))]
            try:
                client.upsert(
                    collection_name=_COLLECTION,
                    points=[
                        PointStruct(
                            id=record_id,
                            vector=vec,
                            payload={
                                "run_id":        run_id,
                                "task":          task,
                                "generation":    generation,
                                "fitness":       fitness,
                                "strategy_tag":  strategy_tag,
                                "mutation_notes": mutation_notes,
                                "code":          code[:4000],   # cap to avoid huge payloads
                            },
                        )
                    ],
                )
            except Exception as exc:
                logger.debug("Qdrant write failed: %s", exc)

        threading.Thread(target=_write, daemon=True).start()


# Module-level singleton — shared across all MemoryStore instances.
qdrant_logger = QdrantLogger()
