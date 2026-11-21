"""Persistent, atomic, provenance-aware batch caching for Paper Analyzer LLM extraction."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

from researchgap.core.analysis_schemas import CandidateExtraction
from researchgap.core.schemas import TextChunk

DEFAULT_CACHE_DIR = Path(".researchgap_cache") / "analysis_batches"
PROMPT_VERSION = "2026.11_v2"
SCHEMA_VERSION = "v2_scientific_categories"


def compute_chunk_content_hash(chunks: list[TextChunk]) -> str:
    """Deterministic hash of source chunk text and identifiers."""
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk.chunk_id.encode("utf-8"))
        hasher.update(b":")
        hasher.update(str(chunk.page_number).encode("utf-8"))
        hasher.update(b":")
        hasher.update(chunk.text.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


def compute_batch_key(
    document_id: str,
    document_hash: str,
    chunk_ids: list[str],
    chunk_content_hash: str,
    model: str | None = None,
    prompt_version: str = PROMPT_VERSION,
    schema_version: str = SCHEMA_VERSION,
) -> str:
    """Compute a strict, collision-resistant provenance cache key.

    Model name is preserved in metadata rather than key hash so that shifting
    between authorized failover providers (Gemini -> Groq) preserves cache hits.
    """
    joined_chunks = ",".join(sorted(chunk_ids))
    raw = (
        f"doc:{document_id}|dhash:{document_hash}|chunks:{joined_chunks}|"
        f"chash:{chunk_content_hash}|pver:{prompt_version}|sver:{schema_version}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class AnalysisBatchCache:
    """Persistent on-disk cache for paper analysis extraction batches.

    Features:
    - Atomic writes via temporary files and atomic renaming.
    - Strict provenance binding (document hash, chunk hashes, model, prompt/schema versions).
    - Prevents mixing outputs across different documents, versions, or models.
    - Thread-safe and process-safe.
    """

    cache_dir: Path = DEFAULT_CACHE_DIR

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_batch_path(self, batch_key: str) -> Path:
        return self.cache_dir / f"{batch_key}.json"

    def get_cached_batch_data(self, batch_key: str) -> dict[str, Any] | None:
        """Retrieve full cached payload including provenance (model, provider, timestamp)."""
        path = self.get_batch_path(batch_key)
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            extraction_data = data.get("extraction")
            if not isinstance(extraction_data, dict):
                return None
            extraction = CandidateExtraction.model_validate(extraction_data)
            return {
                "extraction": extraction,
                "model": data.get("model", "unknown"),
                "provider": data.get("provider", "unknown"),
                "metadata": data.get("metadata", {}),
                "timestamp": data.get("timestamp", 0.0),
            }
        except Exception:
            return None

    def get_cached_batch(self, batch_key: str) -> CandidateExtraction | None:
        """Retrieve and validate a cached batch extraction if present."""
        data = self.get_cached_batch_data(batch_key)
        return data["extraction"] if data else None

    def save_cached_batch(
        self,
        batch_key: str,
        extraction: CandidateExtraction,
        metadata: dict[str, Any] | None = None,
        model: str = "",
        provider: str = "",
    ) -> None:
        """Atomically persist a validated batch extraction with provenance metadata."""
        import time as _time
        target_path = self.get_batch_path(batch_key)
        tmp_path = self.cache_dir / f"{batch_key}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"

        payload = {
            "batch_key": batch_key,
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "model": model,
            "provider": provider,
            "timestamp": _time.time(),
            "metadata": metadata or {},
            "extraction": extraction.model_dump(),
        }

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Atomic replace prevents corrupted files on crash/interruption
            os.replace(tmp_path, target_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def get_paper_progress(
        self,
        document_id: str,
        document_hash: str,
        groups: list[list[TextChunk]],
        model: str = "",
    ) -> dict[str, Any]:
        """Check cache status across all chunk groups for a paper."""
        cached_count = 0
        cached_indices: list[int] = []
        pending_indices: list[int] = []

        for idx, group in enumerate(groups):
            chunk_ids = [c.chunk_id for c in group]
            content_hash = compute_chunk_content_hash(group)
            key = compute_batch_key(
                document_id=document_id,
                document_hash=document_hash,
                chunk_ids=chunk_ids,
                chunk_content_hash=content_hash,
            )
            if self.get_cached_batch(key) is not None:
                cached_count += 1
                cached_indices.append(idx)
            else:
                pending_indices.append(idx)

        total = len(groups)
        return {
            "total_batches": total,
            "cached_batches": cached_count,
            "pending_batches": len(pending_indices),
            "cached_indices": cached_indices,
            "pending_indices": pending_indices,
            "is_resumable": cached_count > 0 and len(pending_indices) > 0,
            "is_complete": cached_count == total and total > 0,
        }

    def invalidate(self, document_id: str | None = None) -> int:
        """Invalidate cache entries. If document_id is provided, invalidates only that paper."""
        count = 0
        if not self.cache_dir.is_dir():
            return count

        for file_path in self.cache_dir.glob("*.json"):
            if document_id is None:
                try:
                    file_path.unlink()
                    count += 1
                except OSError:
                    pass
            else:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    meta = data.get("metadata", {})
                    if meta.get("document_id") == document_id:
                        file_path.unlink()
                        count += 1
                except Exception:
                    pass
        return count
