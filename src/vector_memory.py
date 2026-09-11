"""ChromaDB semantic memory with a safe local JSONL fallback.

ChromaDB is optional at runtime. If the package is unavailable, the prototype
still runs and stores memory records in a local JSONL file so tests and demos do
not fail. This module is retained as an optional future-architecture demonstration.
The assessed dissertation workflow uses short-term session memory by default.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time
import uuid

DEFAULT_MEMORY_DIR = Path(".copilot_memory")
DEFAULT_COLLECTION = "dataset_grounded_copilot_memory"


def _fallback_file(memory_dir: str | Path = DEFAULT_MEMORY_DIR) -> Path:
    path = Path(memory_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / "semantic_memory_fallback.jsonl"


def chromadb_available() -> bool:
    """Check optional ChromaDB availability without importing it.

    Importing ChromaDB just to render a status box can noticeably slow the
    first Streamlit page load.  The real package is imported lazily only when
    the user explicitly enables/uses semantic memory.
    """
    try:
        import importlib.util
        return importlib.util.find_spec("chromadb") is not None
    except Exception:
        return False


def memory_backend_status(memory_dir: str | Path = DEFAULT_MEMORY_DIR) -> Dict[str, Any]:
    fallback = _fallback_file(memory_dir)
    count = 0
    if fallback.exists():
        try:
            count = sum(1 for _ in fallback.open("r", encoding="utf-8"))
        except Exception:
            count = 0
    return {
        "backend": "ChromaDB" if chromadb_available() else "JSONL fallback",
        "chromadb_available": chromadb_available(),
        "memory_dir": str(Path(memory_dir)),
        "fallback_records": count,
        "purpose": "Optional future semantic-memory demonstration; disabled by default in the assessed workflow.",
    }


def add_memory_document(
    text: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    memory_dir: str | Path = DEFAULT_MEMORY_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> str:
    """Add one text record to semantic memory and return its record id."""
    record_id = str(uuid.uuid4())
    metadata = {k: (str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v) for k, v in (metadata or {}).items()}
    metadata["created_at"] = time.time()
    try:
        import chromadb  # type: ignore
        client = chromadb.PersistentClient(path=str(Path(memory_dir)))
        collection = client.get_or_create_collection(collection_name)
        collection.add(documents=[text], metadatas=[metadata], ids=[record_id])
        return record_id
    except Exception:
        # Fallback keeps the prototype deterministic even without optional ChromaDB.
        payload = {"id": record_id, "text": text, "metadata": metadata}
        with _fallback_file(memory_dir).open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        return record_id


def query_memory(
    query: str,
    *,
    dataset_name: str | None = None,
    n_results: int = 5,
    memory_dir: str | Path = DEFAULT_MEMORY_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> List[Dict[str, Any]]:
    """Retrieve relevant memory records. Uses ChromaDB if available, else keyword fallback."""
    try:
        import chromadb  # type: ignore
        client = chromadb.PersistentClient(path=str(Path(memory_dir)))
        collection = client.get_or_create_collection(collection_name)
        where = {"dataset": dataset_name} if dataset_name else None
        result = collection.query(query_texts=[query], n_results=n_results, where=where)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        ids = result.get("ids", [[]])[0]
        return [{"id": i, "text": d, "metadata": m or {}} for i, d, m in zip(ids, docs, metas)]
    except Exception:
        records: List[Dict[str, Any]] = []
        fpath = _fallback_file(memory_dir)
        if not fpath.exists():
            return []
        q_tokens = {t.lower() for t in query.split() if len(t) > 2}
        for line in fpath.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            meta = rec.get("metadata", {}) or {}
            if dataset_name and meta.get("dataset") != dataset_name:
                continue
            text = str(rec.get("text", ""))
            score = sum(1 for t in q_tokens if t in text.lower())
            records.append({"id": rec.get("id"), "text": text, "metadata": meta, "score": score})
        records.sort(key=lambda r: r.get("score", 0), reverse=True)
        return records[:n_results]
