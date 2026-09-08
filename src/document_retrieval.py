"""Advanced document retrieval utilities for the Copilot memory layer.

This module implements a lightweight, fully local retrieval pipeline that can
be demonstrated in the Streamlit prototype without requiring external APIs:

1. document text extraction
2. overlapping chunking with metadata
3. metadata filtering
4. hybrid retrieval: TF-IDF semantic-style search + BM25 keyword search
5. lightweight reranking
6. retrieval metrics: precision@k, recall@k, MRR and nDCG

The code is intentionally transparent so it can be explained in the MSc
methodology and evaluated in Chapter 5.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import io
import math
import re
import uuid

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DocumentChunk:
    """One retrieval-ready document chunk with traceable metadata."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update({f"meta_{k}": v for k, v in self.metadata.items()})
        return data


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_bytes(filename: str, data: bytes) -> str:
    """Extract text from common document files.

    Supported formats are TXT, MD, CSV, JSON, PDF and DOCX. PDF/DOCX support is
    optional: if the relevant package is unavailable, a clear error is raised.
    """
    suffix = Path(filename.lower()).suffix
    if suffix in {".txt", ".md", ".csv", ".json"}:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return _clean_text(data.decode(encoding))
            except UnicodeDecodeError:
                continue
        return _clean_text(data.decode("utf-8", errors="ignore"))

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise ImportError("Install pypdf to extract PDF text: python -m pip install pypdf") from exc
        reader = PdfReader(io.BytesIO(data))
        pages: List[str] = []
        for idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[Page {idx}]\n{page_text}")
        return _clean_text("\n\n".join(pages))

    if suffix == ".docx":
        try:
            from docx import Document  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise ImportError("Install python-docx to extract DOCX text: python -m pip install python-docx") from exc
        doc = Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return _clean_text("\n\n".join(paragraphs))

    raise ValueError(f"Unsupported document type for retrieval: {suffix}")


def _words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def _content_tokens(text: str) -> List[str]:
    return [t for t in _words(text) if len(t) > 2 and t not in ENGLISH_STOP_WORDS]


def chunk_text(
    text: str,
    *,
    document_name: str,
    base_metadata: Optional[Dict[str, Any]] = None,
    chunk_words: int = 700,
    overlap_words: int = 120,
) -> List[DocumentChunk]:
    """Split text into overlapping chunks with traceable metadata.

    Chunks are created at paragraph level when possible, then joined to the
    requested size. The overlap prevents meaning being lost when an explanation
    spans two adjacent chunks.
    """
    if chunk_words <= 50:
        raise ValueError("chunk_words should be greater than 50")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    cleaned = _clean_text(text)
    if not cleaned:
        return []

    tokens = cleaned.split()
    chunks: List[DocumentChunk] = []
    start = 0
    chunk_no = 1
    metadata_base = dict(base_metadata or {})
    metadata_base.setdefault("document_name", document_name)
    metadata_base.setdefault("source_type", "uploaded_document")
    metadata_base.setdefault("chunking", "overlap")
    metadata_base.setdefault("chunk_words", chunk_words)
    metadata_base.setdefault("overlap_words", overlap_words)

    while start < len(tokens):
        end = min(start + chunk_words, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_value = " ".join(chunk_tokens).strip()
        if chunk_text_value:
            meta = dict(metadata_base)
            meta.update({
                "chunk_no": chunk_no,
                "word_start": start,
                "word_end": end,
                "word_count": len(chunk_tokens),
            })
            chunks.append(DocumentChunk(chunk_id=str(uuid.uuid4()), text=chunk_text_value, metadata=meta))
        if end == len(tokens):
            break
        start = max(0, end - overlap_words)
        chunk_no += 1
    return chunks


def chunks_to_dataframe(chunks: Sequence[DocumentChunk]) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame(columns=["chunk_id", "text", "metadata"])
    rows = []
    for c in chunks:
        row = {"chunk_id": c.chunk_id, "text": c.text, "preview": c.text[:220] + ("..." if len(c.text) > 220 else "")}
        row.update(c.metadata)
        rows.append(row)
    return pd.DataFrame(rows)


def _metadata_match(value: Any, expected: str) -> bool:
    if expected is None or str(expected).strip() == "":
        return True
    v = str(value or "").lower()
    e = str(expected).strip().lower()
    return e in v


def filter_chunks_by_metadata(chunks: Sequence[DocumentChunk], filters: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
    """Filter chunks by metadata before retrieval.

    The filter is inclusive/contains-based for demo friendliness. Empty filters
    are ignored.
    """
    if not filters:
        return list(chunks)
    active_filters = {k: v for k, v in filters.items() if v is not None and str(v).strip()}
    if not active_filters:
        return list(chunks)
    output = []
    for chunk in chunks:
        if all(_metadata_match(chunk.metadata.get(k), v) for k, v in active_filters.items()):
            output.append(chunk)
    return output


def _bm25_scores(query: str, docs: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Small BM25 implementation to avoid an additional dependency."""
    if not docs:
        return np.array([])
    tokenised = [_content_tokens(d) for d in docs]
    q_terms = _content_tokens(query)
    if not q_terms:
        return np.zeros(len(docs), dtype=float)
    avgdl = float(np.mean([len(t) for t in tokenised]) or 1.0)
    n_docs = len(docs)
    df: Dict[str, int] = {}
    for terms in tokenised:
        for term in set(terms):
            df[term] = df.get(term, 0) + 1
    scores = np.zeros(n_docs, dtype=float)
    for i, terms in enumerate(tokenised):
        if not terms:
            continue
        tf: Dict[str, int] = {}
        for t in terms:
            tf[t] = tf.get(t, 0) + 1
        dl = len(terms)
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            numerator = tf[term] * (k1 + 1)
            denominator = tf[term] + k1 * (1 - b + b * dl / avgdl)
            scores[i] += idf * numerator / max(denominator, 1e-9)
    return scores


def _tfidf_scores(query: str, docs: Sequence[str]) -> np.ndarray:
    if not docs:
        return np.array([])
    if not query.strip():
        return np.zeros(len(docs), dtype=float)
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(list(docs) + [query])
        return cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    except Exception:
        return np.zeros(len(docs), dtype=float)


def _normalise(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    if max_v - min_v < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - min_v) / (max_v - min_v)


def _rerank_score(query: str, text: str) -> float:
    """Transparent lightweight reranking score.

    This is not a cross-encoder model. It rewards exact query terms, phrase
    overlap and early occurrence. It is safe and reproducible for an MSc demo.
    """
    q_terms = set(_content_tokens(query))
    d_terms = _content_tokens(text)
    if not q_terms or not d_terms:
        return 0.0
    d_set = set(d_terms)
    coverage = len(q_terms & d_set) / len(q_terms)
    density = sum(1 for t in d_terms[:160] if t in q_terms) / max(min(len(d_terms), 160), 1)
    phrase_bonus = 0.0
    q_lower = query.lower().strip()
    t_lower = text.lower()
    if len(q_lower) > 8 and q_lower in t_lower:
        phrase_bonus = 0.25
    return float(coverage * 0.75 + density * 0.25 + phrase_bonus)


def hybrid_search(
    query: str,
    chunks: Sequence[DocumentChunk],
    *,
    metadata_filters: Optional[Dict[str, Any]] = None,
    candidate_limit: int = 100,
    top_k: int = 10,
    semantic_weight: float = 0.55,
    keyword_weight: float = 0.45,
) -> pd.DataFrame:
    """Retrieve chunks using metadata filtering, hybrid search and reranking."""
    filtered = filter_chunks_by_metadata(chunks, metadata_filters)
    if not filtered:
        return pd.DataFrame(columns=["rank", "chunk_id", "hybrid_score", "rerank_score", "final_score", "preview"])

    docs = [c.text for c in filtered]
    semantic = _normalise(_tfidf_scores(query, docs))
    keyword = _normalise(_bm25_scores(query, docs))
    hybrid = semantic_weight * semantic + keyword_weight * keyword

    candidate_order = np.argsort(-hybrid)[: max(1, min(candidate_limit, len(filtered)))]
    rows: List[Dict[str, Any]] = []
    for idx in candidate_order:
        c = filtered[int(idx)]
        rerank = _rerank_score(query, c.text)
        final = 0.65 * float(hybrid[int(idx)]) + 0.35 * rerank
        row = {
            "chunk_id": c.chunk_id,
            "semantic_score": round(float(semantic[int(idx)]), 4),
            "bm25_score": round(float(keyword[int(idx)]), 4),
            "hybrid_score": round(float(hybrid[int(idx)]), 4),
            "rerank_score": round(float(rerank), 4),
            "final_score": round(float(final), 4),
            "preview": c.text[:350] + ("..." if len(c.text) > 350 else ""),
        }
        row.update(c.metadata)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)
    if not out.empty:
        out.insert(0, "rank", range(1, len(out) + 1))
    return out


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 10) -> float:
    rel = set(relevant_ids)
    top = list(retrieved_ids)[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in rel) / len(top)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 10) -> float:
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    top = list(retrieved_ids)[:k]
    return sum(1 for item in top if item in rel) / len(rel)


def mean_reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    rel = set(relevant_ids)
    for idx, item in enumerate(retrieved_ids, start=1):
        if item in rel:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevance: Dict[str, float], k: int = 10) -> float:
    top = list(retrieved_ids)[:k]
    dcg = 0.0
    for idx, item in enumerate(top, start=1):
        rel = float(relevance.get(item, 0.0))
        dcg += (2 ** rel - 1) / math.log2(idx + 1)
    ideal_scores = sorted([float(v) for v in relevance.values()], reverse=True)[:k]
    idcg = 0.0
    for idx, rel in enumerate(ideal_scores, start=1):
        idcg += (2 ** rel - 1) / math.log2(idx + 1)
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_retrieval(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    *,
    relevance_scores: Optional[Dict[str, float]] = None,
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate one retrieval result list with standard IR metrics."""
    rel = set(relevant_ids)
    grades = relevance_scores if relevance_scores is not None else {rid: 1.0 for rid in rel}
    return {
        f"precision@{k}": round(precision_at_k(retrieved_ids, rel, k), 4),
        f"recall@{k}": round(recall_at_k(retrieved_ids, rel, k), 4),
        "mrr": round(mean_reciprocal_rank(retrieved_ids, rel), 4),
        f"ndcg@{k}": round(ndcg_at_k(retrieved_ids, grades, k), 4),
    }


def retrieval_pipeline_description() -> pd.DataFrame:
    """Return a dissertation-ready table explaining the advanced retrieval design."""
    rows = [
        {"Stage": "1. Overlapping chunking", "Purpose": "Split documents into manageable chunks while preserving context across boundaries."},
        {"Stage": "2. Metadata filtering", "Purpose": "Reduce the search space by company, department, document type, topic or access level."},
        {"Stage": "3. Hybrid retrieval", "Purpose": "Run TF-IDF semantic-style search and BM25 keyword search in parallel and combine scores."},
        {"Stage": "4. Reranking", "Purpose": "Reorder 50-100 candidate chunks and pass only the top 5-10 evidence chunks to the Copilot."},
        {"Stage": "5. Retrieval evaluation", "Purpose": "Measure precision, recall, MRR and nDCG using labelled relevant chunks."},
    ]
    return pd.DataFrame(rows)
