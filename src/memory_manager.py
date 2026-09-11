"""Memory helpers for the dissertation prototype.

Short-term session memory is the implemented and evaluated memory mechanism.
Long-term structured/semantic memory components are retained only as optional
future/experimental architecture demonstrations and are disabled by default in
the assessed workflow.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd


def update_short_term_memory(memory: Dict[str, Any], *, dataset: str, question: str = "", context: Optional[Dict[str, Any]] = None, answer_summary: str = "") -> Dict[str, Any]:
    """Update per-session memory used for follow-up questions."""
    memory.setdefault(dataset, {})
    ds_mem = memory[dataset]
    ds_mem["active_dataset"] = dataset
    if question:
        ds_mem["last_question"] = question
    if context:
        ds_mem["last_context"] = context
        for key in ["target_column", "group_col", "rank_col", "text_col", "filter_col", "domain", "topic"]:
            if key in context:
                ds_mem[key] = context[key]
    if answer_summary:
        ds_mem["last_answer_summary"] = answer_summary[:1000]
    return memory


def short_term_memory_table(memory: Dict[str, Any], dataset: Optional[str] = None) -> pd.DataFrame:
    if dataset:
        rows = [{"dataset": dataset, **(memory.get(dataset, {}) or {})}]
    else:
        rows = [{"dataset": k, **v} for k, v in memory.items()]
    return pd.DataFrame(rows)


def memory_architecture_table() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "memory_type": "Implemented short-term session memory",
            "implementation": "Streamlit/application session state",
            "status_in_dissertation": "IMPLEMENTED + EVALUATED",
            "stores": "Active dataset, selected target/model, last question/answer, last topic/column context, chat history",
            "purpose": "Supports coherent follow-up questions during the current analysis session.",
        },
        {
            "memory_type": "Future structured long-term memory",
            "implementation": "PostgreSQL design (SQLite audit helper may be used locally)",
            "status_in_dissertation": "FUTURE / OPTIONAL",
            "stores": "Dataset metadata, model runs, metrics, exports, users and audit records",
            "purpose": "Proposed scalable structured evidence store; not required for the assessed churn workflow.",
        },
        {
            "memory_type": "Future semantic long-term memory",
            "implementation": "ChromaDB / vector-store design with local JSONL experimental fallback",
            "status_in_dissertation": "FUTURE / OPTIONAL — disabled by default",
            "stores": "Report summaries, explanations, document chunks and prior evidence",
            "purpose": "Demonstrates a possible future retrieval-supported architecture; it is not claimed as the core evaluated memory mechanism.",
        },
        {
            "memory_type": "Future retrieval pipeline",
            "implementation": "Chunking + metadata filtering + hybrid search + reranking",
            "status_in_dissertation": "FUTURE / OPTIONAL",
            "stores": "N/A",
            "purpose": "Architecture demonstration for later enterprise extension, not part of the main churn experiment.",
        },
    ])
