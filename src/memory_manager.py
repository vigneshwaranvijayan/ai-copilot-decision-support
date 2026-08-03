"""Short-term and long-term memory helpers for the AI Copilot architecture."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
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
        {"memory_type": "Short-term session memory", "implementation": "Application session state", "stores": "Active dataset, selected target, last question, last answer, last topic/column context, chat history", "purpose": "Supports follow-up questions such as 'why?', 'show that as a graph', or 'same again' during one session."},
        {"memory_type": "Structured long-term memory", "implementation": "PostgreSQL in scalable design; SQLite fallback in prototype", "stores": "Dataset metadata, readiness gates, model runs, metrics, predictions, audit events", "purpose": "Keeps reproducible structured evidence and supports reporting/audit."},
        {"memory_type": "Semantic long-term memory", "implementation": "ChromaDB vector database; JSONL fallback if package unavailable", "stores": "Dataset summaries, readiness reports, Copilot answers, explanations, generated insights and report snippets", "purpose": "Retrieves relevant previous evidence before answering and justifies grounded Copilot behaviour."},
        {"memory_type": "Not used in revised core design", "implementation": "MongoDB removed", "stores": "N/A", "purpose": "Supervisor feedback indicated ChromaDB is a better fit for retrieval-based AI memory than MongoDB in this architecture."},
    ])
