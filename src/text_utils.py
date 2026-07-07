"""Controlled spelling/grammar tolerance for Copilot questions."""
from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Iterable, List, Tuple

COMMON_CORRECTIONS = {
    "chrun": "churn",
    "chrurn": "churn",
    "custmer": "customer",
    "costomer": "customer",
    "contrct": "contract",
    "montly": "monthly",
    "montlycharges": "monthly charges",
    "charg": "charge",
    "missng": "missing",
    "mising": "missing",
    "duplcate": "duplicate",
    "preformace": "performance",
    "preformance": "performance",
    "profmer": "performance",
    "perfomance": "performance",
    "modle": "model",
    "mdoel": "model",
    "predction": "prediction",
    "predcit": "predict",
    "expalin": "explain",
    "explan": "explain",
    "recomend": "recommend",
    "recomendation": "recommendation",
    "recommedation": "recommendation",
    "analaysis": "analysis",
    "analsis": "analysis",
    "anlayze": "analyse",
    "unemploye": "unemployed",
    "unemployee": "unemployed",
    "unemployed": "unemployed",
    "martial": "marital",
    "marrital": "marital",
    "marrial": "marital",
    "grap": "graph",
    "grahp": "graph",
    "chartt": "chart",
    "topdrvier": "top driver",
    "avarage": "average",
    "avg": "average",
    "revnue": "revenue",
    "quanity": "quantity",
    "countr": "country",
}


def normalise_question(text: str, known_terms: Iterable[str] | None = None) -> Tuple[str, List[str]]:
    """Correct common mistakes and fuzzy-match dataset column names/known terms.

    This is not an unrestricted LLM grammar editor. It only normalises user intent so
    the controlled Copilot can match dataset-grounded operations.
    """
    original = text or ""
    cleaned = original.strip().lower()
    cleaned = re.sub(r"[^a-z0-9_\s]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    corrections: List[str] = []

    words = cleaned.split()
    fixed_words = []
    known = sorted(set([str(t).lower() for t in known_terms or []]))
    known_tokens = set()
    for term in known:
        norm_term = term.replace("_", " ").strip()
        # Do not fuzzy-correct normal words into tiny columns such as `y`.
        # Exact matching for short columns is handled later by find_best_column.
        if len(norm_term) > 2:
            known_tokens.add(norm_term)
        for tok in norm_term.split():
            if len(tok) > 2:
                known_tokens.add(tok)

    for word in words:
        fixed = COMMON_CORRECTIONS.get(word, word)
        if fixed == word and known_tokens and len(word) > 2:
            match = get_close_matches(word, list(known_tokens), n=1, cutoff=0.86)
            if match:
                fixed = match[0]
        if fixed != word:
            corrections.append(f"{word} → {fixed}")
        fixed_words.extend(fixed.split())
    return " ".join(fixed_words), corrections


def _query_tokens(query: str) -> set[str]:
    return set((query or "").lower().replace("_", " ").split())


def _column_explicitly_in_query(query: str, column: str) -> bool:
    """Boundary-safe column detection.

    This prevents short column names such as `y` from matching inside normal
    words like `employee`, `type` or `why`.
    """
    q = (query or "").lower().replace("_", " ")
    q_space = f" {q} "
    col_space = str(column).lower().replace("_", " ").strip()
    if not col_space:
        return False
    if len(col_space) <= 2:
        return col_space in _query_tokens(q)
    if " " in col_space:
        return f" {col_space} " in q_space
    return col_space in _query_tokens(q)


def find_best_column(query: str, columns: Iterable[str], min_score: float = 0.68) -> str | None:
    """Find a likely column mentioned in a question using safe exact/fuzzy matching."""
    q = (query or "").lower().replace("_", " ")
    columns_list = list(columns)
    # Exact phrase/token match first, safely handling one-letter columns such as `y`.
    for col in sorted(columns_list, key=len, reverse=True):
        if _column_explicitly_in_query(q, col):
            return col
    # Token overlap, but skip very short columns unless explicitly mentioned.
    q_tokens = set(q.split())
    best_col = None
    best_score = 0.0
    for col in columns_list:
        col_norm = str(col).lower().replace("_", " ").strip()
        if len(col_norm) <= 2:
            continue
        tokens = set(col_norm.split())
        if not tokens:
            continue
        overlap = len(tokens & q_tokens) / len(tokens)
        if overlap > best_score:
            best_col = col
            best_score = overlap
    if best_col and best_score >= min_score:
        return best_col
    # difflib on full column name, again ignoring one/two letter columns unless explicit.
    long_columns = [c for c in columns_list if len(str(c).lower().replace("_", " ").strip()) > 2]
    candidates = [str(c).lower().replace("_", " ") for c in long_columns]
    match = get_close_matches(q, candidates, n=1, cutoff=0.6)
    if match:
        idx = candidates.index(match[0])
        return long_columns[idx]
    return None


def find_all_mentioned_columns(query: str, columns: Iterable[str]) -> List[str]:
    q = (query or "").lower().replace("_", " ")
    found = []
    for col in columns:
        if _column_explicitly_in_query(q, col):
            found.append(col)
    if not found:
        one = find_best_column(query, columns)
        if one:
            found.append(one)
    return found
