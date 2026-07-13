"""Controlled, data-grounded Copilot response layer.

This module intentionally does not call an unrestricted LLM. It uses a
repeatable intent engine that maps user questions to safe, dataset-grounded
operations. If the intent is unclear, it asks for clarification instead of
inventing an answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from difflib import get_close_matches
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px

from .eda import (
    missing_value_table,
    simple_sentiment_summary,
    target_distribution,
    top_numeric_by_category,
)
from .text_utils import find_all_mentioned_columns, find_best_column, normalise_question
from .business_insights import business_insight_engine, detect_business_domain


@dataclass
class CopilotResponse:
    answer: str
    table: Optional[pd.DataFrame] = None
    chart: Optional[Any] = None
    interpreted_question: str = ""
    corrections: List[str] = field(default_factory=list)
    safety_warning: str = "The output is decision support only. A human reviewer should check the context before acting."
    context: Dict[str, Any] = field(default_factory=dict)


RECOMMENDATION_RULES = [
    ("contract", "Month-to-month or short commitment can indicate flexibility to leave. Consider annual-contract incentives or loyalty benefits."),
    ("monthlycharges", "High monthly charges may indicate price sensitivity. Review pricing, discount eligibility or retention offers."),
    ("monthly charges", "High monthly charges may indicate price sensitivity. Review pricing, discount eligibility or retention offers."),
    ("tenure", "Short tenure may mean the customer relationship is weak. Improve onboarding and early engagement."),
    ("support", "Support issues may indicate dissatisfaction. Prioritise support follow-up and service-quality review."),
    ("techsupport", "Lack of technical support may increase risk. Consider targeted support engagement."),
    ("internetservice", "Service type may relate to churn behaviour. Review package fit and service reliability."),
    ("payment", "Payment method may relate to risk or friction. Review billing experience and communication."),
    ("complaint", "Complaint-related drivers require case review before any action."),
    ("feedback", "Negative feedback themes should be reviewed with customer context before deciding on action."),
]

# Transparent business-action themes for feedback/comment/review columns.
# This keeps the Copilot controlled and reproducible: suggestions are generated
# from matched dataset terms, not from invented external knowledge.
BUSINESS_FEEDBACK_THEMES = [
    {
        "theme": "Pricing and monthly charges",
        "keywords": ["charge", "charges", "monthly", "price", "pricing", "cost", "expensive", "overpriced", "bill", "billing"],
        "business_meaning": "Customers may feel the service is too expensive or does not justify the monthly cost.",
        "recommended_action": "Review price-sensitive customers for retention discounts, loyalty benefits, package right-sizing, or clearer billing communication.",
        "safety_warning": "Do not assume price is the only cause; check customer history and profitability before offering discounts.",
    },
    {
        "theme": "Contract and cancellation risk",
        "keywords": ["contract", "month", "month-to-month", "cancel", "cancelled", "canceled", "churn", "leave", "leaving", "switch", "switched"],
        "business_meaning": "Customers may have low commitment or may already be considering cancellation/switching.",
        "recommended_action": "Prioritise proactive retention contact, explain plan options, and consider annual-contract incentives where appropriate.",
        "safety_warning": "Human review is required before retention actions because contract context and customer consent matter.",
    },
    {
        "theme": "Internet/service reliability",
        "keywords": ["internet", "service", "dsl", "fiber", "fibre", "connection", "speed", "slow", "unreliable", "disconnect", "downtime", "outage"],
        "business_meaning": "Feedback may point to reliability, speed, package-fit, or service-quality problems.",
        "recommended_action": "Check service-quality records, prioritise technical follow-up, and offer guidance on a better-fit package or troubleshooting route.",
        "safety_warning": "Avoid promising technical fixes without checking service records and operational feasibility.",
    },
    {
        "theme": "Support experience",
        "keywords": ["support", "help", "helpful", "rude", "agent", "call", "complaint", "complaints", "response", "delay", "delayed"],
        "business_meaning": "Customer support interactions may be affecting satisfaction and trust.",
        "recommended_action": "Escalate repeated support issues, review case notes, and improve follow-up communication for affected customers.",
        "safety_warning": "Support-related recommendations should be checked against actual ticket history before action.",
    },
    {
        "theme": "Early-life onboarding",
        "keywords": ["recently", "signed", "new", "first", "month", "months", "started", "joined", "onboarding"],
        "business_meaning": "New customers may not yet understand the service value or may face early setup problems.",
        "recommended_action": "Improve onboarding messages, early support check-ins, and usage guidance during the first months of the relationship.",
        "safety_warning": "Do not over-contact customers; use an appropriate communication policy and preference settings.",
    },
    {
        "theme": "Value and package fit",
        "keywords": ["value", "package", "plan", "offer", "benefit", "features", "reasonable", "satisfied", "happy", "loyal"],
        "business_meaning": "Customers may be comparing the value of their current plan against needs and alternatives.",
        "recommended_action": "Segment customers by plan/value perception and suggest package reviews, benefit reminders, or upgrade/downgrade guidance.",
        "safety_warning": "Recommendations should support human judgement and should not automatically change customer plans.",
    },
]

POSITIVE_WORDS = {
    "satisfied", "happy", "excellent", "good", "great", "reliable", "reasonable",
    "helpful", "fast", "loyal", "value", "smooth", "recommend", "positive",
    "pleased", "supportive", "stable", "affordable", "clear", "easy", "enjoy",
    "enjoyed", "impressed", "love", "liked", "convenient",
}

NEGATIVE_WORDS = {
    "bad", "poor", "worst", "terrible", "unhappy", "dissatisfied", "frustrated",
    "expensive", "overpriced", "issue", "issues", "problem", "problems", "slow",
    "unreliable", "cancel", "cancelled", "canceled", "churn", "leave", "leaving",
    "complaint", "complaints", "disappointed", "delay", "delayed", "rude", "fail",
    "failed", "broken", "weak", "difficult", "confusing", "regret", "switch",
    "switched", "dropped", "declined", "concern", "concerns", "not", "never",
    "lack", "lacking", "disconnect", "downtime", "unacceptable",
}

STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "from", "have", "been", "using",
    "customer", "customers", "provider", "service", "services", "company", "based",
    "profile", "write", "realistic", "feedback", "about", "into", "after", "before",
    "they", "them", "their", "there", "will", "would", "could", "should", "month",
    "months", "overall", "internet", "now", "had", "has", "was", "were", "are", "not",
    "you", "your", "what", "which", "show", "give", "gave", "tell", "type", "status",
}

AMBIGUOUS_SMALL_COLUMNS = {"y", "id", "no", "yes"}


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def _dataset_source_sentence(dataset_name: str) -> str:
    return f"Source used: active uploaded dataset `{dataset_name}`."


def _norm_text(value: Any) -> str:
    import re
    text = "" if value is None else str(value).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _compact(value: Any) -> str:
    return _norm_text(value).replace(" ", "")


def _query_tokens(q: str) -> set[str]:
    return set(_norm_text(q).split())


def _text_tokens(value: Any) -> List[str]:
    import re
    text = "" if pd.isna(value) else str(value).lower()
    return re.findall(r"[a-z][a-z']+", text)


def _is_explicit_column(q: str, column: Optional[str]) -> bool:
    if not column:
        return False
    q_norm = _norm_text(q)
    q_tokens = set(q_norm.split())
    col_norm = _norm_text(column)
    if not col_norm:
        return False
    if len(col_norm) <= 2:
        return col_norm in q_tokens
    if " " in col_norm:
        return f" {col_norm} " in f" {q_norm} "
    return col_norm in q_tokens or _compact(col_norm) in _compact(q_norm)


def _explicit_column_mentions(q: str, columns: Sequence[str]) -> List[str]:
    found: List[str] = []
    for col in sorted(columns, key=lambda c: len(str(c)), reverse=True):
        if _is_explicit_column(q, col):
            found.append(col)
    return found


def _safe_best_column(q: str, df: pd.DataFrame, *, allow_target: bool = True, target_column: Optional[str] = None) -> Optional[str]:
    columns = list(df.columns)
    explicit = _explicit_column_mentions(q, columns)
    if explicit:
        # Avoid choosing one-letter target columns unless typed explicitly.
        for col in explicit:
            if allow_target or col != target_column:
                return col
    candidates = columns
    if not allow_target and target_column in candidates:
        candidates = [c for c in candidates if c != target_column]
    # Do not fuzzy-match tiny column names such as y.
    candidates = [c for c in candidates if len(_norm_text(c)) > 2]
    return find_best_column(q, candidates)


def _is_short_followup(q: str) -> bool:
    tokens = _query_tokens(q)
    follow_terms = {"same", "again", "this", "that", "those", "them", "it", "more", "top", "bottom", "highest", "lowest", "worst", "best", "next"}
    return 0 < len(tokens) <= 5 and bool(tokens & follow_terms)


def _apply_followup_context(q: str, previous_context: Optional[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not previous_context or not _is_short_followup(q):
        return q
    # If user clearly typed any column, never append old context.
    if _explicit_column_mentions(q, columns):
        return q
    # Feedback/ranking terms are clear enough and should not inherit an old column.
    if any(term in _compact(q) for term in ["feedback", "comment", "review", "sentiment"]):
        return q
    for key in ["text_col", "rank_col", "group_col", "target_column"]:
        value = previous_context.get(key)
        if value and str(value).lower() not in q.lower():
            return f"{q} {value}"
    return q


def _maybe_clarify_columns(q: str, df: pd.DataFrame) -> Optional[CopilotResponse]:
    """Return a clarification response for overly vague column questions."""
    tokens = _query_tokens(q)
    if not tokens & {"status", "type", "category", "categories", "field", "column"}:
        return None
    # If an actual column is mentioned, no clarification needed.
    if _safe_best_column(q, df):
        return None
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])][:8]
    if cat_cols:
        table = pd.DataFrame({"available_categorical_columns": cat_cols})
        return CopilotResponse(
            "I need one dataset column to answer that safely. Choose one of the categorical columns below, for example `show education distribution` or `show marital status`.",
            table=table,
            interpreted_question=q,
            context={"topic": "clarification"},
        )
    return None


# ---------------------------------------------------------------------------
# Dataset/category/value helpers
# ---------------------------------------------------------------------------

def _find_id_like_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        norm = _compact(col)
        if norm in {"id", "customerid", "clientid", "accountid", "userid", "rowid", "customer"} or norm.endswith("id"):
            cols.append(col)
    return cols[:3]


def _categorical_columns(df: pd.DataFrame, *, max_unique: int = 120, include_bool: bool = True) -> List[str]:
    out = []
    for col in df.columns:
        s = df[col]
        if include_bool and pd.api.types.is_bool_dtype(s):
            out.append(col)
        elif not pd.api.types.is_numeric_dtype(s) and int(s.nunique(dropna=True)) <= max_unique:
            out.append(col)
    return out


def _long_text_columns(df: pd.DataFrame) -> List[str]:
    out = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        sample = df[col].dropna().astype(str).head(500)
        if sample.empty:
            continue
        avg_len = sample.str.len().mean()
        if avg_len >= 45 or any(k in _compact(col) for k in ["feedback", "comment", "review", "text", "prompt"]):
            out.append(col)
    return out


def _feedback_columns(df: pd.DataFrame) -> List[str]:
    text_cols = _long_text_columns(df)
    feedback = [c for c in text_cols if any(k in _compact(c) for k in ["feedback", "comment", "review", "sentimenttext"])]
    return feedback or text_cols


def _categorical_distribution_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].astype("string").fillna("Missing").value_counts(dropna=False).rename_axis(column).reset_index(name="count")
    total = int(counts["count"].sum())
    counts["percent"] = (counts["count"] / total * 100).round(2) if total else 0
    return counts


def _find_categorical_value_mentions(q: str, df: pd.DataFrame, exclude_cols: Optional[Sequence[str]] = None) -> List[Tuple[str, Any]]:
    """Find category values mentioned by the user, with small typo tolerance.

    Example: `unemploye marital status` -> (`job`, `unemployed`).
    """
    exclude = set(exclude_cols or [])
    q_norm = _norm_text(q)
    q_space = f" {q_norm} "
    q_tokens = q_norm.split()
    matches: List[Tuple[str, Any]] = []
    for col in _categorical_columns(df, max_unique=150):
        if col in exclude:
            continue
        # Avoid free-text columns and ID-like columns for value matching.
        if col in _long_text_columns(df) or col in _find_id_like_columns(df):
            continue
        values = [v for v in df[col].dropna().unique().tolist() if str(v).strip()]
        if len(values) > 150:
            continue
        value_texts = [_norm_text(v) for v in values]
        for raw, val in zip(value_texts, values):
            if not raw:
                continue
            if len(raw) <= 2:
                if raw in q_tokens:
                    matches.append((col, val))
                    break
            elif f" {raw} " in q_space:
                matches.append((col, val))
                break
        if any(m[0] == col for m in matches):
            continue
        # Fuzzy value matching per token/short phrase.
        for token in q_tokens:
            if len(token) < 4:
                continue
            close = get_close_matches(token, value_texts, n=1, cutoff=0.86)
            if close:
                idx = value_texts.index(close[0])
                matches.append((col, values[idx]))
                break
    # Prefer real filter columns over target y, if multiple.
    matches = sorted(matches, key=lambda x: (str(x[0]).lower() in AMBIGUOUS_SMALL_COLUMNS, str(x[0])))
    return matches


def _selected_row_columns(df: pd.DataFrame, sort_col: str, target_column: Optional[str] = None, limit: int = 10) -> List[str]:
    selected: List[str] = []

    def add(col: Optional[str]) -> None:
        if col and col in df.columns and col not in selected:
            selected.append(col)

    for col in _find_id_like_columns(df):
        add(col)
    add(sort_col)
    add(target_column)
    # Add important context columns without dumping all fields.
    priority_terms = ["name", "job", "marital", "education", "contract", "payment", "country", "product", "customer", "feedback"]
    for term in priority_terms:
        for col in df.columns:
            if col != sort_col and term in _compact(col):
                add(col)
            if len(selected) >= limit:
                return selected[:limit]
    for col in df.columns:
        add(col)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _ranked_rows(df: pd.DataFrame, sort_col: str, highest: bool, target_column: Optional[str] = None, n: int = 10) -> pd.DataFrame:
    clean = df[df[sort_col].notna()].copy()
    if clean.empty:
        return pd.DataFrame()
    table = clean.sort_values(sort_col, ascending=not highest).head(n)
    columns = _selected_row_columns(table, sort_col, target_column=target_column)
    table = table[columns].copy()
    table.insert(0, "row_index", table.index)
    return table.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feedback/sentiment helpers
# ---------------------------------------------------------------------------

def _score_text_rows(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    scored = df[[text_col]].copy()
    neg_counts: List[int] = []
    pos_counts: List[int] = []
    negative_terms: List[str] = []
    positive_terms: List[str] = []
    lengths: List[int] = []
    for value in scored[text_col].fillna(""):
        tokens = _text_tokens(value)
        neg = [t for t in tokens if t in NEGATIVE_WORDS]
        pos = [t for t in tokens if t in POSITIVE_WORDS]
        neg_counts.append(len(neg))
        pos_counts.append(len(pos))
        negative_terms.append(", ".join(dict.fromkeys(neg))[:180])
        positive_terms.append(", ".join(dict.fromkeys(pos))[:180])
        lengths.append(len(tokens))
    scored["negative_keyword_count"] = neg_counts
    scored["positive_keyword_count"] = pos_counts
    scored["sentiment_score"] = scored["negative_keyword_count"] - scored["positive_keyword_count"]
    scored["negative_terms_found"] = negative_terms
    scored["positive_terms_found"] = positive_terms
    scored["feedback_word_count"] = lengths
    return scored


def _truncate_text(value: Any, limit: int = 280) -> str:
    text = "" if pd.isna(value) else str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _feedback_rank_table(df: pd.DataFrame, text_col: str, target_column: Optional[str], worst: bool = True, n: int = 10) -> pd.DataFrame:
    scored = _score_text_rows(df, text_col)
    base = df.copy()
    for c in scored.columns:
        if c != text_col:
            base[c] = scored[c]
    base = base.sort_values(
        ["sentiment_score", "negative_keyword_count", "feedback_word_count"],
        ascending=[not worst, not worst, False],
    ).head(n)
    temp = base.copy()
    temp.insert(0, "row_index", temp.index)
    chosen: List[str] = ["row_index"]
    for col in _find_id_like_columns(df):
        if col not in chosen:
            chosen.append(col)
    if target_column and target_column in temp.columns and target_column not in chosen:
        chosen.append(target_column)
    for col in ["sentiment_score", "negative_keyword_count", "positive_keyword_count", "negative_terms_found", "positive_terms_found", text_col]:
        if col in temp.columns and col not in chosen:
            chosen.append(col)
    out = temp[chosen].copy().reset_index(drop=True)
    if text_col in out.columns:
        out[text_col] = out[text_col].map(_truncate_text)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def _top_text_words_clean(df: pd.DataFrame, text_col: str, n: int = 20) -> pd.DataFrame:
    from collections import Counter

    counter: Counter[str] = Counter()
    for value in df[text_col].dropna().astype(str):
        for token in _text_tokens(value):
            if token not in STOP_WORDS and len(token) > 2:
                counter[token] += 1
    return pd.DataFrame(counter.most_common(n), columns=["word", "count"])


def _feedback_business_action_intent(q: str) -> bool:
    tokens = _query_tokens(q)
    compact = _compact(q)
    feedback_terms = {"feedback", "comment", "comments", "review", "reviews", "sentiment"}
    action_terms = {
        "improve", "improvement", "improvements", "fix", "solve", "solution", "solutions",
        "recommend", "recommendation", "recommendations", "suggest", "suggestion",
        "suggestions", "action", "actions", "idea", "ideas", "strategy", "strategies",
        "business", "retention", "reduce", "increase", "better", "enhance", "handle",
    }
    return bool((tokens & feedback_terms or "customerfeedback" in compact) and (tokens & action_terms or compact.startswith("howtoimprove")))


def _feedback_theme_or_issue_intent(q: str) -> bool:
    """Detect questions asking for common feedback issues/themes.

    These should not be answered as raw value counts of full feedback text.
    Instead, they should use the transparent theme/action evidence table.
    """
    tokens = _query_tokens(q)
    compact = _compact(q)
    feedback_terms = {"feedback", "comment", "comments", "review", "reviews", "sentiment"}
    issue_terms = {
        "issue", "issues", "problem", "problems", "complaint", "complaints",
        "theme", "themes", "common", "most", "main", "major", "pain",
        "painpoint", "painpoints", "reason", "reasons", "cause", "causes",
        "topic", "topics", "concern", "concerns", "weakness", "weaknesses",
    }
    return bool((tokens & feedback_terms or "customerfeedback" in compact) and (tokens & issue_terms))


def _feedback_business_action_table(
    df: pd.DataFrame,
    text_col: str,
    target_column: Optional[str] = None,
    positive_label: Optional[Any] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    text_series = df[text_col].fillna("").astype(str)
    text_norm = text_series.map(_norm_text)
    neg_counts = _score_text_rows(df, text_col)["negative_keyword_count"]

    for rule in BUSINESS_FEEDBACK_THEMES:
        keywords = [_norm_text(k) for k in rule["keywords"]]
        mask = pd.Series(False, index=df.index)
        matched_terms: List[str] = []
        for kw in keywords:
            if not kw:
                continue
            pattern = r"\b" + re.escape(kw).replace(r"\ ", r"\s+") + r"\b"
            term_mask = text_norm.str.contains(pattern, regex=True, na=False)
            if bool(term_mask.any()):
                matched_terms.append(kw)
            mask = mask | term_mask

        evidence_rows = int(mask.sum())
        if evidence_rows == 0:
            continue

        negative_rows = int((mask & (neg_counts > 0)).sum())
        sample_text = ""
        sample_candidates = df.loc[mask & (neg_counts > 0), text_col]
        if sample_candidates.empty:
            sample_candidates = df.loc[mask, text_col]
        if not sample_candidates.empty:
            sample_text = _truncate_text(sample_candidates.astype(str).iloc[0], limit=180)

        target_rate = np.nan
        if target_column and target_column in df.columns and positive_label is not None:
            subset = df.loc[mask, target_column]
            if len(subset) > 0:
                target_rate = (subset.astype(str) == str(positive_label)).mean() * 100

        priority_score = negative_rows * 2 + evidence_rows
        if not np.isnan(target_rate):
            priority_score += float(target_rate) / 10
        if negative_rows >= 20 or (not np.isnan(target_rate) and target_rate >= 40):
            priority = "High"
        elif negative_rows >= 5 or evidence_rows >= 20:
            priority = "Medium"
        else:
            priority = "Low"

        rows.append({
            "priority": priority,
            "theme": rule["theme"],
            "evidence_rows": evidence_rows,
            "negative_evidence_rows": negative_rows,
            "target_positive_rate_percent": round(target_rate, 2) if not np.isnan(target_rate) else None,
            "matched_terms": ", ".join(dict.fromkeys(matched_terms[:10])),
            "business_meaning": rule["business_meaning"],
            "recommended_action": rule["recommended_action"],
            "human_review_warning": rule["safety_warning"],
            "example_feedback": sample_text,
            "priority_score": round(priority_score, 2),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "priority", "theme", "evidence_rows", "negative_evidence_rows",
            "target_positive_rate_percent", "matched_terms", "business_meaning",
            "recommended_action", "human_review_warning", "example_feedback", "priority_score"
        ])
    table = pd.DataFrame(rows).sort_values(
        ["priority_score", "negative_evidence_rows", "evidence_rows"],
        ascending=[False, False, False],
    ).drop(columns=["priority_score"])
    return table.reset_index(drop=True)


def _feedback_business_answer(table: pd.DataFrame, text_col: str, dataset_name: str) -> str:
    if table.empty:
        return (
            f"I could not find enough actionable feedback themes in `{text_col}`. "
            f"Review a sample manually or ask for `worst {text_col}` to inspect individual records. "
            f"{_dataset_source_sentence(dataset_name)}"
        )
    top = table.iloc[0]
    return (
        f"I found {len(table)} business-action themes in `{text_col}`. "
        f"Top priority is **{top['theme']}** based on {int(top['evidence_rows']):,} matching rows "
        f"and {int(top['negative_evidence_rows']):,} negative-evidence rows. "
        "The table gives practical improvement ideas, the evidence terms used, and human-review warnings. "
        f"{_dataset_source_sentence(dataset_name)}"
    )


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

def _top_or_bottom_intent(q: str) -> Optional[bool]:
    tokens = _query_tokens(q)
    high_terms = {"highest", "maximum", "max", "top", "largest", "biggest", "most"}
    low_terms = {"lowest", "minimum", "min", "smallest", "least"}
    if tokens & high_terms:
        return True
    if tokens & low_terms:
        return False
    return None


def _feedback_rank_intent(q: str) -> Optional[bool]:
    tokens = _query_tokens(q)
    compact = _compact(q)
    text_terms = {"feedback", "comment", "comments", "review", "reviews", "text", "sentiment"}
    bad_terms = {"worst", "bad", "negative", "unhappy", "dissatisfied", "complaint", "complaints", "poor", "problem", "problems"}
    good_terms = {"best", "good", "positive", "happy", "satisfied", "excellent"}
    if not (tokens & text_terms or "customerfeedback" in compact):
        return None
    if tokens & bad_terms:
        return True
    if tokens & good_terms:
        return False
    # `who/which customer gave feedback` is row-level. Plain `show feedback themes`
    # should go to the theme/word-summary intent, not worst-row ranking.
    if tokens & {"who", "which", "customers", "customer", "rows", "gave", "give"}:
        return True
    return None


def _is_target_intent(q: str, target_column: Optional[str]) -> bool:
    if not target_column:
        return False
    tokens = _query_tokens(q)
    if _is_explicit_column(q, target_column):
        return True
    # Business aliases. Avoid generic terms such as "distribution" because they
    # can mean a normal categorical column like education or marital.
    aliases = {"target", "positive", "class", "classes", "churn", "attrition", "exited", "retained", "subscribed", "subscription", "conversion", "response"}
    return bool(tokens & aliases) and bool(tokens & {"rate", "distribution", "count", "counts", "show", "by", "positive", "class", "classes", "target"})


def _model_summary(model_output: Any) -> str:
    if model_output is None or getattr(model_output, "best_result", None) is None:
        return "No trained model is available for the active dataset. Select a binary target and train the models first."
    best = model_output.best_result
    metrics = best.metrics
    return (
        f"Best model: **{best.model_name}**. "
        f"F1={metrics.get('f1', float('nan')):.3f}, "
        f"Recall={metrics.get('recall', float('nan')):.3f}, "
        f"Precision={metrics.get('precision', float('nan')):.3f}, "
        f"ROC-AUC={metrics.get('roc_auc', float('nan')):.3f}. "
        "The model was trained only on the selected uploaded dataset."
    )


def _recommend_from_drivers(explanation_output: Any) -> str:
    if explanation_output is None or getattr(explanation_output, "local_importance", None) is None:
        return "Recommendation rules need a trained model and explanation output first."
    local = explanation_output.local_importance.head(8)
    suggestions: List[str] = []
    for feature in local.get("feature", []):
        feature_norm = str(feature).lower().replace("_", " ")
        for token, suggestion in RECOMMENDATION_RULES:
            if token in feature_norm and suggestion not in suggestions:
                suggestions.append(suggestion)
                break
    if not suggestions:
        suggestions.append("Review the top model drivers and compare them with customer/business context before choosing an action.")
    return "Possible action guidance: " + " ".join(f"{idx + 1}. {s}" for idx, s in enumerate(suggestions[:4]))


def _suggestion_table(df: pd.DataFrame, target_column: Optional[str]) -> pd.DataFrame:
    domain = detect_business_domain(df)
    suggestions = [
        "show dataset overview",
        "show missing values",
        "show duplicate rows",
        "show model performance",
        "show top drivers",
        "give business suggestions",
    ]
    if domain == "sales_retail":
        suggestions += ["which products should we purchase more", "which products are declining", "sales improvement ideas"]
    elif domain == "employee_hr":
        suggestions += ["employee support and retention suggestions", "who may need promotion review", "which groups have attrition risk"]
    elif domain in {"customer_feedback", "customer_churn"}:
        suggestions += ["how to improve customer feedback", "customer retention suggestions"]
    elif domain == "bank_marketing":
        suggestions += ["marketing campaign suggestions", "which customer segments should we target"]
    elif domain == "operations_service":
        suggestions += ["operations improvement suggestions", "which service area needs attention"]
    cats = _categorical_columns(df)[:3]
    nums = df.select_dtypes(include=np.number).columns.tolist()[:3]
    feedback = _feedback_columns(df)[:1]
    if target_column and cats:
        suggestions.append(f"show {target_column} by {cats[0]}")
    if nums:
        suggestions.append(f"who has highest {nums[0]}")
    if cats:
        suggestions.append(f"show {cats[0]} distribution")
    if feedback:
        suggestions.append(f"who gave worst {feedback[0]}")
    return pd.DataFrame({"suggested_question": suggestions})


# ---------------------------------------------------------------------------
# Main answer engine
# ---------------------------------------------------------------------------

def _answer_question_core(
    question: str,
    df: pd.DataFrame,
    dataset_name: str = "active_dataset",
    target_column: Optional[str] = None,
    positive_label: Optional[Any] = None,
    model_output: Optional[Any] = None,
    explanation_output: Optional[Any] = None,
    previous_context: Optional[Dict[str, Any]] = None,
) -> CopilotResponse:
    """Return a controlled response using only the active dataset and artefacts."""
    if df is None or df.empty:
        return CopilotResponse("No active dataset is available. Upload or select a dataset first.")

    safe_known_terms = [str(c) for c in df.columns if len(_norm_text(c)) > 2]
    safe_known_terms += [
        "missing", "duplicate", "overview", "summary", "model", "performance", "churn", "target",
        "driver", "drivers", "recommendation", "correlation", "highest", "lowest", "worst", "best",
        "unemployed", "marital", "education", "feedback", "balance", "average", "distribution",
        "business", "insight", "ideas", "improve", "improvement", "strategy", "issue", "issues", "theme", "themes", "pain", "painpoint", "painpoints", "sales", "product",
        "stock", "purchase", "forecast", "employee", "salary", "promotion", "resign", "attrition",
        "campaign", "marketing", "operations", "complaint", "ticket",
    ]
    q, corrections = normalise_question(question, safe_known_terms)
    q = _apply_followup_context(q, previous_context, list(df.columns))
    source = _dataset_source_sentence(dataset_name)
    tokens = _query_tokens(q)

    clarification = _maybe_clarify_columns(q, df)
    if clarification is not None:
        clarification.corrections = corrections
        clarification.interpreted_question = q
        return clarification

    # 1) Dataset overview / quality checks.
    if tokens & {"overview", "summary", "describe", "shape"} or "how many rows" in q or "how many columns" in q:
        table = pd.DataFrame([
            {"metric": "rows", "value": df.shape[0]},
            {"metric": "columns", "value": df.shape[1]},
            {"metric": "missing_cells", "value": int(df.isna().sum().sum())},
            {"metric": "duplicate_rows", "value": int(df.duplicated().sum())},
            {"metric": "numeric_columns", "value": df.select_dtypes(include=np.number).shape[1]},
            {"metric": "categorical_or_text_columns", "value": len([c for c in df.columns if c not in df.select_dtypes(include=np.number).columns])},
        ])
        return CopilotResponse(f"Here is a dataset overview. {source}", table=table, interpreted_question=q, corrections=corrections, context={"topic": "overview"})

    if tokens & {"missing", "null", "empty", "blank"}:
        table = missing_value_table(df).head(40)
        answer = f"The active dataset contains {int(df.isna().sum().sum()):,} missing cells. {source}"
        chart_data = table[table["missing_count"] > 0].head(15)
        chart = px.bar(chart_data, x="column", y="missing_percent", title="Missing values (%)") if not chart_data.empty else None
        return CopilotResponse(answer, table=table, chart=chart, interpreted_question=q, corrections=corrections, context={"topic": "missing"})

    if tokens & {"duplicate", "duplicates"}:
        count = int(df.duplicated().sum())
        return CopilotResponse(f"The active dataset has {count:,} duplicate rows after current cleaning. {source}", interpreted_question=q, corrections=corrections, context={"topic": "duplicates"})

    # 2) Model/explanation intents.
    if tokens & {"model", "performance", "accuracy", "f1", "recall", "precision", "auc"}:
        if model_output is not None and getattr(model_output, "leaderboard", None) is not None:
            return CopilotResponse(_model_summary(model_output) + " " + source, table=model_output.leaderboard, interpreted_question=q, corrections=corrections, context={"topic": "model", "target_column": getattr(model_output, "target_column", target_column)})
        return CopilotResponse(_model_summary(model_output) + " " + source, interpreted_question=q, corrections=corrections, context={"topic": "model"})

    if tokens & {"driver", "drivers", "importance", "features", "feature", "shap"}:
        if explanation_output is not None and getattr(explanation_output, "global_importance", None) is not None:
            table = explanation_output.global_importance.head(20)
            chart = px.bar(table.head(15), x="importance", y="feature", orientation="h", title="Top model drivers") if not table.empty and "importance" in table.columns else None
            return CopilotResponse(f"Here are the strongest model drivers for the active trained model. {source}", table=table, chart=chart, interpreted_question=q, corrections=corrections, context={"topic": "drivers", "target_column": target_column})
        return CopilotResponse("Train a model and generate an explanation first, then I can show SHAP/fallback drivers. " + source, interpreted_question=q, corrections=corrections)

    # General domain-aware business insight questions. This is intentionally
    # handled before generic single-column summaries so questions such as
    # "what should we purchase next month?" or "employee improvement ideas"
    # produce business suggestions rather than raw frequency tables.
    business_terms = {
        "business", "insight", "insights", "idea", "ideas", "strategy", "strategies",
        "improve", "improvement", "improvements", "suggest", "suggestion", "suggestions",
        "recommend", "recommendation", "actions", "action", "purchase", "stock",
        "buy", "reorder", "forecast", "next", "month", "declining", "decline",
        "sales", "product", "employee", "salary", "promotion", "promote", "resign",
        "retention", "campaign", "target", "market", "operations", "complaint",
    }
    # Feedback has its own more specific transparent theme engine below, so let
    # explicit feedback questions pass to that first. Other domains go to the
    # business insight engine.
    if (tokens & business_terms) and not _feedback_business_action_intent(q):
        domain = detect_business_domain(df, q)
        explicit_domain_terms = {
            "sales", "sale", "product", "stock", "purchase", "reorder", "forecast",
            "employee", "salary", "promotion", "promote", "resign", "attrition", "overtime",
            "marketing", "campaign", "target", "customer", "subscription", "subscribe",
            "operation", "operations", "complaint", "ticket", "service", "feedback",
        }
        generic_business_request = bool(tokens & {"business", "insight", "insights", "suggestion", "suggestions", "idea", "ideas", "improve", "improvement"})
        # Do not answer broad future questions such as "best strategy for tomorrow"
        # unless the user gives a business domain or asks for business suggestions.
        if (domain != "general" and (tokens & explicit_domain_terms or generic_business_request)) or generic_business_request:
            result = business_insight_engine(df, q, dataset_name=dataset_name, target_column=target_column, positive_label=positive_label)
            return CopilotResponse(
                result.answer,
                table=result.table,
                chart=result.chart,
                interpreted_question=q,
                corrections=corrections,
                safety_warning=result.warning,
                context=result.context or {"topic": "business_insights", "domain": result.domain},
            )

    # Feedback improvement/business-action questions must be handled before
    # generic model recommendation. Otherwise `how to improve customerfeedback`
    # can fall through to a frequency table or model-only rule.
    if _feedback_business_action_intent(q):
        explicit_cols = _explicit_column_mentions(q, list(df.columns))
        feedback_cols = _feedback_columns(df)
        text_col = next((c for c in explicit_cols if c in feedback_cols), None) or (feedback_cols[0] if feedback_cols else None)
        if not text_col:
            return CopilotResponse("I could not find a feedback/comment/review text column in the active dataset. " + source, interpreted_question=q, corrections=corrections)
        table = _feedback_business_action_table(df, text_col, target_column=target_column, positive_label=positive_label)
        chart = None
        if not table.empty:
            chart = px.bar(table.head(8), x="theme", y="negative_evidence_rows", hover_data=["evidence_rows", "priority"], title=f"Feedback improvement priorities from {text_col}")
        return CopilotResponse(
            _feedback_business_answer(table, text_col, dataset_name),
            table=table,
            chart=chart,
            interpreted_question=q,
            corrections=corrections,
            safety_warning="These are decision-support suggestions generated from transparent keyword/theme rules. A human reviewer must check customer context before acting.",
            context={"topic": "feedback_business_actions", "text_col": text_col, "target_column": target_column},
        )

    # Common feedback issue/theme questions should use transparent theme
    # evidence, not raw full-text value counts.
    if _feedback_theme_or_issue_intent(q):
        explicit_cols = _explicit_column_mentions(q, list(df.columns))
        feedback_cols = _feedback_columns(df)
        text_col = next((c for c in explicit_cols if c in feedback_cols), None) or (feedback_cols[0] if feedback_cols else None)
        if not text_col:
            return CopilotResponse("I could not find a feedback/comment/review text column in the active dataset. " + source, interpreted_question=q, corrections=corrections)
        table = _feedback_business_action_table(df, text_col, target_column=target_column, positive_label=positive_label)
        if table.empty:
            words = _top_text_words_clean(df, text_col, n=20)
            chart = px.bar(words, x="word", y="count", title=f"Top issue words in {text_col}") if not words.empty else None
            return CopilotResponse(
                f"I could not match the feedback to predefined business themes, so I returned cleaned top words from `{text_col}` instead. {source}",
                table=words,
                chart=chart,
                interpreted_question=q,
                corrections=corrections,
                safety_warning="This is a text-summary fallback. A human reviewer should inspect example records before deciding actions.",
                context={"topic": "feedback_themes", "text_col": text_col, "target_column": target_column},
            )
        chart = px.bar(
            table.head(8),
            x="theme",
            y="evidence_rows",
            hover_data=["negative_evidence_rows", "priority"],
            title=f"Most common feedback issues/themes from {text_col}",
        )
        top = table.iloc[0]
        answer = (
            f"The most common actionable feedback issue/theme is **{top['theme']}**, "
            f"with {int(top['evidence_rows']):,} matching row(s) and "
            f"{int(top['negative_evidence_rows']):,} negative-evidence row(s). "
            f"Recommended action: {top['recommended_action']} {source}"
        )
        return CopilotResponse(
            answer,
            table=table,
            chart=chart,
            interpreted_question=q,
            corrections=corrections,
            safety_warning="These themes use transparent keyword rules from uploaded feedback. Review sample records before taking customer-facing actions.",
            context={"topic": "feedback_themes", "text_col": text_col, "target_column": target_column},
        )

    if tokens & {"recommend", "recommendation", "action", "actions", "suggest", "suggestion"}:
        if explanation_output is not None and getattr(explanation_output, "local_importance", None) is not None:
            return CopilotResponse(_recommend_from_drivers(explanation_output) + " " + source, interpreted_question=q, corrections=corrections, context={"topic": "recommendation", "target_column": target_column})
        result = business_insight_engine(df, q, dataset_name=dataset_name, target_column=target_column, positive_label=positive_label)
        return CopilotResponse(result.answer, table=result.table, chart=result.chart, interpreted_question=q, corrections=corrections, safety_warning=result.warning, context=result.context or {"topic": "business_insights", "domain": result.domain})

    # 3) Feedback/text row-level ranking and themes.
    feedback_rank = _feedback_rank_intent(q)
    if feedback_rank is not None:
        explicit_cols = _explicit_column_mentions(q, list(df.columns))
        feedback_cols = _feedback_columns(df)
        text_col = next((c for c in explicit_cols if c in feedback_cols), None) or (feedback_cols[0] if feedback_cols else None)
        if not text_col:
            return CopilotResponse("I could not find a feedback/comment/review text column in the active dataset. " + source, interpreted_question=q, corrections=corrections)
        table = _feedback_rank_table(df, text_col, target_column, worst=feedback_rank, n=10)
        direction = "most negative / worst" if feedback_rank else "most positive / best"
        if table.empty:
            return CopilotResponse(f"I found `{text_col}`, but there are no text values to rank. {source}", interpreted_question=q, corrections=corrections, context={"text_col": text_col})
        answer = (
            f"These are the records with the {direction} `{text_col}` based on a simple transparent keyword score. "
            f"I show row index and customer/account ID when available because the dataset may not contain a real person name. {source}"
        )
        chart = px.bar(table, x="rank", y="sentiment_score", hover_data=[c for c in [target_column, "negative_terms_found"] if c and c in table.columns], title=f"Ranked {text_col} sentiment score") if "sentiment_score" in table.columns else None
        return CopilotResponse(answer, table=table, chart=chart, interpreted_question=q, corrections=corrections, context={"text_col": text_col, "target_column": target_column})

    if tokens & {"feedback", "comment", "comments", "review", "reviews", "theme", "themes", "word", "words", "sentiment", "text"}:
        explicit_cols = _explicit_column_mentions(q, list(df.columns))
        text_cols = _feedback_columns(df)
        text_col = next((c for c in explicit_cols if c in text_cols), None) or (text_cols[0] if text_cols else None)
        if text_col:
            table = _top_text_words_clean(df, text_col)
            sent = simple_sentiment_summary(df, text_col)
            chart = px.bar(table, x="word", y="count", title=f"Top non-generic words in {text_col}") if not table.empty else None
            answer = (
                f"I analysed text-like values in `{text_col}`. Rows with positive keywords: {sent['positive_keyword_rows']}; "
                f"rows with negative keywords: {sent['negative_keyword_rows']}. {source}"
            )
            return CopilotResponse(answer, table=table, chart=chart, interpreted_question=q, corrections=corrections, context={"text_col": text_col, "target_column": target_column})

    # 4) Filtered category distribution: `unemployed marital status`, `admin education`.
    requested_col = _safe_best_column(q, df, allow_target=False, target_column=target_column)
    value_matches = _find_categorical_value_mentions(q, df, exclude_cols=[requested_col] if requested_col else [])
    if requested_col and requested_col in df.columns and not pd.api.types.is_numeric_dtype(df[requested_col]) and value_matches:
        filter_col, filter_value = value_matches[0]
        filtered = df[df[filter_col].astype("string").str.lower() == str(filter_value).lower()].copy()
        if filtered.empty:
            return CopilotResponse(f"I matched `{filter_col} = {filter_value}`, but no rows were found after filtering. {source}", interpreted_question=q, corrections=corrections)
        table = _categorical_distribution_table(filtered, requested_col).head(30)
        chart = px.bar(table, x=requested_col, y="count", hover_data=["percent"], title=f"{requested_col} distribution where {filter_col} = {filter_value}")
        answer = f"For records where `{filter_col}` is `{filter_value}`, here is the distribution of `{requested_col}`. Matched {len(filtered):,} rows. {source}"
        return CopilotResponse(answer, table=table, chart=chart, interpreted_question=q, corrections=corrections, context={"group_col": requested_col, "filter_col": filter_col, "filter_value": str(filter_value), "target_column": target_column})

    # Count a matched categorical value if no requested output column exists.
    if value_matches and (tokens & {"how", "many", "count", "number"} or len(tokens) <= 4):
        filter_col, filter_value = value_matches[0]
        mask = df[filter_col].astype("string").str.lower() == str(filter_value).lower()
        count = int(mask.sum())
        pct = count / len(df) * 100 if len(df) else 0
        table = pd.DataFrame([{"filter_column": filter_col, "value": filter_value, "count": count, "percent": round(pct, 2)}])
        return CopilotResponse(f"There are {count:,} rows where `{filter_col}` is `{filter_value}` ({pct:.2f}% of the active dataset). {source}", table=table, interpreted_question=q, corrections=corrections, context={"filter_col": filter_col, "filter_value": str(filter_value)})

    # 5) Group comparisons.
    mentioned_cols = find_all_mentioned_columns(q, df.columns)
    explicit_cols = _explicit_column_mentions(q, list(df.columns))
    group_words = tokens & {"by", "per", "group", "groups", "compare", "across", "versus", "vs"}
    if group_words or len(mentioned_cols) >= 2:
        cols = explicit_cols or mentioned_cols
        # Target-rate by category: `show y by job`, `churn by contract`.
        group_cols = [c for c in cols if c != target_column and c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
        numeric_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if target_column and (_is_target_intent(q, target_column) or target_column in cols) and group_cols:
            group_col = group_cols[0]
            temp = df[[target_column, group_col]].copy()
            pos = positive_label if positive_label is not None else temp[target_column].dropna().value_counts().index[0]
            temp["positive_rate"] = (temp[target_column].astype(str) == str(pos)).astype(int)
            table = temp.groupby(group_col, dropna=False)["positive_rate"].agg(["mean", "count"]).reset_index()
            table["positive_rate_percent"] = (table["mean"] * 100).round(2)
            table = table.sort_values("positive_rate_percent", ascending=False).head(30)
            chart = px.bar(table, x=group_col, y="positive_rate_percent", hover_data=["count"], title=f"{target_column} positive rate by {group_col}")
            return CopilotResponse(f"Here is `{target_column}` positive rate by `{group_col}`. {source}", table=table, chart=chart, interpreted_question=q, corrections=corrections, context={"group_col": group_col, "target_column": target_column})
        if group_cols and numeric_cols:
            group_col, num_col = group_cols[0], numeric_cols[0]
            agg = "mean" if tokens & {"average", "avg", "mean"} else "sum" if tokens & {"total", "sum"} else "mean"
            table = top_numeric_by_category(df, group_col, num_col, agg=agg).head(30)
            chart = px.bar(table, x=group_col, y=table.columns[-1], title=f"{agg.title()} {num_col} by {group_col}") if not table.empty else None
            return CopilotResponse(f"Here is `{agg}` of `{num_col}` by `{group_col}`. {source}", table=table, chart=chart, interpreted_question=q, corrections=corrections, context={"group_col": group_col, "rank_col": num_col})

    # 6) Target distribution.
    if target_column and _is_target_intent(q, target_column):
        dist = target_distribution(df, target_column)
        chart = px.bar(dist, x=target_column, y="count", title=f"Distribution of {target_column}") if not dist.empty else None
        if positive_label is not None and target_column in df.columns:
            pos_rate = (df[target_column].astype(str) == str(positive_label)).mean() * 100
            answer = f"The positive rate for `{target_column}` is {pos_rate:.2f}%. {source}"
        else:
            answer = f"Here is the target distribution for `{target_column}`. {source}"
        return CopilotResponse(answer, table=dist, chart=chart, interpreted_question=q, corrections=corrections, context={"target_column": target_column})

    # 7) Top/bottom numeric rows.
    rank_highest = _top_or_bottom_intent(q)
    rank_col = _safe_best_column(q, df, allow_target=False, target_column=target_column)
    if rank_highest is not None and rank_col and rank_col in df.columns and pd.api.types.is_numeric_dtype(df[rank_col]):
        table = _ranked_rows(df, rank_col, highest=rank_highest, target_column=target_column, n=10)
        direction = "highest" if rank_highest else "lowest"
        answer = f"These are the records with the {direction} values for `{rank_col}`. I show dataset rows because the active data may not contain a person/customer name. {source}"
        chart = px.bar(table, x="row_index", y=rank_col, title=f"Top records by {rank_col}") if not table.empty and rank_col in table.columns else None
        return CopilotResponse(answer, table=table, chart=chart, interpreted_question=q, corrections=corrections, context={"rank_col": rank_col, "target_column": target_column})

    # 8) Single column distribution/profile.
    col = _safe_best_column(q, df, allow_target=False, target_column=target_column)
    if col and col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe().rename("value").reset_index().rename(columns={"index": "statistic"})
            chart = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")
            return CopilotResponse(f"Here is the numeric profile for `{col}`. {source}", table=desc, chart=chart, interpreted_question=q, corrections=corrections, context={"rank_col": col})
        counts = _categorical_distribution_table(df, col).head(30)
        chart = px.bar(counts, x=col, y="count", hover_data=["percent"], title=f"Top values in {col}")
        return CopilotResponse(f"Here are the most frequent values for `{col}`. {source}", table=counts, chart=chart, interpreted_question=q, corrections=corrections, context={"group_col": col, "target_column": target_column})

    # 9) Safe fallback with dynamic suggested questions.
    table = _suggestion_table(df, target_column)
    return CopilotResponse(
        "I could not safely map that question to a supported data-grounded operation, so I did not guess. Try one of the suggested questions below. " + source,
        table=table,
        interpreted_question=q,
        corrections=corrections,
        context={"topic": "fallback"},
    )


# ---------------------------------------------------------------------------
# Presentation wrapper: every Copilot answer follows the same academic frame
# ---------------------------------------------------------------------------

def _business_use_hint(context: Dict[str, Any], question: str) -> str:
    topic = str(context.get("topic", "")).lower() if context else ""
    domain = str(context.get("domain", "")).lower() if context else ""
    q = _norm_text(question)
    if "missing" in topic:
        return "Use this to prioritise data-quality fixes before modelling or reporting. Columns with high missingness may need collection changes, imputation or exclusion."
    if "model" in topic:
        return "Use the leaderboard to choose the most balanced decision-support model. F1 and ROC-AUC are useful for comparing models, but weak metrics should be reported as limitations."
    if "drivers" in topic:
        return "Use the strongest drivers to explain why the model behaves as it does and to identify areas that may need operational attention."
    if "feedback" in topic:
        return "Use the ranked feedback/themes to decide which customer pain points need service, pricing, support or communication improvements."
    if "business" in topic or domain:
        return "Use these suggestions as an action shortlist. Prioritise items with stronger dataset evidence, then validate with a manager or domain expert."
    if "rank" in context or "rank_col" in context:
        return "Use the top/bottom records to review unusual cases, outliers or priority customers/items, not to make automatic decisions."
    if "group_col" in context:
        return "Use the group comparison to identify stronger and weaker segments, then investigate causes before taking action."
    if any(t in q for t in ["improve", "suggest", "action", "business", "purchase", "stock", "employee", "customer"]):
        return "Use this as decision-support evidence only. The system does not use external market, HR or customer context unless you upload it."
    return "Use this result as evidence for analysis, reporting or follow-up questions. Ask for business suggestions if you want recommended actions from this dataset."


def _frame_copilot_answer(response: CopilotResponse, *, question: str, dataset_name: str, df: pd.DataFrame) -> CopilotResponse:
    """Make every answer clear, evidence-based and dissertation-friendly."""
    if response.answer.strip().startswith("### Copilot answer"):
        return response
    context = response.context or {}
    cols_used = []
    for key in ["target_column", "group_col", "rank_col", "text_col", "filter_col"]:
        value = context.get(key)
        if value and value not in cols_used:
            cols_used.append(str(value))
    if not cols_used:
        # Use explicit columns mentioned in the question when available.
        cols_used = _explicit_column_mentions(question, list(df.columns))[:4]
    cols_text = ", ".join(f"`{c}`" for c in cols_used) if cols_used else "dataset-level summary"
    evidence_count = ""
    if response.table is not None and not response.table.empty:
        evidence_count = f"Table returned: {len(response.table):,} row(s)."
    elif response.chart is not None:
        evidence_count = "Visual chart returned."
    else:
        evidence_count = "Text evidence returned."
    business_hint = _business_use_hint(context, question)
    response.answer = (
        "### Copilot answer\n"
        f"**Question understood as:** {response.interpreted_question or question}\n\n"
        f"**Dataset used:** `{dataset_name}` ({len(df):,} rows, {df.shape[1]:,} columns).\n\n"
        f"**Columns/artefacts used:** {cols_text}.\n\n"
        f"**Evidence from the uploaded data:** {response.answer}\n\n"
        f"**Business / decision-support use:** {business_hint}\n\n"
        f"**Evidence format:** {evidence_count}"
    )
    return response


def answer_question(
    question: str,
    df: pd.DataFrame,
    dataset_name: str = "active_dataset",
    target_column: Optional[str] = None,
    positive_label: Optional[Any] = None,
    model_output: Optional[Any] = None,
    explanation_output: Optional[Any] = None,
    previous_context: Optional[Dict[str, Any]] = None,
) -> CopilotResponse:
    response = _answer_question_core(
        question,
        df,
        dataset_name=dataset_name,
        target_column=target_column,
        positive_label=positive_label,
        model_output=model_output,
        explanation_output=explanation_output,
        previous_context=previous_context,
    )
    return _frame_copilot_answer(response, question=question, dataset_name=dataset_name, df=df)
