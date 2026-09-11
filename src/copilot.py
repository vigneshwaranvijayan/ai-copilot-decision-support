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
from .readiness_gates import answer_readiness_status, class_balance_status
from .validation import build_readiness_report
from .explainability import top_driver_sentences
from .modeling import infer_positive_label


@dataclass
class CopilotResponse:
    answer: str
    table: Optional[pd.DataFrame] = None
    chart: Optional[Any] = None
    interpreted_question: str = ""
    corrections: List[str] = field(default_factory=list)
    safety_warning: str = "The output is decision support only. A human reviewer should check the context before acting."
    context: Dict[str, Any] = field(default_factory=dict)
    # Explicit evaluation fields are populated by the presentation wrapper and
    # exported to CSV so a reviewer does not need to parse markdown text.
    grounding_status: str = ""
    grounding_reason: str = ""
    confidence: str = ""
    intent: str = ""
    limitation_or_refusal: bool = False
    # Separates successful grounded answers from successful limitation/safety
    # behaviour.  This is exported for Chapter 5 so a correct refusal is not
    # confused with a system failure.
    response_type: str = "GROUNDED_ANSWER"


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
    "switched", "dropped", "declined", "concern", "concerns", "never",
    "lack", "lacking", "disconnect", "downtime", "unacceptable",
}

THEME_NEGATIVE_HINTS = {
    # Use concern phrases, not broad neutral terms such as "monthly" or "charge" alone.
    # This prevents neutral generated feedback mentioning monthly charges from being counted as negative.
    "Pricing and monthly charges": ["expensive", "overpriced", "costly", "too expensive", "too high", "high charge", "high charges", "high monthly", "billing issue", "billing issues", "billing problem", "price increase", "charge increase", "overcharged"],
    "Contract and cancellation risk": ["cancel", "cancelled", "canceled", "churn", "leave", "leaving", "switch", "switched", "regret"],
    "Internet/service reliability": ["slow", "unreliable", "disconnect", "downtime", "outage", "broken", "poor", "bad", "problem", "issues"],
    "Support experience": ["rude", "delay", "delayed", "complaint", "complaints", "unhelpful", "poor", "bad", "problem", "issues"],
    "Early-life onboarding": ["confusing", "difficult", "problem", "issues", "cancel", "churn", "regret", "not_satisfied"],
    "Value and package fit": ["poor", "bad", "lack", "lacking", "expensive", "overpriced", "not_satisfied", "disappointed"],
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

# Reuse readiness evidence for the same in-memory dataframe/target. Rebuilding
# the full readiness report for every chat question adds needless latency.
_READINESS_REPORT_CACHE: Dict[Tuple[int, str, str], Any] = {}

def _cached_readiness_report(df: pd.DataFrame, dataset_name: str, target_column: Optional[str]):
    key = (id(df), str(dataset_name), str(target_column or ""))
    report = _READINESS_REPORT_CACHE.get(key)
    if report is None:
        report = build_readiness_report(df, dataset_name=dataset_name, target_column=target_column)
        # Tiny bounded cache: Streamlit normally has only a few active datasets.
        if len(_READINESS_REPORT_CACHE) >= 16:
            _READINESS_REPORT_CACHE.clear()
        _READINESS_REPORT_CACHE[key] = report
    return report


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def _dataset_source_sentence(dataset_name: str) -> str:
    return f"Source used: active uploaded dataset `{dataset_name}`."


def _simple_readiness_status(df: pd.DataFrame) -> Tuple[str, float, str]:
    """Small built-in quality signal used in every Copilot answer.

    Thresholds are aligned with the research readiness framework:
    PASS <=5% missing and <=5% duplicates; WARNING <=20% missing and
    <=15% duplicates; FAIL beyond those thresholds or no usable data.
    """
    if df is None or df.empty or df.shape[1] < 2:
        return "FAIL", 0.0, "Dataset is empty or has too few columns."
    total_cells = max(int(df.shape[0] * df.shape[1]), 1)
    missing_pct = float(df.isna().sum().sum()) / total_cells * 100
    duplicate_pct = float(df.duplicated().sum()) / max(len(df), 1) * 100
    score = max(0.0, round(100 - min(missing_pct * 2.0, 55) - min(duplicate_pct * 1.2, 30), 2))
    if missing_pct > 20 or duplicate_pct > 15:
        return "FAIL", score, f"Readiness failed: {missing_pct:.2f}% missing cells and {duplicate_pct:.2f}% duplicate rows."
    if missing_pct > 5 or duplicate_pct > 5:
        return "WARNING", score, f"Review quality before decisions: {missing_pct:.2f}% missing cells and {duplicate_pct:.2f}% duplicate rows."
    return "PASS", score, f"Quality checks passed: {missing_pct:.2f}% missing cells and {duplicate_pct:.2f}% duplicate rows."


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


def _asks_for_identifier(q: str) -> bool:
    tokens = _query_tokens(q)
    compact = _compact(q)
    return bool(tokens & {"id", "ids", "customerid", "userid", "accountid", "identifier", "identifiers"} or "customerid" in compact)


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
    # ID-like fields are useful for showing records but weak as analysis columns.
    # Exclude them unless the user explicitly asks for identifiers.
    if not _asks_for_identifier(q):
        id_like = set(_find_id_like_columns(df)) if "_find_id_like_columns" in globals() else set()
        candidates = [c for c in candidates if c not in id_like]
    return find_best_column(q, candidates)


def _is_short_followup(q: str) -> bool:
    tokens = _query_tokens(q)
    compact = _compact(q)
    follow_terms = {"same", "again", "this", "that", "those", "them", "it", "its", "more", "top", "bottom", "highest", "lowest", "worst", "best", "next"}
    if 0 < len(tokens) <= 7 and bool(tokens & follow_terms):
        return True
    # Natural follow-ups can be longer than five words.
    return any(phrase in compact for phrase in [
        "explainthatagain", "explaintheagain", "simplebusinesslanguage",
        "whyisthatmodelbetter", "whyisthismodelbetter", "itsmainchurndrivers",
        "whatwereitsmaindrivers", "whatwereitsmainchurndrivers",
    ])


def _apply_followup_context(q: str, previous_context: Optional[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not previous_context or not _is_short_followup(q):
        return q
    # If user clearly typed any column, never append old column context.
    if _explicit_column_mentions(q, columns):
        return q
    tokens = _query_tokens(q)
    compact = _compact(q)
    topic = str(previous_context.get("topic", "")).lower()

    # Explicit conversational follow-up mappings. These keep the same evidence
    # topic instead of blindly appending a column name.  Never erase an explicit
    # customer/record/prediction reference: that wording requests LOCAL SHAP.
    if "simplebusinesslanguage" in compact or "explainthatagain" in compact:
        local_terms = {"customer", "customers", "record", "case", "prediction", "predicted"}
        if tokens & local_terms or "thiscustomer" in compact:
            return q
        if topic == "local_prediction_explanation":
            return "explain this customer prediction in simple business language"
        if topic in {"drivers", "shap_explanation"}:
            return "explain shap result in simple business language"
        if topic == "model":
            return "explain why the selected model is used for decision support in simple business language"
    if "whyisthatmodelbetter" in compact or "whyisthismodelbetter" in compact:
        return "why is the selected model better for decision support"
    if "itsmainchurndrivers" in compact or "whatwereitsmaindrivers" in compact or "whatwereitsmainchurndrivers" in compact:
        return "what are the main churn drivers"

    # Data-purpose and broad improvement questions should stand alone.
    if tokens & {"use", "uses", "usage", "purpose", "help", "helps", "benefit", "benefits", "data", "dataset"}:
        return q
    # Feedback/ranking/technical terms are clear enough and should not inherit an old column.
    if any(term in compact for term in ["feedback", "comment", "review", "sentiment", "technical", "techincal", "issue", "problem", "complaint"]):
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



def _dataset_kind_hint(df: pd.DataFrame, dataset_name: str, target_column: Optional[str]) -> str:
    """Infer a lightweight data-domain label for safer domain-specific answers."""
    compact_cols = {_compact(c) for c in df.columns}
    name = _compact(dataset_name)
    target = _compact(target_column or "")
    if "bank" in name or {"balance", "job", "marital", "education", "housing", "loan", "campaign", "pdays", "poutcome", "y"}.issubset(compact_cols & {"balance", "job", "marital", "education", "housing", "loan", "campaign", "pdays", "poutcome", "y"}):
        # Require several banking/marketing features, not just one generic column.
        bank_hits = len(compact_cols & {"balance", "job", "marital", "education", "housing", "loan", "campaign", "pdays", "previous", "poutcome", "y"})
        if bank_hits >= 4 or target in {"y", "deposit", "subscribed", "response"}:
            return "bank_marketing"
    if "churn" in name or target == "churn" or "churn" in compact_cols:
        return "customer_churn"
    if compact_cols & {"invoice", "invoiceno", "stockcode", "quantity", "unitprice", "revenue", "sales"}:
        return "retail_sales"
    if compact_cols & {"customerfeedback", "feedback", "review", "comment"}:
        return "customer_feedback"
    return "general"


def _domain_context_guard(q: str, df: pd.DataFrame, dataset_name: str, target_column: Optional[str]) -> Optional[CopilotResponse]:
    """Stop unsupported domain-specific questions instead of forcing them onto the active dataset.

    Example: asking a bank-marketing question while the active data is Telco churn should produce a
    clear FAIL-style answer rather than campaign suggestions from `churn`.
    """
    tokens = _query_tokens(q)
    compact_cols = {_compact(c) for c in df.columns}
    kind = _dataset_kind_hint(df, dataset_name, target_column)
    source = _dataset_source_sentence(dataset_name)

    # Explicit missing column checks for common benchmark/business questions.
    required_terms = {
        "balance": {"balance", "accountbalance"},
    }
    for term, possible_cols in required_terms.items():
        class_balance_intent = term == "balance" and bool(tokens & {"class", "classes", "balanced", "imbalanced", "imbalance", "target", "churn"})
        if term in tokens and not class_balance_intent and not (compact_cols & possible_cols):
            table = pd.DataFrame([{
                "requested_evidence": term,
                "status": "FAIL",
                "reason": f"The active dataset does not contain a `{term}` column.",
                "available_target": target_column or "not selected",
            }])
            return CopilotResponse(
                f"I cannot answer this reliably because the question asks about `{term}`, but the active dataset does not contain a `{term}` column. {source}",
                table=table,
                interpreted_question=q,
                safety_warning="No answer was generated because the required evidence is missing from the active dataset.",
                context={"topic": "data_readiness_fail", "target_column": target_column},
            )

    bank_question = bool(tokens & {"bank", "campaign", "marketing"}) or "bankmarketing" in _compact(q)
    response_question = bool(tokens & {"response", "accepted", "subscribe", "subscribed", "subscription", "deposit"})
    if bank_question and kind != "bank_marketing":
        # Allow generic words like customer/target, but refuse clear campaign/bank tasks on Telco/other data.
        table = pd.DataFrame([{
            "active_dataset_type": kind,
            "requested_context": "bank marketing / campaign response",
            "status": "FAIL",
            "reason": "Upload/select the bank marketing dataset before asking campaign-response questions.",
        }])
        return CopilotResponse(
            f"I cannot treat the active dataset as a bank marketing dataset. The current data appears to be `{kind}` and the selected target is `{target_column or 'not selected'}`. Upload/select the bank marketing dataset, then ask this question again. {source}",
            table=table,
            interpreted_question=q,
            safety_warning="Domain mismatch: the Copilot refused to generate unsupported marketing conclusions from the wrong dataset.",
            context={"topic": "domain_mismatch", "target_column": target_column},
        )
    if response_question and kind == "customer_churn" and not (compact_cols & {"response", "campaign", "y", "deposit", "subscribed"}):
        table = pd.DataFrame([{
            "requested_context": "campaign/marketing response",
            "active_target": target_column or "not selected",
            "status": "FAIL",
            "reason": "The active customer-churn dataset does not contain campaign response fields.",
        }])
        return CopilotResponse(
            f"I cannot answer campaign-response questions from the current churn dataset because it does not contain campaign/response fields. {source}",
            table=table,
            interpreted_question=q,
            safety_warning="No campaign recommendation was generated because the required campaign evidence is missing.",
            context={"topic": "domain_mismatch", "target_column": target_column},
        )
    return None


def _target_column_response(q: str, df: pd.DataFrame, dataset_name: str, target_column: Optional[str], positive_label: Optional[Any]) -> Optional[CopilotResponse]:
    tokens = _query_tokens(q)
    compact = _compact(q)
    if not (("target" in tokens and "column" in tokens) or "targetcolumn" in compact or "selectedtarget" in compact):
        return None
    source = _dataset_source_sentence(dataset_name)
    if target_column and target_column in df.columns:
        counts = df[target_column].dropna().astype(str).value_counts()
        unique_values = counts.head(10).to_dict()
        class_count = int(counts.shape[0])
        table = pd.DataFrame([{
            "selected_target_column": target_column,
            "positive_label": positive_label if positive_label is not None else "auto/not specified",
            "class_count": class_count,
            "unique_values_preview": str(unique_values),
            "row_count": len(df),
        }])
        if "how" in tokens and "many" in tokens and "classes" in tokens:
            answer = f"The selected target column `{target_column}` has **{class_count} classes**: {', '.join(map(str, counts.index.tolist()))}. {source}"
        elif "valid" in tokens and bool(tokens & {"classification", "target"}):
            valid = class_count >= 2
            answer = f"{'Yes' if valid else 'No'}. The selected target `{target_column}` has {class_count} usable class values, so it is {'valid' if valid else 'not valid'} for the current classification workflow. {source}"
        else:
            answer = f"The selected target column for the active dataset is `{target_column}`. Positive label: `{positive_label}`. {source}"
        return CopilotResponse(
            answer,
            table=table,
            interpreted_question=q,
            context={"topic": "target_column", "target_column": target_column},
        )
    table = pd.DataFrame({"candidate_target_columns": _target_candidates(df)})
    return CopilotResponse(
        f"No target column is currently selected. Choose one of the candidate binary target columns, then run modelling. {source}",
        table=table,
        interpreted_question=q,
        safety_warning="Model answers require an explicitly selected target column.",
        context={"topic": "target_column"},
    )



def _class_balance_response(q: str, df: pd.DataFrame, dataset_name: str, target_column: Optional[str], positive_label: Optional[Any]) -> Optional[CopilotResponse]:
    """Answer class-balance questions without confusing `balanced` with a bank `balance` column."""
    tokens = _query_tokens(q)
    if not (tokens & {"balance", "balanced", "imbalanced", "imbalance"}):
        return None
    if not (tokens & {"class", "classes", "target", "churn", "positive", "negative", "dataset"}):
        return None
    if not target_column or target_column not in df.columns:
        candidates = _target_candidates(df)
        table = pd.DataFrame({"candidate_target_columns": candidates})
        return CopilotResponse(
            f"I can check class balance after a target column is selected. Candidate target columns are listed below. {_dataset_source_sentence(dataset_name)}",
            table=table,
            interpreted_question=q,
            context={"topic": "class_balance"},
        )
    counts = df[target_column].dropna().astype(str).value_counts().rename_axis("class_value").reset_index(name="count")
    total = max(int(counts["count"].sum()), 1)
    counts["percent"] = (counts["count"] / total * 100).round(2)
    status, minority, message = class_balance_status(df[target_column])
    pos = positive_label if positive_label is not None else "auto/not specified"
    majority = 100.0 - minority if counts.shape[0] == 2 else float(counts["percent"].max())
    balance_interpretation = (
        f"The classes are not perfectly balanced ({minority:.2f}% minority versus {majority:.2f}% majority), "
        f"but this is within the dissertation's {status} readiness threshold. "
    )
    answer = (
        f"The selected target `{target_column}` has {counts.shape[0]} class values. "
        + balance_interpretation
        + f"Positive class used for churn modelling: `{pos}`. {message} {_dataset_source_sentence(dataset_name)}"
    )
    return CopilotResponse(
        answer,
        table=counts,
        interpreted_question=q,
        safety_warning="Class balance is decision-support evidence. Use recall, precision, F1-score and ROC-AUC rather than accuracy alone.",
        context={"topic": "class_balance", "target_column": target_column},
    )

def _target_candidates(df: pd.DataFrame) -> List[str]:
    candidates: List[str] = []
    for col in df.columns:
        nunique = int(df[col].dropna().nunique())
        name = _compact(col)
        if 2 <= nunique <= 5 or name in {"churn", "y", "target", "response", "deposit", "subscribed", "attrition", "exited"}:
            candidates.append(col)
    return candidates[:12]



def _numeric_target_relationship_response(
    q: str,
    df: pd.DataFrame,
    target_column: Optional[str],
    positive_label: Optional[Any],
    dataset_name: str,
) -> Optional[CopilotResponse]:
    """Answer questions like 'does balance affect campaign response?' using binned target rate."""
    tokens = _query_tokens(q)
    compact = _compact(q)
    relationship_terms = {"affect", "affects", "influence", "influences", "impact", "impacts", "linked", "link", "relationship", "correlation", "correlate", "effect", "relate", "relates", "related", "compared", "compare", "comparison", "higher", "lower"}
    if not target_column or target_column not in df.columns:
        return None
    if not (tokens & relationship_terms or "affect" in compact or "influence" in compact):
        return None
    explicit = _explicit_column_mentions(q, list(df.columns))
    numeric_cols = [c for c in explicit if c != target_column and c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    # In phrases such as "does balance affect campaign response", `campaign` is often
    # part of the response context, while `balance` is the predictor being tested.
    # Prefer non-response/context numeric columns when more than one numeric column is mentioned.
    if len(numeric_cols) > 1:
        preferred = [c for c in numeric_cols if _compact(c) not in {"campaign", "response", "y", "target"}]
        if preferred:
            numeric_cols = preferred
    if not numeric_cols:
        candidate = _safe_best_column(q, df, allow_target=False, target_column=target_column)
        if candidate and candidate in df.columns and pd.api.types.is_numeric_dtype(df[candidate]):
            numeric_cols = [candidate]
    if not numeric_cols:
        return None
    num_col = numeric_cols[0]
    work = df[[num_col, target_column]].copy()
    work[num_col] = pd.to_numeric(work[num_col], errors="coerce")
    work = work.dropna(subset=[num_col, target_column])
    if work.empty or work[num_col].nunique() < 2:
        return CopilotResponse(
            f"I found `{num_col}`, but there is not enough numeric variation to compare it with `{target_column}`. {_dataset_source_sentence(dataset_name)}",
            interpreted_question=q,
            context={"topic": "target_relationship", "rank_col": num_col, "target_column": target_column},
        )
    pos = positive_label if positive_label is not None else work[target_column].dropna().value_counts().index[0]
    try:
        work["__band"] = pd.qcut(work[num_col], q=min(4, work[num_col].nunique()), duplicates="drop")
    except Exception:
        work["__band"] = pd.cut(work[num_col], bins=min(4, max(2, work[num_col].nunique())), duplicates="drop")
    work["__positive"] = (work[target_column].astype(str) == str(pos)).astype(int)
    table = work.groupby("__band", observed=False).agg(
        min_value=(num_col, "min"),
        max_value=(num_col, "max"),
        record_count=(target_column, "size"),
        positive_rate=("__positive", "mean"),
    ).reset_index()
    table["band"] = table["__band"].astype(str)
    table["positive_rate_percent"] = (table["positive_rate"] * 100).round(2)
    table = table[["band", "min_value", "max_value", "record_count", "positive_rate_percent"]]
    chart = px.bar(table, x="band", y="positive_rate_percent", hover_data=["record_count"], title=f"{target_column} positive rate by {num_col} band")
    high = table.loc[table["positive_rate_percent"].idxmax()]
    low = table.loc[table["positive_rate_percent"].idxmin()]
    answer = (
        f"I compared `{num_col}` bands against the selected target `{target_column}`. "
        f"The highest positive-rate band is {high['band']} with {high['positive_rate_percent']:.2f}% positive rate; "
        f"the lowest is {low['band']} with {low['positive_rate_percent']:.2f}%. "
        "This shows an observed dataset pattern, not proof of causation. "
        f"{_dataset_source_sentence(dataset_name)}"
    )
    return CopilotResponse(
        answer,
        table=table,
        chart=chart,
        interpreted_question=q,
        safety_warning="Correlation is not causation. Use this as evidence for investigation, not as an automatic decision rule.",
        context={"topic": "target_relationship", "rank_col": num_col, "target_column": target_column},
    )


def _highest_target_segments(df: pd.DataFrame, target_column: Optional[str], positive_label: Optional[Any], dataset_name: str, q: str) -> Optional[CopilotResponse]:
    tokens = _query_tokens(q)
    if not target_column or target_column not in df.columns:
        return None
    if not (tokens & {"highest", "high", "risk", "risky", "group", "groups", "segment", "segments"}):
        return None
    if not (_is_explicit_column(q, target_column) or _compact(target_column) in _compact(q) or "churn" in tokens or "risk" in tokens):
        return None
    pos = positive_label if positive_label is not None else df[target_column].dropna().value_counts().index[0]
    min_count = max(20, int(len(df) * 0.01))
    id_like = set(_find_id_like_columns(df))
    rows: List[Dict[str, Any]] = []
    for col in _categorical_columns(df, max_unique=80):
        if col == target_column or col in id_like:
            continue
        temp = df[[col, target_column]].dropna().copy()
        if temp.empty:
            continue
        temp["__positive"] = (temp[target_column].astype(str) == str(pos)).astype("int8")
        grouped = temp.groupby(col, dropna=False)["__positive"].agg(["mean", "count"]).reset_index()
        grouped = grouped[grouped["count"] >= min_count]
        if grouped.empty:
            continue
        for rec in grouped.itertuples(index=False):
            rows.append({
                "segment_column": col,
                "segment_value": getattr(rec, str(col)) if str(col).isidentifier() and hasattr(rec, str(col)) else rec[0],
                "positive_rate_percent": round(float(rec.mean) * 100, 2),
                "record_count": int(rec.count),
            })
    if not rows:
        return None
    table = pd.DataFrame(rows).sort_values(["positive_rate_percent", "record_count"], ascending=[False, False]).head(20)
    chart = px.bar(table.head(12), x="positive_rate_percent", y="segment_value", color="segment_column", orientation="h", hover_data=["record_count"], title=f"Highest {target_column} positive-rate segments")
    top = table.iloc[0]
    return CopilotResponse(
        f"The highest observed `{target_column}` positive-rate segment is `{top['segment_column']} = {top['segment_value']}` with {top['positive_rate_percent']:.2f}% positive rate across {int(top['record_count']):,} records. {_dataset_source_sentence(dataset_name)}",
        table=table,
        chart=chart,
        interpreted_question=q,
        context={"topic": "group_segments", "target_column": target_column, "group_col": str(top['segment_column'])},
    )

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
        neg: List[str] = []
        pos: List[str] = []
        for i, token in enumerate(tokens):
            prev_token = tokens[i - 1] if i > 0 else ""
            next_token = tokens[i + 1] if i + 1 < len(tokens) else ""
            # Standalone "not" is too broad for business feedback. Count it only
            # when it clearly negates a positive term, for example "not satisfied".
            if token == "not" and next_token in POSITIVE_WORDS:
                neg.append(f"not_{next_token}")
                continue
            if token in NEGATIVE_WORDS and token != "not":
                neg.append(token)
                continue
            # Avoid counting "satisfied" as positive in phrases like "not satisfied".
            if token in POSITIVE_WORDS and prev_token != "not":
                pos.append(token)
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



def _dataset_use_case_intent(q: str) -> bool:
    """Detect questions asking what the dataset/app can be used for."""
    tokens = _query_tokens(q)
    compact = _compact(q)
    use_terms = {"use", "uses", "usage", "purpose", "purposes", "usecase", "usecases", "help", "helps", "benefit", "benefits", "useful"}
    data_terms = {"data", "dataset", "file", "table", "columns", "information"}
    return bool((tokens & use_terms and (tokens & data_terms or "thisdata" in compact or "thisdataset" in compact)) or "whatuses" in compact or "howcanthisdatahelp" in compact)


def _dataset_use_case_response(q: str, df: pd.DataFrame, dataset_name: str, target_column: Optional[str]) -> CopilotResponse:
    kind = _dataset_kind_hint(df, dataset_name, target_column)
    compact_cols = {_compact(c) for c in df.columns}
    uses: List[Dict[str, str]] = []

    uses.append({
        "use_case": "Data quality and readiness review",
        "how_it_helps": "Check missing values, duplicate rows, schema suitability and whether the dataset is safe enough for analysis.",
        "evidence_used": "row/column counts, missing-cell rate, duplicate-row rate, column types",
    })
    uses.append({
        "use_case": "Exploratory dashboard and visual analysis",
        "how_it_helps": "Summarise important patterns, compare groups and generate charts for business review.",
        "evidence_used": "numeric summaries, category counts, target rates, generated graphs",
    })
    if target_column and target_column in df.columns:
        uses.append({
            "use_case": f"Predictive modelling for `{target_column}`",
            "how_it_helps": "Train classification/regression models and compare accuracy, F1, recall, precision or ROC-AUC where suitable.",
            "evidence_used": f"selected target column `{target_column}` and model leaderboard",
        })
        uses.append({
            "use_case": "Explainable prediction support",
            "how_it_helps": "Explain important model drivers using SHAP or fallback feature importance, then translate drivers into plain English.",
            "evidence_used": "model output, feature importance, SHAP/fallback explanation",
        })
    if kind == "customer_churn" or "churn" in compact_cols:
        uses.append({
            "use_case": "Customer churn retention planning",
            "how_it_helps": "Identify high-risk groups, likely churn drivers and retention actions for manager review.",
            "evidence_used": "churn rate, customer segments, model drivers, recommendation rules",
        })
    if _feedback_columns(df):
        uses.append({
            "use_case": "Customer feedback improvement planning",
            "how_it_helps": "Find service, support, price or contract themes and convert them into improvement ideas.",
            "evidence_used": "feedback text column, theme evidence rows, concern terms",
        })
    if kind == "bank_marketing":
        uses.append({
            "use_case": "Campaign-response targeting",
            "how_it_helps": "Analyse which groups respond better and support future marketing targeting decisions.",
            "evidence_used": "campaign/customer attributes and response target",
        })
    if kind == "retail_sales":
        uses.append({
            "use_case": "Sales and stock decision support",
            "how_it_helps": "Identify strong sellers, weak products, revenue patterns and stock/promotion priorities.",
            "evidence_used": "quantity, price, revenue, product and date columns",
        })

    table = pd.DataFrame(uses)
    answer = (
        f"This dataset can be used for {len(table)} decision-support purpose(s). "
        "The main value is not only viewing data, but using the data for readiness checks, visual analysis, modelling, explainability, business recommendations and safe human review. "
        f"{_dataset_source_sentence(dataset_name)}"
    )
    return CopilotResponse(
        answer,
        table=table,
        interpreted_question=q,
        safety_warning="These are possible uses based only on the active dataset structure. Do not use the dataset for unsupported decisions without the required columns and human review.",
        context={"topic": "dataset_use_cases", "target_column": target_column},
    )


def _technical_or_service_issue_intent(q: str) -> bool:
    tokens = _query_tokens(q)
    compact = _compact(q)
    technical_terms = {"technical", "techincal", "tech", "internet", "network", "connection", "connections", "speed", "slow", "outage", "downtime", "service", "support"}
    issue_terms = {"issue", "issues", "problem", "problems", "complaint", "complaints", "found", "facing", "face", "reported", "report", "pain", "concern", "concerns"}
    return bool(tokens & technical_terms and (tokens & issue_terms or "technicalissue" in compact or "techincalissue" in compact or "serviceissue" in compact))


def _generic_improvement_intent(q: str, df: pd.DataFrame) -> bool:
    """Broad improvement request. Prefer feedback themes when feedback exists."""
    tokens = _query_tokens(q)
    compact = _compact(q)
    action_terms = {"improve", "improvement", "improvements", "develop", "fix", "solve", "better", "idea", "ideas", "suggest", "suggestion", "suggestions", "recommend", "recommendation", "area", "areas"}
    generic_terms = {"best", "which", "what", "any", "one", "where", "how", "business", "customer", "customers", "service"}
    return bool(_feedback_columns(df) and (tokens & action_terms) and (tokens & generic_terms or "howtoimprove" in compact))


def _prioritise_technical_issue_table(table: pd.DataFrame) -> pd.DataFrame:
    """For explicit technical/service issue questions, prefer technical themes.

    The general table also considers churn/target rate, which is useful for
    improvement planning. For direct technical-issue questions, the user expects
    internet/connection/service evidence before pricing or generic support.
    """
    if table is None or table.empty or "theme" not in table.columns:
        return table
    order = {
        "Internet/service reliability": 0,
        "Support experience": 1,
        "Pricing and monthly charges": 2,
        "Contract and cancellation risk": 3,
        "Early-life onboarding": 4,
        "Value and package fit": 5,
    }
    out = table.copy()
    out["__tech_order"] = out["theme"].map(order).fillna(9)
    sort_cols = ["__tech_order"]
    ascending = [True]
    if "negative_evidence_rows" in out.columns:
        sort_cols.append("negative_evidence_rows")
        ascending.append(False)
    if "evidence_rows" in out.columns:
        sort_cols.append("evidence_rows")
        ascending.append(False)
    out = out.sort_values(sort_cols, ascending=ascending).drop(columns=["__tech_order"])
    return out.reset_index(drop=True)



def _feedback_business_action_table(
    df: pd.DataFrame,
    text_col: str,
    target_column: Optional[str] = None,
    positive_label: Optional[Any] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    text_series = df[text_col].fillna("").astype(str)
    text_norm = text_series.map(_norm_text)
    scored_text = _score_text_rows(df, text_col)
    neg_counts = scored_text["negative_keyword_count"]
    pos_counts = scored_text["positive_keyword_count"]
    sentiment_score = scored_text["sentiment_score"]

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

        # Negative/concern evidence is stricter than general theme matching.
        # A row should not be counted as negative just because it contains a broad
        # word such as "service" or "internet". It must either contain a theme-
        # specific concern term or have more negative than positive evidence.
        theme_negative_terms = [_norm_text(t) for t in THEME_NEGATIVE_HINTS.get(rule["theme"], [])]
        theme_negative_mask = pd.Series(False, index=df.index)
        matched_negative_terms: List[str] = []
        for neg_kw in theme_negative_terms:
            if not neg_kw:
                continue
            pattern = r"\b" + re.escape(neg_kw).replace(r"\ ", r"\s+") + r"\b"
            term_mask = text_norm.str.contains(pattern, regex=True, na=False)
            if bool(term_mask.any()):
                matched_negative_terms.append(neg_kw)
            theme_negative_mask = theme_negative_mask | term_mask

        concern_mask = mask & ((sentiment_score > 0) | theme_negative_mask)
        negative_rows = int(concern_mask.sum())
        sample_text = ""
        sample_candidates = df.loc[concern_mask, text_col]
        if sample_candidates.empty:
            sample_candidates = df.loc[mask, text_col]
        if not sample_candidates.empty:
            sample_text = _truncate_text(sample_candidates.astype(str).iloc[0], limit=180)

        target_rate = np.nan
        if target_column and target_column in df.columns and positive_label is not None:
            subset = df.loc[mask, target_column]
            if len(subset) > 0:
                target_rate = (subset.astype(str) == str(positive_label)).mean() * 100

        priority_score = negative_rows * 3 + evidence_rows * 0.15
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
            "negative_evidence_terms": ", ".join(dict.fromkeys(matched_negative_terms[:10])),
            "business_meaning": rule["business_meaning"],
            "recommended_action": rule["recommended_action"],
            "human_review_warning": rule["safety_warning"],
            "example_feedback": sample_text,
            "priority_score": round(priority_score, 2),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "priority", "theme", "evidence_rows", "negative_evidence_rows",
            "target_positive_rate_percent", "matched_terms", "negative_evidence_terms", "business_meaning",
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
    top_items = []
    for _, row in table.head(3).iterrows():
        top_items.append(
            f"**{row['theme']}** ({int(row['negative_evidence_rows']):,} concern row(s); "
            f"action: {row['recommended_action']})"
        )
    top_summary = " Top improvement priorities: " + " ".join(f"{i+1}. {item}" for i, item in enumerate(top_items))
    return (
        f"I found {len(table)} business-action themes / actionable feedback theme(s) in `{text_col}`. "
        f"The highest-priority concern is **{top['theme']}**, based on {int(top['negative_evidence_rows']):,} "
        f"negative/concern evidence row(s) from {int(top['evidence_rows']):,} theme-matching row(s)."
        f"{top_summary} "
        "The detailed table below shows the evidence terms, business meaning, recommended action and human-review warning. "
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
        f"Preferred model under the dissertation's **F1-first selection rule**: **{best.model_name}**. "
        f"F1={metrics.get('f1', float('nan')):.3f}, "
        f"Recall={metrics.get('recall', float('nan')):.3f}, "
        f"Precision={metrics.get('precision', float('nan')):.3f}, "
        f"ROC-AUC={metrics.get('roc_auc', float('nan')):.3f}. "
        "F1 is the primary selection metric because it balances precision and recall; ROC-AUC, recall and precision are tie-breakers. "
        "The model was trained only on the selected uploaded dataset."
    )


def _metric_specific_model_answer(q: str, model_output: Any) -> str:
    """Return model answers that match the exact metric asked by the user.

    This prevents a question such as "Which model has the best ROC-AUC?" from
    incorrectly repeating the overall best-by-F1 model. The dissertation can
    then report that the Copilot answers are grounded in the model leaderboard.
    """
    if model_output is None or getattr(model_output, "leaderboard", None) is None or model_output.leaderboard.empty:
        return _model_summary(model_output)

    lb = model_output.leaderboard.copy()
    q_lower = q.lower()

    # First resolve questions about one named model.  Without this branch a
    # question such as "What is the accuracy of Logistic Regression?" can be
    # mistaken for "Which model has the highest accuracy?".
    named_row = _model_row_from_question(q, model_output)
    if named_row is not None:
        model_name = str(named_row.get("model", "model"))
        metric_aliases_named = [
            ("roc_auc", ["roc-auc", "roc auc", "auc"]),
            ("f1", ["f1", "f1-score", "f1 score"]),
            ("recall", ["recall", "sensitivity"]),
            ("precision", ["precision"]),
            ("accuracy", ["accuracy"]),
        ]
        for metric, aliases in metric_aliases_named:
            if any(alias in q_lower for alias in aliases) and metric in named_row.index:
                return f"The **{metric.replace('_', '-').upper()}** of **{model_name}** is {float(named_row[metric]):.3f}."
        if "performance" in q_lower or "metrics" in q_lower or "score" in q_lower:
            return (
                f"Performance of **{model_name}**: Accuracy={float(named_row.get('accuracy', np.nan)):.3f}, "
                f"Precision={float(named_row.get('precision', np.nan)):.3f}, Recall={float(named_row.get('recall', np.nan)):.3f}, "
                f"F1={float(named_row.get('f1', np.nan)):.3f}, ROC-AUC={float(named_row.get('roc_auc', np.nan)):.3f}."
            )

    # Metric meaning/decision-support questions should explain the metric,
    # rather than returning the leaderboard winner for that metric.
    if "recall" in q_lower and ("why" in q_lower or "important" in q_lower or "mean" in q_lower):
        return (
            "Recall measures how many actual churn cases the model correctly identifies. It matters in churn prediction because low recall means many genuinely at-risk customers are missed (false negatives)."
        )
    if "precision" in q_lower and ("why" in q_lower or "important" in q_lower or "mean" in q_lower):
        return (
            "Precision measures how many customers flagged as churn actually belong to the churn class. It matters because low precision creates more false positives and may waste retention effort or lead to unnecessary interventions."
        )
    if ("f1" in q_lower or "f1-score" in q_lower or "f1 score" in q_lower) and ("what does" in q_lower or "mean" in q_lower):
        return (
            "F1-score is the harmonic mean of precision and recall. In this analysis it is the primary model-selection metric because it rewards a useful balance between finding churn cases and avoiding too many false alarms."
        )
    if ("roc-auc" in q_lower or "roc auc" in q_lower or "auc" in q_lower) and ("what does" in q_lower or "tell us" in q_lower or "mean" in q_lower):
        return (
            "ROC-AUC measures how well the model separates churn from non-churn cases across possible classification thresholds. A higher value means better ranking/separation ability, but it does not by itself choose the business operating threshold."
        )

    metric_aliases = [
        ("roc_auc", ["roc-auc", "roc auc", "auc"]),
        ("f1", ["f1", "f1-score", "f1 score"]),
        ("recall", ["recall", "sensitivity"]),
        ("precision", ["precision"]),
        ("accuracy", ["accuracy"]),
    ]

    def best_for(metric: str) -> str:
        if metric not in lb.columns:
            return _model_summary(model_output)
        row = lb.sort_values(metric, ascending=False).iloc[0]
        return (
            f"The model with the best **{metric.replace('_', '-').upper()}** is **{row['model']}** "
            f"with {metric.replace('_', '-').upper()}={float(row[metric]):.3f}. "
            f"For context, its F1={float(row.get('f1', float('nan'))):.3f}, "
            f"Recall={float(row.get('recall', float('nan'))):.3f}, "
            f"Precision={float(row.get('precision', float('nan'))):.3f} and "
            f"ROC-AUC={float(row.get('roc_auc', float('nan'))):.3f}."
        )

    for metric, aliases in metric_aliases:
        if any(alias in q_lower for alias in aliases):
            if "accuracy alone" in q_lower or ("why" in q_lower and "accuracy" in q_lower):
                break
            return best_for(metric)

    if "accuracy alone" in q_lower or ("why" in q_lower and "accuracy" in q_lower):
        return (
            "Accuracy alone is not enough for churn prediction because the classes are not perfectly balanced. "
            "A model can appear accurate by predicting the majority non-churn class while missing important churn cases. "
            "For decision support, recall shows how many actual churn cases are found, precision shows how many flagged churn cases are correct, "
            "F1 balances precision and recall, and ROC-AUC shows class-separation ability across thresholds."
        )

    if "compare" in q_lower or {"logistic", "random", "gradient", "mlp"} & set(q_lower.split()):
        best_f1 = lb.sort_values("f1", ascending=False).iloc[0]
        best_auc = lb.sort_values("roc_auc", ascending=False).iloc[0] if "roc_auc" in lb.columns else best_f1
        return (
            f"The four assessed models are compared in the leaderboard. By F1-score, the best model is **{best_f1['model']}** "
            f"(F1={float(best_f1['f1']):.3f}). By ROC-AUC, the strongest class-separation result is **{best_auc['model']}** "
            f"(ROC-AUC={float(best_auc['roc_auc']):.3f}). This means the preferred model can depend on the metric: "
            "F1 is useful for balanced decision-support selection, while ROC-AUC is useful for ranking/separation ability."
        )

    if "good enough" in q_lower or "decision support" in q_lower or "should be used" in q_lower or "why is that model better" in q_lower:
        return _model_decision_support_summary(model_output)

    return _model_summary(model_output)


def _model_decision_support_summary(model_output: Any) -> str:
    """Plain-English verdict for supervisor/reviewer questions about model usefulness."""
    if model_output is None or getattr(model_output, "best_result", None) is None:
        return "No trained model is available, so I cannot judge decision-support suitability."
    best = model_output.best_result
    metrics = best.metrics
    f1 = float(metrics.get("f1", np.nan))
    recall = float(metrics.get("recall", np.nan))
    precision = float(metrics.get("precision", np.nan))
    roc_auc = float(metrics.get("roc_auc", np.nan))
    verdict = "limited"
    if (not np.isnan(roc_auc) and roc_auc >= 0.80) and (not np.isnan(f1) and f1 >= 0.60):
        verdict = "suitable for decision support, but not for automatic decisions"
    elif (not np.isnan(roc_auc) and roc_auc >= 0.70) or (not np.isnan(f1) and f1 >= 0.50):
        verdict = "moderately useful for decision support, with clear limitations"
    caveats = []
    if not np.isnan(precision) and precision < 0.60:
        caveats.append("precision is below 0.60, so some flagged cases may be false positives")
    if not np.isnan(recall) and recall < 0.60:
        caveats.append("recall is below 0.60, so some positive cases may be missed")
    if not caveats:
        caveats.append("a human reviewer should still validate decisions against business context")
    return (
        f"Preferred model under the F1-first selection rule: **{best.model_name}**. F1={f1:.3f}, Recall={recall:.3f}, Precision={precision:.3f}, ROC-AUC={roc_auc:.3f}. "
        f"Overall verdict: the model is **{verdict}**. Main caveat: {'; '.join(caveats)}. "
        "Use the model to prioritise review, not to make automatic customer decisions."
    )


def _recommend_from_drivers(explanation_output: Any, positive_only: bool = False) -> str:
    """Translate local explanation evidence into transparent action themes.

    When ``positive_only`` is True, recommendations are generated only from
    features with a positive signed local SHAP contribution. This prevents the
    system from recommending action because of a feature that actually reduced
    the selected customer's predicted churn risk.
    """
    if explanation_output is None or getattr(explanation_output, "local_importance", None) is None:
        return "Recommendation rules need a trained model and explanation output first."
    local = explanation_output.local_importance.copy()
    if positive_only and "contribution" in local.columns:
        local["contribution"] = pd.to_numeric(local["contribution"], errors="coerce")
        local = local[local["contribution"] > 0].sort_values("contribution", ascending=False)
    local = local.head(8)
    suggestions: List[str] = []
    for feature in local.get("feature", []):
        feature_norm = str(feature).lower().replace("_", " ")
        for token, suggestion in RECOMMENDATION_RULES:
            if token in feature_norm and suggestion not in suggestions:
                suggestions.append(suggestion)
                break
    if not suggestions:
        suggestions.append("Review the strongest positive local model drivers and compare them with the customer's current business context before choosing an action.")
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


def _model_row_from_question(q: str, model_output: Any) -> Optional[pd.Series]:
    """Resolve a named assessed model from natural-language reviewer questions."""
    if model_output is None or getattr(model_output, "leaderboard", None) is None:
        return None
    lb = model_output.leaderboard
    if lb is None or lb.empty or "model" not in lb.columns:
        return None
    text = _norm_text(q)
    token_set = set(text.split())
    aliases = {
        "logistic regression": (["logistic regression", "logistic"], ["lr"]),
        "random forest": (["random forest", "randomforest"], ["rf"]),
        "gradient boosting": (["gradient boosting", "gradientboosting"], ["gb"]),
        "mlp neural network baseline": (["mlp", "neural network", "neuralnetwork"], []),
    }
    for canonical, (phrases, short_tokens) in aliases.items():
        if any(name in text for name in phrases) or any(tok in token_set for tok in short_tokens):
            mask = lb["model"].astype(str).str.lower().str.contains(canonical.split()[0], regex=False)
            if canonical.startswith("mlp"):
                mask = lb["model"].astype(str).str.lower().str.contains("mlp", regex=False)
            if mask.any():
                return lb.loc[mask].iloc[0]
    return None


def _model_metrics_table_row(row: pd.Series) -> pd.DataFrame:
    keep = [c for c in ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "training_rows", "training_time_sec"] if c in row.index]
    return pd.DataFrame([{c: row[c] for c in keep}])


def _confusion_evidence(model_output: Any) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, int]]]:
    if model_output is None or getattr(model_output, "best_result", None) is None:
        return None, None
    conf = np.asarray(getattr(model_output.best_result, "confusion", None))
    if conf.shape != (2, 2):
        return None, None
    tn, fp, fn, tp = [int(x) for x in conf.ravel()]
    table = pd.DataFrame([
        {"confusion_term": "true negatives", "count": tn, "meaning": "Actual non-churn correctly predicted as non-churn."},
        {"confusion_term": "false positives", "count": fp, "meaning": "Actual non-churn incorrectly flagged as churn."},
        {"confusion_term": "false negatives", "count": fn, "meaning": "Actual churn incorrectly predicted as non-churn."},
        {"confusion_term": "true positives", "count": tp, "meaning": "Actual churn correctly predicted as churn."},
    ])
    return table, {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _feature_relationship_summary(
    df: pd.DataFrame,
    target_column: Optional[str],
    positive_label: Optional[Any],
    *,
    kind: str,
    max_features: int = 12,
) -> pd.DataFrame:
    """Rank simple descriptive relationships with the binary target.

    This is deliberately descriptive rather than causal.  Categorical columns
    are ranked by the spread in observed positive rates across categories;
    numeric columns are ranked by the spread across quartile-style bands.
    """
    if not target_column or target_column not in df.columns:
        return pd.DataFrame()
    pos = positive_label if positive_label is not None else infer_positive_label(df[target_column])
    rows: List[Dict[str, Any]] = []
    # Avoid identifier-like columns without importing the modelling helper.
    id_cols = set(_find_id_like_columns(df))
    for col in df.columns:
        if col == target_column or col in id_cols:
            continue
        series = df[col]
        is_num = pd.api.types.is_numeric_dtype(series)
        if kind == "categorical" and is_num:
            continue
        if kind == "numeric" and not is_num:
            continue
        try:
            tmp = df[[col, target_column]].copy()
            tmp = tmp.dropna(subset=[target_column])
            tmp["__positive"] = (tmp[target_column].astype(str) == str(pos)).astype(int)
            if is_num:
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
                tmp = tmp.dropna(subset=[col])
                if tmp[col].nunique() < 2:
                    continue
                try:
                    tmp["__group"] = pd.qcut(tmp[col], q=min(4, tmp[col].nunique()), duplicates="drop")
                except Exception:
                    tmp["__group"] = pd.cut(tmp[col], bins=min(4, tmp[col].nunique()), duplicates="drop")
                grp = tmp.groupby("__group", observed=False)["__positive"].agg(["mean", "count"]).reset_index()
            else:
                if tmp[col].nunique(dropna=True) < 2 or tmp[col].nunique(dropna=True) > 120:
                    continue
                grp = tmp.groupby(col, dropna=False)["__positive"].agg(["mean", "count"]).reset_index()
            grp = grp[grp["count"] >= max(10, int(len(tmp) * 0.005))]
            if len(grp) < 2:
                continue
            spread = float(grp["mean"].max() - grp["mean"].min()) * 100
            rows.append({
                "feature": col,
                "relationship_type": "numeric bands" if is_num else "category groups",
                "positive_rate_spread_percent_points": round(spread, 2),
                "groups_compared": int(len(grp)),
                "interpretation": "Larger spread means stronger descriptive separation across observed groups; it does not prove causation.",
            })
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("positive_rate_spread_percent_points", ascending=False).head(max_features) if rows else pd.DataFrame()


def _dataset_pattern_summary(
    df: pd.DataFrame,
    dataset_name: str,
    target_column: Optional[str],
    positive_label: Optional[Any],
) -> Tuple[str, pd.DataFrame]:
    """Create a compact, auditable descriptive pattern summary for the active dataset."""
    rows: List[Dict[str, Any]] = []
    source = _dataset_source_sentence(dataset_name)
    if target_column and target_column in df.columns:
        pos = positive_label if positive_label is not None else infer_positive_label(df[target_column])
        overall = (df[target_column].astype(str) == str(pos)).mean() * 100
        rows.append({"pattern": "Overall target rate", "evidence": f"{overall:.2f}% of records are positive for {target_column}={pos}."})

        contract = next((c for c in df.columns if _compact(c) == "contract"), None)
        if contract:
            tmp = df[[contract, target_column]].dropna().copy()
            tmp["__positive"] = (tmp[target_column].astype(str) == str(pos)).astype(int)
            grp = tmp.groupby(contract)["__positive"].agg(["mean", "count"]).reset_index().sort_values("mean", ascending=False)
            if not grp.empty:
                r = grp.iloc[0]
                rows.append({"pattern": "Contract", "evidence": f"{r[contract]} has the highest observed positive rate at {float(r['mean'])*100:.2f}% across {int(r['count']):,} records."})

        tenure = next((c for c in df.columns if _compact(c) == "tenure"), None)
        if tenure:
            rel = _numeric_target_relationship_response(f"how does {tenure} relate to {target_column}", df, target_column, pos, dataset_name)
            if rel is not None and rel.table is not None and not rel.table.empty:
                t = rel.table.sort_values("positive_rate_percent", ascending=False)
                hi, lo = t.iloc[0], t.iloc[-1]
                rows.append({"pattern": "Tenure", "evidence": f"Highest tenure-band positive rate is {float(hi['positive_rate_percent']):.2f}%; lowest is {float(lo['positive_rate_percent']):.2f}%."})

        monthly = next((c for c in df.columns if _compact(c) == "monthlycharges"), None)
        if monthly:
            rel = _numeric_target_relationship_response(f"how does {monthly} relate to {target_column}", df, target_column, pos, dataset_name)
            if rel is not None and rel.table is not None and not rel.table.empty:
                t = rel.table.sort_values("positive_rate_percent", ascending=False)
                hi, lo = t.iloc[0], t.iloc[-1]
                rows.append({"pattern": "Monthly charges", "evidence": f"Highest monthly-charge-band positive rate is {float(hi['positive_rate_percent']):.2f}%; lowest is {float(lo['positive_rate_percent']):.2f}%."})

    if not rows:
        rows.append({"pattern": "Dataset profile", "evidence": f"{len(df):,} rows, {df.shape[1]:,} columns, {int(df.isna().sum().sum()):,} missing cells and {int(df.duplicated().sum()):,} duplicate rows."})
    table = pd.DataFrame(rows)
    answer = "Main evidence-backed patterns: " + " ".join(f"{r['pattern']}: {r['evidence']}" for r in rows) + " These are descriptive associations, not causal conclusions. " + source
    return answer, table


def _local_prediction_values(model_output: Any, explanation_output: Any) -> Tuple[Optional[str], Optional[float]]:
    """Return the currently explained record's predicted label and positive-class probability."""
    label = getattr(explanation_output, "predicted_label", None) if explanation_output is not None else None
    probability = getattr(explanation_output, "predicted_probability", None) if explanation_output is not None else None
    if label is not None or probability is not None:
        try:
            probability = float(probability) if probability is not None else None
        except Exception:
            probability = None
        return str(label) if label is not None else None, probability
    if model_output is None or getattr(model_output, "best_result", None) is None:
        return None, None
    best = model_output.best_result
    try:
        row = best.X_test.head(1)
        label = best.pipeline.predict(row)[0]
        probability = None
        if hasattr(best.pipeline, "predict_proba"):
            proba = best.pipeline.predict_proba(row)[0]
            classes = list(getattr(best.pipeline, "classes_", []))
            idx = next((i for i, c in enumerate(classes) if str(c) == str(getattr(model_output, "positive_label", best.positive_label))), 1 if len(proba) > 1 else 0)
            probability = float(proba[idx])
        return str(label), probability
    except Exception:
        return None, None



def _reviewer_question_response(
    q: str,
    df: pd.DataFrame,
    dataset_name: str,
    target_column: Optional[str],
    positive_label: Optional[Any],
    model_output: Optional[Any],
    explanation_output: Optional[Any],
    previous_context: Optional[Dict[str, Any]],
) -> Optional[CopilotResponse]:
    """High-priority deterministic answers for dissertation review questions.

    These intents are deliberately explicit because they are evaluated as part
    of RQ2.  A reviewer question must either receive evidence that directly
    answers it or a clear limitation/refusal; it must never be routed to a
    vaguely related churn-rate or domain table.
    """
    tokens = _query_tokens(q)
    compact = _compact(q)
    source = _dataset_source_sentence(dataset_name)

    # ------------------------------------------------------------------
    # High-priority reviewer/evaluation intents. These are intentionally
    # checked before broader churn/model handlers so one question maps to the
    # exact evidence it asks for rather than to a vaguely related statistic.
    # ------------------------------------------------------------------

    # Active dataset identity.
    if "dataset" in tokens and bool(tokens & {"loaded", "current", "currently", "active"}):
        table = pd.DataFrame([{
            "dataset_name": dataset_name,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "memory_mb": round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 3),
        }])
        return CopilotResponse(
            f"The currently loaded dataset is **{dataset_name}** with {len(df):,} rows and {df.shape[1]:,} columns. {source}",
            table=table,
            interpreted_question=q,
            context={"topic": "dataset_identity", "target_column": target_column},
        )

    # Data-quality, data-type and validation-limitation questions are grounded
    # directly in the dissertation PASS/WARNING/FAIL gate table.
    quality_question = (
        ("data" in tokens and "quality" in tokens)
        or "dataquality" in compact
        or ("data" in tokens and "types" in tokens)
        or "datatypes" in compact
        or (bool(tokens & {"limitation", "limitations"}) and bool(tokens & {"validation", "readiness", "data"}))
    )
    if quality_question:
        rep = build_readiness_report(df, dataset_name=dataset_name, target_column=target_column)
        table = rep.gates_dataframe()
        if "types" in tokens or "datatypes" in compact:
            view = table[table["gate"].astype(str).str.contains("Data types", case=False, regex=False)].copy()
            status = view.iloc[0]["status"] if not view.empty else "UNKNOWN"
            message = view.iloc[0]["message"] if not view.empty else "No explicit data-type gate was produced."
            return CopilotResponse(
                f"Data-type suitability is **{status}**. {message} {source}",
                table=view if not view.empty else table,
                interpreted_question=q,
                context={"topic": "data_types_readiness", "target_column": target_column},
            )
        non_pass = table[(table["status"] != "PASS") & ~table["gate"].astype(str).str.startswith("Supplementary")].copy() if not table.empty else pd.DataFrame()
        missing_cells = int(df.isna().sum().sum())
        dup = int(df.duplicated().sum())
        if bool(tokens & {"limitation", "limitations"}):
            if non_pass.empty:
                answer = (
                    f"No blocking validation limitation was identified: all core readiness gates pass. "
                    f"The dataset still contains {missing_cells:,} missing cell(s) and {dup:,} duplicate row(s), "
                    "so these should be reported transparently even though they remain within the PASS thresholds. "
                )
                view = table
            else:
                answer = "The validation limitations are the non-PASS core readiness gates shown in the evidence table. "
                view = non_pass
            return CopilotResponse(
                answer + source,
                table=view,
                interpreted_question=q,
                context={"topic": "validation_limitations", "target_column": target_column},
            )
        return CopilotResponse(
            f"Overall data quality/readiness is **{rep.overall_status}** with quality score {rep.quality_score:.1f}/100. "
            f"The current data contains {missing_cells:,} missing cell(s) and {dup:,} duplicate row(s). {source}",
            table=table,
            interpreted_question=q,
            context={"topic": "data_quality_summary", "target_column": target_column},
        )

    # Explicit negative-class percentage; do not answer with the positive rate.
    if target_column and target_column in df.columns and (
        "didnotchurn" in compact or "notchurn" in compact or
        ({"percentage", "customers"} <= tokens and "not" in tokens and "churn" in tokens)
    ):
        pos = positive_label if positive_label is not None else infer_positive_label(df[target_column])
        pos_rate = (df[target_column].astype(str) == str(pos)).mean() * 100
        neg_rate = 100.0 - pos_rate
        counts = target_distribution(df, target_column)
        return CopilotResponse(
            f"{neg_rate:.2f}% of customers are in the non-churn class; {pos_rate:.2f}% are in the churn/positive class. {source}",
            table=counts,
            interpreted_question=q,
            context={"topic": "target_distribution", "target_column": target_column},
        )

    # Main EDA/visual pattern summary.
    if (
        ("main" in tokens and "patterns" in tokens)
        or ("important" in tokens and "pattern" in tokens and bool(tokens & {"visualisation", "visualisations", "visualization", "visualizations"}))
        or "mainpatterns" in compact
    ):
        answer, table = _dataset_pattern_summary(df, dataset_name, target_column, positive_label)
        return CopilotResponse(
            answer,
            table=table,
            interpreted_question=q,
            context={"topic": "visual_pattern_summary", "target_column": target_column},
        )

    # Direct month-to-month comparison. This must precede generic target
    # distribution matching because the question asks for a segment comparison.
    if target_column and target_column in df.columns and "monthtomonth" in compact and "churn" in tokens:
        contract = next((c for c in df.columns if _compact(c) == "contract"), None)
        if contract:
            pos = positive_label if positive_label is not None else infer_positive_label(df[target_column])
            tmp = df[[contract, target_column]].dropna().copy()
            tmp["__positive"] = (tmp[target_column].astype(str) == str(pos)).astype(int)
            table = tmp.groupby(contract)["__positive"].agg(["mean", "count"]).reset_index()
            table["positive_rate_percent"] = (table["mean"] * 100).round(2)
            table = table.drop(columns=["mean"]).sort_values("positive_rate_percent", ascending=False)
            mtm = table[table[contract].astype(str).str.lower().str.contains("month-to-month", regex=False)]
            if not mtm.empty:
                rate = float(mtm.iloc[0]["positive_rate_percent"])
                other = table[~table.index.isin(mtm.index)]
                max_other = float(other["positive_rate_percent"].max()) if not other.empty else float("nan")
                conclusion = "Yes" if other.empty or rate > max_other else "No"
                return CopilotResponse(
                    f"{conclusion}. Month-to-month customers have an observed churn rate of {rate:.2f}% in the active dataset"
                    + (f", compared with a highest non-month-to-month rate of {max_other:.2f}%. " if not np.isnan(max_other) else ". ")
                    + "This is an association, not proof that the contract causes churn. " + source,
                    table=table,
                    interpreted_question=q,
                    context={"topic": "target_relationship", "target_column": target_column, "group_col": contract},
                )

    # Recommendation-limitations follow-up must answer the limitation itself,
    # not repeat the previous recommendation.
    if bool(tokens & {"limitation", "limitations"}) and "recommendation" in tokens:
        limitation_rows = pd.DataFrame([
            {"limitation": "Association is not causation", "meaning": "Observed churn differences and SHAP explain model behaviour; they do not prove an intervention will change churn."},
            {"limitation": "Model error", "meaning": "The preferred model has false positives and false negatives, so recommendations can be based on imperfect risk estimates."},
            {"limitation": "Dataset scope", "meaning": "The recommendation uses the uploaded Telco dataset and may not generalise to other populations or future periods."},
            {"limitation": "Human/business context", "meaning": "Current customer circumstances, feasibility, fairness, consent and business policy still require human review."},
        ])
        return CopilotResponse(
            "The recommendation has four main limitations: it is based on association/model evidence rather than causal proof; the model can make errors; the evidence is limited to the uploaded dataset; and a human must check current customer/business context before acting. " + source,
            table=limitation_rows,
            interpreted_question=q,
            safety_warning="Do not treat a recommendation as an automatic customer decision.",
            context={"topic": "recommendation_limitations", "target_column": target_column},
        )

    # If the user asks what the business should CONSIDER because short tenure is
    # a driver, answer with action guidance first and include the relationship
    # table as evidence. This must precede the generic tenure relationship route.
    if target_column and "tenure" in tokens and "driver" in tokens and bool(tokens & {"business", "consider", "should"}):
        tenure_col = next((c for c in df.columns if _compact(c) == "tenure"), None)
        rel = _numeric_target_relationship_response(f"how does {tenure_col or 'tenure'} relate to {target_column}", df, target_column, positive_label, dataset_name) if tenure_col else None
        return CopilotResponse(
            "If short tenure is an important churn driver, consider strengthening onboarding, early-life support and proactive engagement for newer customers. Use the observed tenure/churn relationship to prioritise human review, but do not assume tenure itself causes churn or automatically target every new customer. " + source,
            table=rel.table if rel is not None else None,
            chart=rel.chart if rel is not None else None,
            interpreted_question=q,
            context={"topic": "recommendation", "target_column": target_column, "rank_col": tenure_col},
        )

    # Direct shorter-tenure comparison.
    if target_column and "tenure" in tokens and bool(tokens & {"short", "shorter", "new", "newer"}) and "churn" in tokens:
        tenure_col = next((c for c in df.columns if _compact(c) == "tenure"), None)
        canonical_q = f"how does {tenure_col or 'tenure'} relate to {target_column}"
        rel = _numeric_target_relationship_response(canonical_q, df, target_column, positive_label, dataset_name)
        if rel is not None and rel.table is not None and not rel.table.empty:
            ordered = rel.table.sort_values("positive_rate_percent", ascending=False)
            hi = ordered.iloc[0]
            lo = ordered.iloc[-1]
            answer = (
                f"Yes. In the active dataset, the shortest tenure band `{hi.get('band', hi.iloc[0])}` has an observed churn/positive rate "
                f"of {float(hi['positive_rate_percent']):.2f}%, compared with {float(lo['positive_rate_percent']):.2f}% in the lowest-risk tenure band. "
                "Customers with shorter tenure therefore appear more likely to churn in this dataset, but this is an association rather than proof that short tenure causes churn. "
                + source
            )
            return CopilotResponse(
                answer,
                table=rel.table,
                chart=rel.chart,
                interpreted_question=q,
                context={"topic": "target_relationship", "target_column": target_column, "rank_col": tenure_col},
            )

    # Compare monthly charges between the two target classes using descriptive
    # statistics, rather than returning the overall churn percentage.
    if target_column and target_column in df.columns and "monthlycharges" in compact and bool(tokens & {"differ", "difference", "between"}) and "churn" in tokens:
        mc = next((c for c in df.columns if _compact(c) == "monthlycharges"), None)
        if mc:
            tmp = df[[mc, target_column]].copy()
            tmp[mc] = pd.to_numeric(tmp[mc], errors="coerce")
            table = tmp.dropna().groupby(target_column)[mc].agg(["count", "mean", "median", "min", "max"]).reset_index()
            table[["mean", "median", "min", "max"]] = table[["mean", "median", "min", "max"]].round(2)
            return CopilotResponse(
                f"Monthly charges differ descriptively between the churn classes; the table shows count, mean, median and range for each class. This is association evidence, not causality. {source}",
                table=table,
                interpreted_question=q,
                context={"topic": "target_relationship", "target_column": target_column, "rank_col": mc},
            )

    # Dataset-level categorical/numeric relationships are different from SHAP
    # model importance, so return descriptive association rankings here.
    if "features" in tokens and "churn" in tokens and bool(tokens & {"categorical", "numeric"}) and "shap" not in tokens:
        kind = "categorical" if "categorical" in tokens else "numeric"
        table = _feature_relationship_summary(df, target_column, positive_label, kind=kind)
        if not table.empty:
            top = table.iloc[0]
            return CopilotResponse(
                f"The strongest descriptive {kind} relationship in this ranking is `{top['feature']}` with an observed positive-rate spread of {float(top['positive_rate_spread_percent_points']):.2f} percentage points across groups/bands. This is exploratory association evidence, not causality. {source}",
                table=table,
                interpreted_question=q,
                context={"topic": "feature_relationships", "target_column": target_column},
            )

    # Confusion-matrix interpretation and exact counts.
    if "confusion" in tokens or bool(tokens & {"positives", "negatives"}) and bool(tokens & {"true", "false"}):
        table, vals = _confusion_evidence(model_output)
        if table is None or vals is None:
            return CopilotResponse(
                "A confusion matrix is not available yet. Train the classification models first, then ask again. " + source,
                interpreted_question=q,
                context={"topic": "model", "target_column": target_column, "evidence_missing": True},
            )
        if "true" in tokens and "positives" in tokens:
            answer = f"There are **{vals['tp']:,} true positives**: actual churn cases correctly predicted as churn. "
        elif "true" in tokens and "negatives" in tokens:
            answer = f"There are **{vals['tn']:,} true negatives**: actual non-churn cases correctly predicted as non-churn. "
        elif "false" in tokens and "positives" in tokens:
            answer = f"There are **{vals['fp']:,} false positives**: non-churn cases incorrectly flagged as churn. "
        elif "false" in tokens and "negatives" in tokens:
            answer = f"There are **{vals['fn']:,} false negatives**: churn cases incorrectly predicted as non-churn. "
        else:
            answer = (
                f"The confusion matrix contains TN={vals['tn']:,}, FP={vals['fp']:,}, FN={vals['fn']:,} and TP={vals['tp']:,}. "
                "True positives/negatives are correct classifications; false positives/negatives are model errors. "
            )
        return CopilotResponse(
            answer + source,
            table=table,
            interpreted_question=q,
            context={"topic": "confusion_matrix", "target_column": target_column},
        )

    # Explicit model list.
    if bool(tokens & {"model", "models"}) and "trained" in tokens and bool(tokens & {"machine", "learning"}) and model_output is not None and getattr(model_output, "leaderboard", None) is not None:
        lb = model_output.leaderboard.copy()
        names = lb["model"].astype(str).tolist() if not lb.empty else []
        return CopilotResponse(
            "The assessed classification models trained are: " + ", ".join(names) + ". " + source,
            table=lb,
            interpreted_question=q,
            context={"topic": "model", "target_column": target_column},
        )

    # Does the F1-selected model win every metric? It usually need not.
    if "preferred" in tokens and "model" in tokens and "every" in tokens and "metric" in tokens and model_output is not None and getattr(model_output, "leaderboard", None) is not None:
        lb = model_output.leaderboard.copy()
        best_name = getattr(getattr(model_output, "best_result", None), "model_name", "preferred model")
        winners = []
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if metric in lb.columns:
                r = lb.sort_values(metric, ascending=False).iloc[0]
                winners.append({"metric": metric, "best_model": r["model"], "best_score": round(float(r[metric]), 4), "preferred_model_wins": str(r["model"]) == str(best_name)})
        table = pd.DataFrame(winners)
        all_win = bool(table["preferred_model_wins"].all()) if not table.empty else False
        return CopilotResponse(
            ("Yes" if all_win else "No") + f". The preferred model `{best_name}` is selected by the F1-first rule; it does not need to be best on every metric. The table shows the winner for each metric. {source}",
            table=table,
            interpreted_question=q,
            context={"topic": "model", "target_column": target_column},
        )

    # SHAP concepts, causality and limitations must be answered conceptually,
    # not by returning a generic feature-importance list.
    shap_question = "shap" in tokens or "featureimportance" in compact or ("feature" in tokens and "importance" in tokens)
    if shap_question and ("difference" in tokens and "global" in tokens and "local" in tokens):
        table = pd.DataFrame([
            {"explanation_scope": "Global SHAP", "meaning": "Summarises which features influence model behaviour across many records."},
            {"explanation_scope": "Local SHAP", "meaning": "Shows how feature contributions push one selected record's prediction up or down."},
        ])
        return CopilotResponse(
            "Global SHAP explains model behaviour across the dataset/sample, while local SHAP explains the contribution of features for one selected prediction. Neither proves causation. " + source,
            table=table,
            interpreted_question=q,
            context={"topic": "shap_concept", "target_column": target_column},
        )
    if shap_question and "global" in tokens and bool(tokens & {"show", "shows", "explanation"}):
        gi = getattr(explanation_output, "global_importance", None) if explanation_output is not None else None
        if gi is not None and not gi.empty:
            top = gi.head(10)
            names = ", ".join(map(str, top["feature"].head(5).tolist())) if "feature" in top.columns else "the highest-ranked features"
            return CopilotResponse(
                f"The global SHAP explanation ranks the features that most strongly influence the model across the analysed sample. The leading features include {names}. Global importance shows magnitude of influence overall, not direction or causality for every customer. {source}",
                table=top,
                interpreted_question=q,
                context={"topic": "drivers", "target_column": target_column},
            )
    if shap_question and "useful" in tokens:
        return CopilotResponse(
            "SHAP is useful here because it adds transparent explanation evidence to model predictions: global SHAP identifies important drivers across the model, and local SHAP explains an individual prediction. This helps the Copilot translate model output into reviewable business language while keeping human review central. SHAP still does not prove causation. " + source,
            interpreted_question=q,
            context={"topic": "shap_concept", "target_column": target_column},
        )
    if (shap_question or "causes" in tokens or "causation" in tokens) and bool(tokens & {"prove", "causes", "cause", "causal", "causation"}):
        return CopilotResponse(
            "No. SHAP or feature importance cannot prove that a feature **causes** churn. It explains how the trained model used observed feature patterns. Establishing causality would require an appropriate causal research design and additional evidence. " + source,
            interpreted_question=q,
            safety_warning="Do not turn SHAP association/explanation evidence into a causal claim.",
            context={"topic": "shap_causality", "target_column": target_column},
        )
    if shap_question and bool(tokens & {"limitation", "limitations", "interpreting", "interpret"}):
        table = pd.DataFrame([
            {"limitation": "Not causality", "meaning": "SHAP explains model behaviour; it does not prove a feature causes churn."},
            {"limitation": "Model dependence", "meaning": "The explanation is only as valid as the trained model and its preprocessing."},
            {"limitation": "Data dependence", "meaning": "Results reflect the uploaded dataset and may not generalise to new populations."},
            {"limitation": "Correlated features", "meaning": "Attribution can be difficult to interpret when predictors are correlated."},
            {"limitation": "Local versus global", "meaning": "A global driver need not push every individual prediction in the same direction."},
        ])
        return CopilotResponse(
            "When interpreting SHAP, treat it as model-explanation evidence rather than causal proof. Check model quality, data representativeness, correlated features and whether the explanation is global or local before acting. " + source,
            table=table,
            interpreted_question=q,
            context={"topic": "shap_limitations", "target_column": target_column},
        )

    # Local/current-customer probability and classification.
    local_customer_question = bool(tokens & {"customer", "record", "case"}) and bool(tokens & {"predicted", "prediction", "risk", "churn"})
    if (
        local_customer_question
        and ("what" in tokens and "risk" in tokens or ("is" in tokens and "predicted" in tokens))
        and "why" not in tokens
        and not bool(tokens & {"factor", "factors", "feature", "features", "increase", "increasing", "reduce", "reduces", "decrease", "decreasing"})
        and "explain" not in tokens
    ):
        label, prob = _local_prediction_values(model_output, explanation_output)
        if label is None and prob is None:
            return CopilotResponse(
                "No current-record prediction evidence is available. Select/generate an individual prediction and local explanation first. " + source,
                interpreted_question=q,
                context={"topic": "local_prediction", "target_column": target_column, "evidence_missing": True},
            )
        pos = positive_label if positive_label is not None else getattr(model_output, "positive_label", None)
        predicted_positive = str(label) == str(pos) if label is not None and pos is not None else (prob is not None and prob >= 0.5)
        risk_text = f" The estimated positive-class/churn probability is {prob*100:.2f}%." if prob is not None else ""
        return CopilotResponse(
            f"The currently explained record is predicted as **{label}**. {('This is the churn/positive class.' if predicted_positive else 'This is the non-churn/negative class.')} {risk_text} This is a probability-based model output, not a guarantee. {source}",
            table=pd.DataFrame([{"predicted_label": label, "positive_class": pos, "predicted_probability": round(prob, 6) if prob is not None else None}]),
            interpreted_question=q,
            context={"topic": "local_prediction", "target_column": target_column},
        )

    # Local positive drivers and single strongest local contributor.
    if bool(tokens & {"customer", "record", "case"}) and explanation_output is not None and getattr(explanation_output, "local_importance", None) is not None:
        local = explanation_output.local_importance.copy()
        if "contribution" in local.columns:
            local["contribution"] = pd.to_numeric(local["contribution"], errors="coerce")
        if bool(tokens & {"increasing", "increase", "increases"}) and "risk" in tokens:
            inc = local[local["contribution"] > 0].sort_values("contribution", ascending=False) if "contribution" in local.columns else pd.DataFrame()
            if not inc.empty:
                return CopilotResponse(
                    "For the currently explained customer, the table lists the signed local SHAP factors that increase the model's churn-risk prediction, ordered from strongest upward contribution. These are model contributions, not causal guarantees. " + source,
                    table=inc.head(10),
                    interpreted_question=q,
                    context={"topic": "local_prediction_explanation", "target_column": target_column},
                )
        if "feature" in tokens and bool(tokens & {"most", "contributed", "contribute"}) and not local.empty:
            sort_col = "absolute_contribution" if "absolute_contribution" in local.columns else "contribution"
            row = local.sort_values(sort_col, ascending=False).iloc[0]
            feature = row.get("feature", "feature")
            contribution = row.get("contribution", np.nan)
            direction = "increased" if pd.notna(contribution) and float(contribution) > 0 else "reduced" if pd.notna(contribution) and float(contribution) < 0 else "influenced"
            return CopilotResponse(
                f"The strongest local contributor for the currently explained customer is **{feature}**; it {direction} the model output for this record. {source}",
                table=local.head(10),
                interpreted_question=q,
                context={"topic": "local_prediction_explanation", "target_column": target_column},
            )

    if bool(tokens & {"customer", "record", "case"}) and bool(tokens & {"action", "considered", "consider"}):
        local = getattr(explanation_output, "local_importance", None) if explanation_output is not None else None
        if local is not None and not local.empty:
            label, prob = _local_prediction_values(model_output, explanation_output)
            pos = positive_label if positive_label is not None else getattr(model_output, "positive_label", None)
            predicted_positive = str(label) == str(pos) if label is not None and pos is not None else (prob is not None and prob >= 0.5)
            work = local.copy()
            if "contribution" in work.columns:
                work["contribution"] = pd.to_numeric(work["contribution"], errors="coerce")
                positive_drivers = work[work["contribution"] > 0].sort_values("contribution", ascending=False)
            else:
                positive_drivers = pd.DataFrame()

            if not predicted_positive:
                prob_text = f" ({prob*100:.2f}% estimated churn probability)" if prob is not None else ""
                answer = (
                    f"The currently explained customer is predicted as **{label}**, the non-churn/negative class{prob_text}. "
                    "The model therefore does not justify treating this customer as high risk or triggering a churn-specific intervention automatically. "
                    "A reasonable action is routine monitoring and, if a human reviewer has an independent business reason to investigate, review the strongest positive local SHAP contributors shown in the table. "
                )
            else:
                recommendation = _recommend_from_drivers(explanation_output, positive_only=True)
                prob_text = f" ({prob*100:.2f}% estimated churn probability)" if prob is not None else ""
                answer = (
                    f"The currently explained customer is predicted as **{label}**, the churn/positive class{prob_text}. "
                    "Use the prediction to prioritise human review, then consider only actions that correspond to the customer's actual current context and the positive local SHAP contributors. "
                    + recommendation + " "
                )
            return CopilotResponse(
                answer + source,
                table=positive_drivers.head(10) if not positive_drivers.empty else local.head(10),
                interpreted_question=q,
                safety_warning="Do not automatically contact, penalise, cancel or change a customer plan solely because of the model output.",
                context={"topic": "recommendation", "target_column": target_column, "predicted_label": label, "predicted_probability": prob},
            )
        return CopilotResponse(
            "I cannot make a customer-specific recommendation without a current local prediction/explanation. Select a record and generate its local explanation first. " + source,
            interpreted_question=q,
            context={"topic": "supported_limitation", "target_column": target_column, "evidence_missing": True},
        )

    if "monthtomonth" in compact and bool(tokens & {"action", "consider", "considered"}):
        contract = next((c for c in df.columns if _compact(c) == "contract"), None)
        table = pd.DataFrame()
        evidence = ""
        if contract and target_column and target_column in df.columns:
            pos = positive_label if positive_label is not None else infer_positive_label(df[target_column])
            tmp = df[[contract, target_column]].dropna().copy()
            tmp["__positive"] = (tmp[target_column].astype(str) == str(pos)).astype(int)
            table = tmp.groupby(contract)["__positive"].agg(["mean", "count"]).reset_index()
            table["positive_rate_percent"] = (table["mean"] * 100).round(2)
            table = table.drop(columns=["mean"]).sort_values("positive_rate_percent", ascending=False)
            mtm = table[table[contract].astype(str).str.lower().str.contains("month-to-month", regex=False)]
            if not mtm.empty:
                evidence = f" In this dataset, month-to-month customers have a {float(mtm.iloc[0]['positive_rate_percent']):.2f}% observed churn rate."
        return CopilotResponse(
            "For month-to-month customers, consider human-reviewed loyalty/longer-contract incentives, clearer value communication and proactive retention support where appropriate."
            + evidence + " Do not automatically change contracts; the observed relationship is not causal proof. " + source,
            table=table if not table.empty else None,
            interpreted_question=q,
            context={"topic": "recommendation", "target_column": target_column, "group_col": contract},
        )

    # Human review / final-decision / automatic-adverse-action safeguards.
    if "human" in tokens and "review" in tokens and bool(tokens & {"recommendation", "action", "taking", "before"}):
        return CopilotResponse(
            "Yes. A human should review the recommendation before any customer action. The model and Copilot provide decision support only; the reviewer must check current customer context, feasibility, fairness and potential consequences. " + source,
            interpreted_question=q,
            safety_warning="Human review is required before action.",
            context={"topic": "human_review", "target_column": target_column},
        )
    if (("automatically" in tokens or "automatic" in tokens) and bool(tokens & {"cancel", "cancellation", "customers"})):
        return CopilotResponse(
            "No. The business should not automatically cancel or penalise customers because a churn model marks them as high risk. Use the prediction only to prioritise human-reviewed retention investigation. " + source,
            interpreted_question=q,
            safety_warning="Automatic adverse customer action from model output is refused.",
            context={"topic": "safe_refusal", "target_column": target_column},
        )
    if "final" in tokens and "decision" in tokens and "model" in tokens and bool(tokens & {"only", "using"}):
        return CopilotResponse(
            "No. A final business/customer decision should not be made using only this model. Combine the prediction with current business/customer context, policy checks and accountable human review. " + source,
            interpreted_question=q,
            safety_warning="The model is decision support, not an autonomous decision-maker.",
            context={"topic": "safe_refusal", "target_column": target_column},
        )

    # Feature-specific business guidance even when the user says "consider"
    # rather than the literal words recommend/recommendation/action.
    business_consider = bool(tokens & {"business", "consider", "should"})
    if business_consider and "tenure" in tokens and "driver" in tokens:
        tenure = next((c for c in df.columns if _compact(c) == "tenure"), None)
        rel = _numeric_target_relationship_response(q, df, target_column, positive_label, dataset_name) if tenure and target_column else None
        return CopilotResponse(
            "If short tenure is an important churn driver, consider strengthening onboarding, early-life support and proactive engagement for newer customers. Use the observed tenure/churn evidence to prioritise review; do not assume tenure itself causes churn. " + source,
            table=rel.table if rel is not None else None,
            interpreted_question=q,
            context={"topic": "recommendation", "target_column": target_column, "rank_col": tenure},
        )
    if business_consider and "monthlycharges" in compact and "churn" in tokens:
        mc = next((c for c in df.columns if _compact(c) == "monthlycharges"), None)
        rel = _numeric_target_relationship_response(q, df, target_column, positive_label, dataset_name) if mc and target_column else None
        return CopilotResponse(
            "If monthly charges are associated with churn, consider human-reviewed package-fit checks, price-sensitivity analysis, billing clarity and targeted retention offers where appropriate. The dataset association does not prove price causes churn. " + source,
            table=rel.table if rel is not None else None,
            interpreted_question=q,
            context={"topic": "recommendation", "target_column": target_column, "rank_col": mc},
        )

    # Dataset/readiness suitability and exported readiness evidence.
    readiness_intent = (
        "readiness" in tokens
        or "datareadiness" in compact
        or ("dataset" in tokens and bool(tokens & {"suitable", "ready", "modelling", "modeling", "good"}))
        or ("good" in tokens and bool(tokens & {"modelling", "modeling"}))
    )
    if readiness_intent:
        rep = build_readiness_report(df, dataset_name=dataset_name, target_column=target_column)
        table = rep.gates_dataframe()
        failed = table.loc[table["status"] == "FAIL", "gate"].tolist() if not table.empty else []
        warned = table.loc[table["status"] == "WARNING", "gate"].tolist() if not table.empty else []
        if rep.overall_status == "PASS":
            meaning = "All core dissertation readiness gates pass for the selected target and current cleaned dataset."
        elif rep.overall_status == "WARNING":
            meaning = "The workflow can continue only with visible limitations because one or more core readiness gates are warnings."
        else:
            meaning = "One or more core readiness gates fail, so reliable automatic modelling/answering should be restricted until the issue is fixed."
        details = []
        if warned:
            details.append("Warnings: " + ", ".join(warned))
        if failed:
            details.append("Failures: " + ", ".join(failed))
        answer = (
            f"Data-readiness result: **{rep.overall_status}** with quality score {rep.quality_score:.1f}/100. "
            f"{meaning} {' '.join(details)} {source}"
        )
        return CopilotResponse(
            answer,
            table=table,
            interpreted_question=q,
            safety_warning="Readiness is a gate for decision support. WARNING requires visible limitations; FAIL should restrict modelling or unsupported Copilot answers.",
            context={"topic": "readiness_evidence", "target_column": target_column},
        )

    # Explicit export capability question. Keep future semantic-memory output
    # out of the core list because it is disabled by default.
    if "export" in compact or "download" in compact:
        if tokens & {"analysis", "result", "results", "evidence", "report", "chat", "answers", "exported"} or "whatcanbeexported" in compact:
            table = pd.DataFrame([
                {"export_item": "Full results ZIP + HTML/Markdown report", "purpose": "one-click dissertation evidence package"},
                {"export_item": "Cleaned dataset / cleaning report", "purpose": "reproducibility and data-preparation evidence"},
                {"export_item": "Readiness gates CSV", "purpose": "PASS/WARNING/FAIL evidence"},
                {"export_item": "Model leaderboard + confusion matrix", "purpose": "model-performance evidence"},
                {"export_item": "Global and local explanation-driver CSVs", "purpose": "SHAP/fallback explanation evidence"},
                {"export_item": "Copilot chat answers with grounding/confidence fields", "purpose": "RQ2 answer-quality evidence"},
                {"export_item": "Session audit + runtime/process evidence", "purpose": "traceability and performance evidence"},
            ])
            return CopilotResponse(
                "The core analysis can be exported as a full results ZIP plus separate dataset, readiness, model, explanation, Copilot, audit and runtime evidence files. Optional future semantic-memory events are exported only if that experimental feature is manually enabled. " + source,
                table=table,
                interpreted_question=q,
                context={"topic": "export_evidence"},
            )

    # Evidence/grounding questions about the current answer architecture.
    if (("evidence" in tokens and bool(tokens & {"use", "used", "show", "what", "supports", "support"}) and "model" not in tokens)
        or ("grounded" in tokens and "dataset" in tokens) or "whyevidence" in compact):
        # When this is a follow-up to a recommendation, preserve the actual
        # recommendation context instead of returning only a generic provenance
        # statement.
        if previous_context and str(previous_context.get("topic", "")).lower() == "recommendation":
            group_col = previous_context.get("group_col")
            rank_col = previous_context.get("rank_col")
            if group_col and group_col in df.columns and target_column and target_column in df.columns:
                pos = positive_label if positive_label is not None else infer_positive_label(df[target_column])
                tmp = df[[group_col, target_column]].dropna().copy()
                tmp["__positive"] = (tmp[target_column].astype(str) == str(pos)).astype(int)
                rec_table = tmp.groupby(group_col)["__positive"].agg(["mean", "count"]).reset_index()
                rec_table["positive_rate_percent"] = (rec_table["mean"] * 100).round(2)
                rec_table = rec_table.drop(columns=["mean"]).sort_values("positive_rate_percent", ascending=False)
                if _compact(str(group_col)) == "contract":
                    mtm = rec_table[rec_table[group_col].astype(str).str.lower().str.contains("month-to-month", regex=False)]
                    if not mtm.empty:
                        rate = float(mtm.iloc[0]["positive_rate_percent"])
                        return CopilotResponse(
                            f"The recommendation is supported by the immediately preceding `{group_col}` evidence: month-to-month customers have an observed churn rate of {rate:.2f}% in the active dataset. "
                            "This is descriptive association evidence, so the action remains advisory and requires human review. " + source,
                            table=rec_table,
                            interpreted_question=q,
                            context={"topic": "grounding_evidence", "target_column": target_column, "group_col": group_col},
                        )
            if rank_col and rank_col in df.columns and target_column and target_column in df.columns:
                rel = _numeric_target_relationship_response(f"how does {rank_col} relate to {target_column}", df, target_column, positive_label, dataset_name)
                if rel is not None:
                    return CopilotResponse(
                        f"The recommendation is supported by the preceding `{rank_col}`-versus-`{target_column}` relationship evidence shown in the table. "
                        "It is an observed association, not proof of causation, so human review is still required. " + source,
                        table=rel.table,
                        chart=rel.chart,
                        interpreted_question=q,
                        context={"topic": "grounding_evidence", "target_column": target_column, "rank_col": rank_col},
                    )

        rows = [
            {"evidence_source": "active cleaned dataset", "availability": "AVAILABLE", "detail": f"{len(df):,} rows × {df.shape[1]:,} columns"},
            {"evidence_source": "dissertation readiness gates", "availability": "AVAILABLE", "detail": "PASS/WARNING/FAIL validation"},
            {"evidence_source": "short-term session context", "availability": "AVAILABLE" if previous_context else "NO PRIOR CONTEXT", "detail": str((previous_context or {}).get("topic", ""))},
        ]
        if model_output is not None and getattr(model_output, "best_result", None) is not None:
            rows.append({"evidence_source": "trained model metrics", "availability": "AVAILABLE", "detail": getattr(model_output.best_result, "model_name", "model")})
        else:
            rows.append({"evidence_source": "trained model metrics", "availability": "NOT AVAILABLE", "detail": "train a model for prediction evidence"})
        if explanation_output is not None and getattr(explanation_output, "global_importance", None) is not None:
            rows.append({"evidence_source": "SHAP/fallback explanation", "availability": "AVAILABLE", "detail": getattr(explanation_output, "method", "explanation")})
        else:
            rows.append({"evidence_source": "SHAP/fallback explanation", "availability": "NOT AVAILABLE", "detail": "generate explanation for driver evidence"})
        table = pd.DataFrame(rows)
        return CopilotResponse(
            "The answer is grounded because the controlled Copilot is restricted to the active cleaned dataset, readiness checks, available model metrics, SHAP/fallback explanation artefacts and current-session context/rules. It does not invent external customer or market evidence. " + source,
            table=table,
            interpreted_question=q,
            context={"topic": "grounding_evidence", "target_column": target_column},
        )

    # Prediction limitations.
    if "limitation" in tokens or "limitations" in tokens:
        if tokens & {"prediction", "model", "churn", "result"}:
            table = pd.DataFrame([
                {"limitation": "Probabilistic prediction", "meaning": "A churn score is not a guarantee about an individual customer."},
                {"limitation": "Dataset dependence", "meaning": "Results reflect the uploaded dataset, preprocessing and train/test split."},
                {"limitation": "False positives/false negatives", "meaning": "Precision and recall show that some customers will be misclassified."},
                {"limitation": "SHAP is explanatory evidence, not causality", "meaning": "A driver can influence model output without proving a causal business mechanism."},
                {"limitation": "Missing external context", "meaning": "Complaints, competitor offers, current service incidents and other unconnected evidence are not known."},
                {"limitation": "Human review required", "meaning": "Retention/customer decisions should not be automated from the model alone."},
            ])
            return CopilotResponse(
                "The prediction has important limitations: it is probabilistic, can make false-positive and false-negative errors, depends on the current dataset/preprocessing, and SHAP explains model behaviour rather than proving causation. External customer context is absent unless it is in the uploaded data. " + source,
                table=table,
                interpreted_question=q,
                context={"topic": "prediction_limitations", "target_column": target_column},
            )

    # Strong safety refusals for over-reliance / unsupported evidence.
    if ("guarantee" in tokens and "churn" in tokens) or "guaranteethiscustomerwillchurn" in compact:
        return CopilotResponse(
            "No. The model cannot guarantee that a customer will churn. It estimates risk from patterns in the training data, and the prediction can be wrong. Treat it as a prioritisation signal for human review, not a certainty. " + source,
            interpreted_question=q,
            safety_warning="No churn prediction is guaranteed. Human review and current customer context are required.",
            context={"topic": "safe_refusal", "target_column": target_column},
        )
    if ("replace" in tokens and bool(tokens & {"manager", "human", "decision"})) or "replacemanagersdecision" in compact:
        return CopilotResponse(
            "No. This model and Copilot are designed to support a manager or analyst, not replace their decision. The human reviewer remains responsible for checking context, fairness, feasibility and consequences before action. " + source,
            interpreted_question=q,
            safety_warning="Human decision ownership must be retained; do not delegate final customer action to the model.",
            context={"topic": "safe_refusal", "target_column": target_column},
        )
    if "automatic" in tokens and bool(tokens & {"cancellation", "cancel", "cancelation"}):
        return CopilotResponse(
            "No. Automatic cancellation should not be recommended from churn risk. A high-risk prediction indicates that a customer may need human-reviewed retention investigation; it is not evidence to cancel or penalise the customer. " + source,
            interpreted_question=q,
            safety_warning="Automatic cancellation or adverse customer action from model output is refused.",
            context={"topic": "safe_refusal", "target_column": target_column},
        )
    if "external" in tokens and bool(tokens & {"complaint", "complaints", "customer"}):
        table = pd.DataFrame([{
            "requested_evidence": "external customer complaints",
            "status": "NOT AVAILABLE",
            "safe_next_step": "Upload/connect the complaint data, then re-run the analysis.",
        }])
        return CopilotResponse(
            "I cannot use external customer complaints that are not present in or connected to the current evidence. I would need that complaint source to be uploaded/connected before making claims from it. " + source,
            table=table,
            interpreted_question=q,
            safety_warning="Unsupported external evidence was refused rather than inferred.",
            context={"topic": "safe_refusal"},
        )

    # Local/current-record prediction explanation. This must run before the
    # generic target-distribution intent so 'why will this customer churn?' is
    # never answered with only the overall churn rate.
    why_churn = (
        "why" in tokens
        and ("churn" in tokens or "predicted" in tokens or "prediction" in tokens)
        and bool(tokens & {"customer", "customers", "record", "case", "predicted"})
    )
    if why_churn:
        if explanation_output is not None and getattr(explanation_output, "local_importance", None) is not None and not explanation_output.local_importance.empty:
            local = explanation_output.local_importance.copy().head(10)
            sentences = top_driver_sentences(local, top_n=5)
            method = getattr(explanation_output, "method", "SHAP/fallback")
            label, prob = _local_prediction_values(model_output, explanation_output)
            pos = positive_label if positive_label is not None else getattr(model_output, "positive_label", None)
            predicted_positive = str(label) == str(pos) if label is not None and pos is not None else (prob is not None and prob >= 0.5)
            prob_text = f" with an estimated churn probability of {prob*100:.2f}%" if prob is not None else ""
            if predicted_positive:
                premise = f"The currently explained record is predicted as **{label}**, the churn/positive class{prob_text}. "
            else:
                premise = (
                    f"The currently explained record is actually predicted as **{label}**, the non-churn/negative class{prob_text}; "
                    "so it is **not** currently predicted to churn. The local explanation below shows why the model output is lower and which factors still push risk upward. "
                )
            answer = (
                premise
                + f"The {method} local explanation identifies the strongest contributors to the model output. "
                + " ".join(sentences)
                + " This explains the model's prediction for the selected record; it does not prove what the customer will actually do. "
                + source
            )
            return CopilotResponse(
                answer,
                table=local,
                interpreted_question=q,
                safety_warning="Local explanation is record-specific and probabilistic. Confirm the correct customer/record and review context before action.",
                context={"topic": "local_prediction_explanation", "target_column": target_column, "predicted_label": label, "predicted_probability": prob},
            )
        return CopilotResponse(
            "I cannot explain an individual churn prediction yet because no local SHAP/fallback explanation is available. Generate/select a record explanation in the Model page and ask again. " + source,
            interpreted_question=q,
            safety_warning="No individual explanation was guessed because the required local evidence is missing.",
            context={"topic": "local_prediction_explanation", "target_column": target_column, "evidence_missing": True},
        )

    # SHAP explanation in plain business language.
    # Default to GLOBAL explanation unless the question explicitly refers to a
    # customer/record/prediction. This avoids silently answering a global SHAP
    # question with one customer's local explanation.
    if ("shap" in tokens and bool(tokens & {"explain", "result", "results", "simple", "business"})) or "simplebusinesslanguage" in compact:
        asks_local = bool(tokens & {"customer", "record", "case", "prediction", "predicted"}) or "thiscustomer" in compact
        if asks_local and explanation_output is not None and getattr(explanation_output, "local_importance", None) is not None and not explanation_output.local_importance.empty:
            local = explanation_output.local_importance.head(8).copy()
            sentences = top_driver_sentences(local, top_n=5)
            label, prob = _local_prediction_values(model_output, explanation_output)
            pred_text = ""
            if label is not None:
                pred_text = f"The selected record is predicted as **{label}**"
                if prob is not None:
                    pred_text += f" with {prob*100:.2f}% estimated churn probability"
                pred_text += ". "
            return CopilotResponse(
                "In simple business language: " + pred_text
                + "local SHAP shows which processed customer features pushed this record's churn-risk prediction up or down. "
                + " ".join(sentences)
                + " The size of a contribution shows influence on this model output, not a causal guarantee. "
                + source,
                table=local,
                interpreted_question=q,
                context={"topic": "shap_explanation", "target_column": target_column},
            )
        if explanation_output is not None and getattr(explanation_output, "global_importance", None) is not None and not explanation_output.global_importance.empty:
            global_imp = explanation_output.global_importance.head(10).copy()
            top_names = ", ".join(map(str, global_imp["feature"].head(5).tolist())) if "feature" in global_imp.columns else "the highest-ranked features"
            return CopilotResponse(
                f"In simple business language: global SHAP shows which features the trained model relies on most across many records. "
                f"The leading drivers are {top_names}. Larger global SHAP importance means the feature has more influence on model predictions overall, "
                "but it does not tell the same direction for every customer and it does not prove causation. " + source,
                table=global_imp,
                interpreted_question=q,
                context={"topic": "shap_explanation", "target_column": target_column},
            )

    # Directional drivers: use local SHAP signs only; never infer direction from
    # absolute global importance.
    direction = None
    asks_directional_features = bool(tokens & {"feature", "features", "driver", "drivers", "factor", "factors"})
    if asks_directional_features and bool(tokens & {"increase", "increases", "higher", "raise", "raises"}) and ("risk" in tokens or "churn" in tokens):
        direction = "increase"
    elif asks_directional_features and bool(tokens & {"reduce", "reduces", "decrease", "decreases", "lower"}) and ("risk" in tokens or "churn" in tokens):
        direction = "reduce"
    if direction:
        local = getattr(explanation_output, "local_importance", None) if explanation_output is not None else None
        if local is not None and not local.empty and "contribution" in local.columns and pd.to_numeric(local["contribution"], errors="coerce").notna().any():
            work = local.copy()
            work["contribution"] = pd.to_numeric(work["contribution"], errors="coerce")
            if direction == "increase":
                work = work[work["contribution"] > 0].sort_values("contribution", ascending=False)
                phrase = "increase the model's predicted churn risk"
            else:
                work = work[work["contribution"] < 0].sort_values("contribution", ascending=True)
                phrase = "reduce the model's predicted churn risk"
            return CopilotResponse(
                f"For the currently explained record, these local SHAP contributions {phrase}. This direction is record-specific; global absolute importance alone does not establish direction. {source}",
                table=work.head(10),
                interpreted_question=q,
                context={"topic": "local_prediction_explanation", "target_column": target_column},
            )
        return CopilotResponse(
            "I cannot safely state increase/decrease direction from the available global importance alone. Direction requires signed local SHAP contributions for a selected record. " + source,
            interpreted_question=q,
            context={"topic": "drivers", "target_column": target_column, "evidence_missing": True},
        )

    # Feature-specific, transparent rule-based recommendations. The answer also
    # returns observed target evidence where possible so the recommendation is
    # auditable instead of being a free-standing business rule.
    if tokens & {"recommend", "recommendation", "action"}:
        if "contract" in tokens and any(_compact(c) == "contract" for c in df.columns):
            contract_col = next(c for c in df.columns if _compact(c) == "contract")
            table = pd.DataFrame()
            evidence_sentence = ""
            if target_column and target_column in df.columns:
                pos = positive_label if positive_label is not None else df[target_column].dropna().value_counts().index[0]
                tmp = df[[contract_col, target_column]].dropna().copy()
                tmp["__positive"] = (tmp[target_column].astype(str) == str(pos)).astype(int)
                table = tmp.groupby(contract_col, dropna=False)["__positive"].agg(["mean", "count"]).reset_index()
                table["positive_rate_percent"] = (table["mean"] * 100).round(2)
                table = table.drop(columns=["mean"]).sort_values("positive_rate_percent", ascending=False)
                if not table.empty:
                    top = table.iloc[0]
                    evidence_sentence = f" In the active data, `{top[contract_col]}` has the highest observed {target_column} positive rate at {top['positive_rate_percent']:.2f}% across {int(top['count']):,} records."
            return CopilotResponse(
                "Contract-related action guidance:" + evidence_sentence + " Consider human-reviewed loyalty/longer-contract incentives and clearer value communication for relevant segments. This is an observed association and decision-support suggestion; do not automatically change a customer's contract. " + source,
                table=table if not table.empty else None,
                interpreted_question=q,
                context={"topic": "recommendation", "target_column": target_column, "group_col": contract_col},
            )
        if ("monthly" in tokens and ("charges" in tokens or "charge" in tokens)) or "monthlycharges" in compact:
            mc = next((c for c in df.columns if _compact(c) == "monthlycharges"), None)
            if mc:
                table = pd.DataFrame()
                evidence_sentence = ""
                if target_column and target_column in df.columns:
                    tmp = df[[mc, target_column]].copy()
                    tmp[mc] = pd.to_numeric(tmp[mc], errors="coerce")
                    tmp = tmp.dropna()
                    pos = positive_label if positive_label is not None else tmp[target_column].value_counts().index[0]
                    if not tmp.empty and tmp[mc].nunique() >= 2:
                        try:
                            tmp["charge_band"] = pd.qcut(tmp[mc], q=min(4, tmp[mc].nunique()), duplicates="drop")
                        except Exception:
                            tmp["charge_band"] = pd.cut(tmp[mc], bins=4, duplicates="drop")
                        tmp["__positive"] = (tmp[target_column].astype(str) == str(pos)).astype(int)
                        table = tmp.groupby("charge_band", observed=False).agg(record_count=(target_column, "size"), positive_rate=("__positive", "mean")).reset_index()
                        table["charge_band"] = table["charge_band"].astype(str)
                        table["positive_rate_percent"] = (table["positive_rate"] * 100).round(2)
                        table = table.drop(columns=["positive_rate"]).sort_values("positive_rate_percent", ascending=False)
                        if not table.empty:
                            top = table.iloc[0]
                            evidence_sentence = f" The highest observed {target_column} positive-rate charge band is `{top['charge_band']}` at {top['positive_rate_percent']:.2f}% across {int(top['record_count']):,} records."
                return CopilotResponse(
                    "Monthly-charge action guidance:" + evidence_sentence + " Review package fit, price sensitivity, discount eligibility and billing clarity for relevant customers. This does not prove that price causes churn and should not trigger automatic offers or plan changes. " + source,
                    table=table if not table.empty else None,
                    interpreted_question=q,
                    context={"topic": "recommendation", "target_column": target_column, "rank_col": mc},
                )

    return None


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

    # Do not fuzzy-normalise normal words into identifier columns (for example
    # "customer" -> "customerid"). ID columns can still be displayed as row
    # context, but they should not drive intent selection.
    id_like_cols = set(_find_id_like_columns(df))
    safe_known_terms = [str(c) for c in df.columns if len(_norm_text(c)) > 2 and c not in id_like_cols]
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

    class_balance_response = _class_balance_response(q, df, dataset_name, target_column, positive_label)
    if class_balance_response is not None:
        class_balance_response.corrections = corrections
        class_balance_response.interpreted_question = q
        return class_balance_response

    system_response = _system_or_safety_response(q, df, dataset_name, target_column, positive_label)
    if system_response is not None:
        system_response.corrections = corrections
        system_response.interpreted_question = q
        return system_response

    reviewer_response = _reviewer_question_response(
        q, df, dataset_name, target_column, positive_label,
        model_output, explanation_output, previous_context,
    )
    if reviewer_response is not None:
        reviewer_response.corrections = corrections
        reviewer_response.interpreted_question = q
        return reviewer_response

    domain_guard = _domain_context_guard(q, df, dataset_name, target_column)
    if domain_guard is not None:
        domain_guard.corrections = corrections
        domain_guard.interpreted_question = q
        return domain_guard

    prediction_evidence_response = _prediction_evidence_response(q, df, dataset_name, target_column, positive_label, model_output, explanation_output)
    if prediction_evidence_response is not None:
        prediction_evidence_response.corrections = corrections
        prediction_evidence_response.interpreted_question = q
        return prediction_evidence_response

    retention_response = _retention_or_churn_action_response(q, df, dataset_name, target_column, positive_label, model_output, explanation_output)
    if retention_response is not None:
        retention_response.corrections = corrections
        retention_response.interpreted_question = q
        return retention_response

    relationship_response = _numeric_target_relationship_response(q, df, target_column, positive_label, dataset_name)
    if relationship_response is not None:
        relationship_response.corrections = corrections
        relationship_response.interpreted_question = q
        return relationship_response

    categorical_relationship_response = _categorical_target_relationship_response(q, df, target_column, positive_label, dataset_name)
    if categorical_relationship_response is not None:
        categorical_relationship_response.corrections = corrections
        categorical_relationship_response.interpreted_question = q
        return categorical_relationship_response

    target_response = _target_column_response(q, df, dataset_name, target_column, positive_label)
    if target_response is not None:
        target_response.corrections = corrections
        target_response.interpreted_question = q
        return target_response

    if _dataset_use_case_intent(q):
        response = _dataset_use_case_response(q, df, dataset_name, target_column)
        response.corrections = corrections
        response.interpreted_question = q
        return response

    # Route broad improvement and technical/service issue questions to feedback
    # evidence when a feedback text column exists. This prevents wrong answers
    # such as customerID frequency tables for "technical issue customer found".
    if _technical_or_service_issue_intent(q) or _generic_improvement_intent(q, df):
        feedback_cols = _feedback_columns(df)
        text_col = feedback_cols[0] if feedback_cols else None
        if text_col:
            table = _feedback_business_action_table(df, text_col, target_column=target_column, positive_label=positive_label)
            if table.empty:
                words = _top_text_words_clean(df, text_col, n=20)
                chart = px.bar(words, x="word", y="count", title=f"Top feedback words in {text_col}") if not words.empty else None
                return CopilotResponse(
                    f"I interpreted this as a feedback/service issue question. I found `{text_col}`, but no predefined issue theme had enough evidence, so I returned cleaned feedback words instead. {source}",
                    table=words,
                    chart=chart,
                    interpreted_question=q,
                    corrections=corrections,
                    safety_warning="This is a text-summary fallback. Review sample feedback rows before deciding actions.",
                    context={"topic": "feedback_themes", "text_col": text_col, "target_column": target_column},
                )
            if _technical_or_service_issue_intent(q):
                table = _prioritise_technical_issue_table(table)
            chart = px.bar(table.head(8), x="theme", y="negative_evidence_rows", hover_data=["evidence_rows", "priority"], title=f"Feedback/service improvement priorities from {text_col}")
            # Use a specific answer sentence for technical/service issue questions.
            if _technical_or_service_issue_intent(q):
                top = table.iloc[0]
                answer = (
                    f"I interpreted this as asking whether customers reported technical or service issues. "
                    f"Using `{text_col}`, the highest-priority issue theme is **{top['theme']}** with "
                    f"{int(top['negative_evidence_rows']):,} concern row(s) from {int(top['evidence_rows']):,} matching row(s). "
                    f"Recommended action: {top['recommended_action']} {source}"
                )
                topic = "feedback_technical_issues"
            else:
                answer = _feedback_business_answer(table, text_col, dataset_name)
                topic = "feedback_business_actions"
            return CopilotResponse(
                answer,
                table=table,
                chart=chart,
                interpreted_question=q,
                corrections=corrections,
                safety_warning="These are decision-support suggestions generated from transparent feedback/theme rules. A human reviewer must check sample records and customer context before acting.",
                context={"topic": topic, "text_col": text_col, "target_column": target_column},
            )

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

    if (tokens & {"missing", "null", "empty", "blank"}) and (tokens & {"duplicate", "duplicates", "duplicated"}):
        missing_count = int(df.isna().sum().sum())
        duplicate_count = int(df.duplicated().sum())
        total_cells = max(int(df.shape[0] * df.shape[1]), 1)
        table = pd.DataFrame([
            {"quality_check": "missing cells", "count": missing_count, "percent": round(missing_count / total_cells * 100, 3)},
            {"quality_check": "duplicate rows", "count": duplicate_count, "percent": round(duplicate_count / max(len(df), 1) * 100, 3)},
        ])
        return CopilotResponse(
            f"The active dataset contains {missing_count:,} missing cell(s) and {duplicate_count:,} duplicate row(s) after current cleaning. {source}",
            table=table,
            interpreted_question=q,
            corrections=corrections,
            context={"topic": "data_quality_summary"},
        )

    if tokens & {"missing", "null", "empty", "blank"}:
        table = missing_value_table(df).head(40)
        total_missing = int(df.isna().sum().sum())
        missing_cols = table.loc[table["missing_count"] > 0, "column"].astype(str).tolist() if "missing_count" in table.columns else []
        if missing_cols:
            col_text = ", ".join(f"`{c}`" for c in missing_cols[:12])
            answer = f"The active dataset contains {total_missing:,} missing cells. Missing values occur in: {col_text}. {source}"
        else:
            answer = f"The active dataset contains {total_missing:,} missing cells and no column currently has a missing value. {source}"
        chart_data = table[table["missing_count"] > 0].head(15)
        chart = px.bar(chart_data, x="column", y="missing_percent", title="Missing values (%)") if not chart_data.empty else None
        return CopilotResponse(answer, table=table, chart=chart, interpreted_question=q, corrections=corrections, context={"topic": "missing"})

    if tokens & {"duplicate", "duplicates"}:
        count = int(df.duplicated().sum())
        return CopilotResponse(f"The active dataset has {count:,} duplicate rows after current cleaning. {source}", interpreted_question=q, corrections=corrections, context={"topic": "duplicates"})

    # 2) Explanation/driver intents must be checked before generic model/performance intents.
    if tokens & {"driver", "drivers", "importance", "features", "feature", "factor", "factors", "riskfactor", "riskfactors", "shap"} or ("risk" in tokens and "factor" in tokens):
        if explanation_output is not None and getattr(explanation_output, "global_importance", None) is not None:
            table = explanation_output.global_importance.head(20)
            chart = px.bar(table.head(15), x="importance", y="feature", orientation="h", title="Top model drivers") if not table.empty and "importance" in table.columns else None
            top_names = ", ".join(map(str, table["feature"].head(5).tolist())) if "feature" in table.columns else "the highest-ranked features"
            return CopilotResponse(
                f"The strongest global model drivers include {top_names}. The table gives the full ranked explanation evidence. Global importance indicates influence magnitude, not causal effect or customer-specific direction. {source}",
                table=table,
                chart=chart,
                interpreted_question=q,
                corrections=corrections,
                context={"topic": "drivers", "target_column": target_column},
            )
        if model_output is not None and getattr(model_output, "best_result", None) is not None:
            return CopilotResponse("A model has been trained, but explanation drivers have not been generated yet. Open the Model page and click the SHAP/fallback explanation step, then ask this again. " + _model_summary(model_output) + " " + source, interpreted_question=q, corrections=corrections, context={"topic": "drivers", "target_column": target_column})
        return CopilotResponse("Train a model and generate an explanation first, then I can show SHAP/fallback drivers. " + source, interpreted_question=q, corrections=corrections, context={"topic": "drivers"})

    # 2) Model/explanation intents.
    if (tokens & {"model", "models", "performance", "accuracy", "f1", "recall", "precision", "auc"}
        or "roc auc" in q or "roc-auc" in q or "good enough" in q or "decision support" in q
        or "why is that model better" in q
        or ("compare" in tokens and bool(tokens & {"logistic", "random", "gradient", "mlp"}))):
        if model_output is not None and getattr(model_output, "leaderboard", None) is not None:
            model_answer = _metric_specific_model_answer(q, model_output)
            return CopilotResponse(model_answer + " " + source, table=model_output.leaderboard, interpreted_question=q, corrections=corrections, context={"topic": "model", "target_column": getattr(model_output, "target_column", target_column)})
        return CopilotResponse(_model_summary(model_output) + " " + source, interpreted_question=q, corrections=corrections, context={"topic": "model"})

    if tokens & {"driver", "drivers", "importance", "features", "feature", "factor", "factors", "riskfactor", "riskfactors", "shap"} or ("risk" in tokens and "factor" in tokens):
        if explanation_output is not None and getattr(explanation_output, "global_importance", None) is not None:
            table = explanation_output.global_importance.head(20)
            chart = px.bar(table.head(15), x="importance", y="feature", orientation="h", title="Top model drivers") if not table.empty and "importance" in table.columns else None
            top_names = ", ".join(map(str, table["feature"].head(5).tolist())) if "feature" in table.columns else "the highest-ranked features"
            return CopilotResponse(
                f"The strongest global model drivers include {top_names}. The table gives the full ranked explanation evidence. Global importance indicates influence magnitude, not causal effect or customer-specific direction. {source}",
                table=table,
                chart=chart,
                interpreted_question=q,
                corrections=corrections,
                context={"topic": "drivers", "target_column": target_column},
            )
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
            f"{int(top['negative_evidence_rows']):,} negative/concern evidence row(s). "
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

    # 6) Highest target-risk segments across categorical columns.
    segment_response = _highest_target_segments(df, target_column, positive_label, dataset_name, q)
    if segment_response is not None:
        segment_response.corrections = corrections
        segment_response.interpreted_question = q
        return segment_response

    # 7) Target distribution.
    if target_column and _is_target_intent(q, target_column):
        dist = target_distribution(df, target_column)
        chart = px.bar(dist, x=target_column, y="count", title=f"Distribution of {target_column}") if not dist.empty else None
        if positive_label is not None and target_column in df.columns:
            pos_rate = (df[target_column].astype(str) == str(positive_label)).mean() * 100
            neg_rate = 100.0 - pos_rate
            answer = f"The `{target_column}` distribution is {pos_rate:.2f}% positive/churn and {neg_rate:.2f}% negative/non-churn. {source}"
        else:
            answer = f"Here is the target distribution for `{target_column}`. {source}"
        return CopilotResponse(answer, table=dist, chart=chart, interpreted_question=q, corrections=corrections, context={"topic": "target_distribution", "target_column": target_column})

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


def _answer_grounding_status(response: CopilotResponse, df: pd.DataFrame) -> Tuple[str, str]:
    """Assess whether the returned answer has the evidence required by its intent.

    Suggested-question tables are *not* evidence. A controlled refusal or
    limitation *is* successful RQ2 behaviour when it correctly identifies that
    evidence is unavailable or an action would be unsafe. Those responses are
    exported as PASS with a separate response_type such as SAFE_REFUSAL or
    SUPPORTED_LIMITATION, so they are not confused with routing failures.
    """
    ctx = response.context or {}
    topic = str(ctx.get("topic", "")).lower()
    answer_text = _norm_text(response.answer)

    if topic in {"fallback", "clarification"}:
        return "FAIL", "The question could not be mapped to a supported evidence operation; no claim was guessed."
    if topic == "safe_refusal":
        return "PASS", "The request was correctly refused using a transparent safety/decision-support rule; no unsupported or automated adverse claim was generated."
    if topic in {"data_readiness_fail", "supported_limitation"}:
        return "PASS", "The answer correctly reports that required evidence is unavailable and provides a limitation/next step instead of guessing."
    if ctx.get("evidence_missing"):
        return "PASS", "The intent was understood and the response correctly states that the required model/explanation evidence is missing instead of inventing it."
    if ctx.get("evidence_partial"):
        return "WARNING", "Only partial evidence is available; the answer is intentionally limited."
    if topic in {"model", "drivers", "local_prediction_explanation", "shap_explanation"} and (
        "train a model" in answer_text or "not available" in answer_text or "not generated" in answer_text
    ):
        return "FAIL", "The requested model/explanation artefact is not available yet."

    # Most supported handlers return a table/chart.  Some transparent rule and
    # system/safety answers are text-only but still have an explicit recognised
    # topic and dataset/column context.
    has_structured_evidence = bool((response.table is not None and not response.table.empty) or response.chart is not None)
    has_explicit_context = bool(topic or ctx.get("target_column") or ctx.get("group_col") or ctx.get("rank_col") or ctx.get("text_col"))
    if has_structured_evidence or (has_explicit_context and len(response.answer.strip()) > 20):
        return "PASS", "The answer is supported by the active dataset, available model/explanation artefacts, session context or transparent project rules."
    return "WARNING", "The answer has limited structured evidence and should be reviewed cautiously."





# ---------------------------------------------------------------------------
# v15.3 reviewer-stability helpers: system/meta answers, safety refusal,
# retention/churn actions, categorical target relationships and prediction evidence.
# ---------------------------------------------------------------------------

def _system_or_safety_response(
    q: str,
    df: pd.DataFrame,
    dataset_name: str,
    target_column: Optional[str],
    positive_label: Optional[Any],
) -> Optional[CopilotResponse]:
    """Answer supported system/research/audit/memory questions and refuse unsafe requests.

    These are allowed because the project must explain trust, logging, memory,
    graphs and safety during a supervisor/reviewer demo. They do not require
    external knowledge or unrestricted LLM reasoning.
    """
    tokens = _query_tokens(q)
    compact = _compact(q)
    source = _dataset_source_sentence(dataset_name)

    # v15.5: reviewer/project meta questions must not fall through to generic
    # dataset frequency/refusal output. They are part of the research artefact,
    # and the answer is grounded in the implemented architecture, not external
    # web knowledge.
    if (("research" in tokens and ("gap" in tokens or "contribution" in tokens)) or
        ("different" in tokens and "dashboard" in tokens) or
        ("normal" in tokens and "dashboard" in tokens) or
        ("main" in tokens and "contribution" in tokens)):
        table = pd.DataFrame([
            {"gap_or_contribution": "Data readiness gap", "how_addressed": "PASS/WARNING/FAIL checks before answer generation."},
            {"gap_or_contribution": "Grounding gap", "how_addressed": "Each answer shows dataset, columns, model/explanation evidence, limitations and warnings."},
            {"gap_or_contribution": "Explanation-to-action gap", "how_addressed": "Model drivers and feedback themes are translated into plain-English recommendations."},
            {"gap_or_contribution": "Memory/traceability gap", "how_addressed": "Implemented short-term session context plus export/audit evidence; ChromaDB/JSONL remains an optional future extension."},
            {"gap_or_contribution": "Evaluation gap", "how_addressed": "The system is evaluated by model metrics, grounding, trust, usability and decision-support value."},
        ])
        return CopilotResponse(
            "The research gap is not that dashboards or ML models are missing. The gap addressed here is a reproducible and evaluable Copilot workflow that connects data readiness, predictive modelling, XAI explanation, controlled evidence-grounded answering, short-term session memory, business recommendation, safety warning and exportable evaluation evidence in one decision-support artefact. Long-term retrieval is a future extension, not a core evaluated claim. This is different from a normal dashboard because it does not only show charts; it checks readiness, explains model evidence, grounds answers and refuses unsupported questions. " + source,
            table=table,
            interpreted_question=q,
            context={"topic": "research_contribution"},
        )

    if "chromadb" in compact or ("chroma" in tokens and "db" in tokens):
        table = pd.DataFrame([
            {"memory_layer": "Short-term session state", "role": "implemented and evaluated memory for the active dataset, target/model context and follow-up questions"},
            {"memory_layer": "ChromaDB / JSONL", "role": "optional future semantic-memory demonstration; disabled by default"},
            {"memory_layer": "PostgreSQL", "role": "future structured/live business-data and audit store in the scalable design"},
        ])
        return CopilotResponse(
            "ChromaDB is present only as an **optional future semantic memory demonstration** and is disabled by default in the assessed workflow. The implemented dissertation memory is short-term session state. If the experimental option is manually enabled, ChromaDB (or a JSONL fallback) can store/retrieve evidence summaries to demonstrate how a later enterprise version could support long-term retrieval. " + source,
            table=table,
            interpreted_question=q,
            context={"topic": "memory_architecture"},
        )

    if ("export" in tokens or "download" in tokens) and (("chat" in tokens) or ("evidence" in tokens) or ("answers" in tokens) or ("report" in tokens)):
        table = pd.DataFrame([
            {"export_item": "Copilot chat answers", "where": "Export Data page", "purpose": "dissertation screenshot/evaluation evidence"},
            {"export_item": "cleaned dataset CSV", "where": "Export Data page", "purpose": "reproducibility"},
            {"export_item": "readiness gates", "where": "Export Data page", "purpose": "PASS/WARNING/FAIL proof"},
            {"export_item": "audit/session log", "where": "Export Data page", "purpose": "traceability of user questions and answers"},
            {"export_item": "research framework", "where": "Export Data page", "purpose": "supervisor/reviewer evidence"},
        ])
        return CopilotResponse(
            "Yes. Use the Export Data page to download the full results ZIP/HTML report, cleaned dataset, readiness gates, model/explanation evidence, Copilot chat answers, session audit log, runtime evidence and research framework. Experimental semantic-memory events appear only if that future/optional feature has been manually enabled. " + source,
            table=table,
            interpreted_question=q,
            context={"topic": "audit_logging"},
        )

    if ("competitor" in tokens and ("price" in tokens or "pricing" in tokens)) or ("market" in tokens and "price" in tokens):
        table = pd.DataFrame([{
            "requested_evidence": "competitor pricing / external market data",
            "answer_status": "REFUSED",
            "reason": "The active Telco churn dataset does not contain competitor pricing evidence.",
            "safe_next_step": "Upload or connect competitor/market pricing data, then ask the question again.",
        }])
        return CopilotResponse(
            "I cannot answer competitor-pricing questions from this dataset because the active dataset does not contain competitor or external market pricing evidence. This request is refused instead of guessed. " + source,
            table=table,
            interpreted_question=q,
            safety_warning="Unsupported external-market question refused. Connect the required data source before analysis.",
            context={"topic": "safe_refusal"},
        )

    legal_terms = {"legal", "law", "lawsuit", "regulation", "contractual", "court", "solicitor", "lawyer"}
    if tokens & legal_terms:
        table = pd.DataFrame([{
            "question_type": "legal / regulated advice",
            "answer_status": "REFUSED",
            "reason": "The active dataset does not contain legal evidence and the Copilot is not a legal adviser.",
            "safe_action": "Escalate to a qualified human reviewer or legal/compliance team.",
        }])
        return CopilotResponse(
            "I cannot provide legal advice or make legal conclusions from this dataset. The system can only provide dataset-grounded operational decision support, so legal/compliance questions must be reviewed by a qualified human. " + source,
            table=table,
            interpreted_question=q,
            safety_warning="Unsupported legal advice request refused. Human legal/compliance review is required.",
            context={"topic": "safe_refusal"},
        )

    if ("safety" in tokens and ("warning" in tokens or "warn" in tokens)) or ("automatic" in tokens and "decision" in tokens):
        table = pd.DataFrame([
            {"control": "Human review", "purpose": "Predictions and recommendations support decisions but do not make final customer actions automatically."},
            {"control": "Evidence grounding", "purpose": "Responses must be traceable to active dataset evidence, model metrics, explanation artefacts or memory/audit records."},
            {"control": "Limitation statement", "purpose": "The answer states missing external context and uncertainty."},
            {"control": "Refusal logic", "purpose": "Unsupported or high-risk questions are refused rather than guessed."},
        ])
        return CopilotResponse(
            "Before acting on churn predictions, the safety warning is: use the output as decision support only. A human reviewer must check the customer context, data quality, model limitations and fairness/compliance implications before any retention or service action. " + source,
            table=table,
            interpreted_question=q,
            safety_warning="Do not make automatic customer decisions from the model or Copilot answer.",
            context={"topic": "safety_warning", "target_column": target_column},
        )

    if ("graph" in tokens or "graphs" in tokens or "visual" in tokens or "visualisation" in tokens or "visualization" in tokens) and ("active" in tokens or "dataset" in tokens or "only" in tokens or "generated" in tokens):
        table = pd.DataFrame([
            {"visual_evidence_source": "active cleaned DataFrame", "status": "USED"},
            {"visual_evidence_source": "external websites or market data", "status": "NOT USED"},
            {"visual_evidence_source": "model/explanation artefacts", "status": "USED only for model-driver charts when generated"},
        ])
        return CopilotResponse(
            "Yes. Graphs are generated from the currently active integrated dataset in the application. External market or web data is not used unless the user explicitly connects that data source. " + source,
            table=table,
            interpreted_question=q,
            context={"topic": "visual_grounding"},
        )

    if "memory" in tokens and ("consume" in tokens or "usage" in tokens or "ram" in tokens or "size" in tokens):
        mem_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
        table = pd.DataFrame([{
            "dataset": dataset_name,
            "rows": len(df),
            "columns": df.shape[1],
            "memory_mb": round(mem_mb, 3),
            "local_performance_note": "Charts use aggregation/sampling where needed; auto-modelling is cached once per dataset.",
        }])
        return CopilotResponse(
            f"The active cleaned dataset consumes approximately {mem_mb:.3f} MB of RAM inside pandas. This is small enough for the local Streamlit proof-of-concept, and the app also uses caching/one-time auto-modelling and chart aggregation to reduce repeated work. {source}",
            table=table,
            interpreted_question=q,
            safety_warning="Memory usage is an implementation estimate from the loaded dataset, not a production capacity guarantee.",
            context={"topic": "dataset_memory_usage"},
        )

    if "short" in tokens and "memory" in tokens:
        table = pd.DataFrame([
            {"memory_item": "active dataset name", "purpose": "keeps follow-up questions tied to the selected dataset"},
            {"memory_item": "last question and interpreted context", "purpose": "supports follow-up wording such as 'same again' or 'why'"},
            {"memory_item": "selected target/model/explanation state", "purpose": "connects model evidence to later Copilot answers"},
            {"memory_item": "chat history for the current session", "purpose": "allows export and audit of questions and answers"},
        ])
        return CopilotResponse(
            "Short-term memory stores session context such as the active dataset, recent question/answer context, selected target, model state, explanation state and chat history. It supports follow-up questions during the current Streamlit session. " + source,
            table=table,
            interpreted_question=q,
            context={"topic": "memory_design"},
        )

    if "mongodb" in compact or ("mongo" in tokens and "db" in tokens):
        table = pd.DataFrame([
            {"storage_layer": "PostgreSQL", "role": "structured/live business data, metadata and audit records"},
            {"storage_layer": "ChromaDB", "role": "future/optional semantic long-term memory demonstration; disabled by default"},
            {"storage_layer": "MongoDB", "role": "removed from the core design because it is less directly suited to semantic retrieval than a vector store"},
        ])
        return CopilotResponse(
            "MongoDB was removed from the proposed enterprise extension. The implemented dissertation workflow uses short-term session memory. PostgreSQL and ChromaDB are retained only as future/optional architecture choices for structured records and semantic retrieval respectively. " + source,
            table=table,
            interpreted_question=q,
            context={"topic": "memory_architecture"},
        )

    if ("audit" in tokens and ("trail" in tokens or "prove" in tokens or "log" in tokens)) or ("logged" in tokens and "copilot" in tokens) or ("answers" in tokens and "logged" in tokens):
        table = pd.DataFrame([
            {"logged_item": "dataset name/source", "why_it_matters": "proves which data source supported the answer"},
            {"logged_item": "question and interpreted intent", "why_it_matters": "shows how the Copilot understood the user"},
            {"logged_item": "readiness/grounding status", "why_it_matters": "shows whether evidence was sufficient"},
            {"logged_item": "columns/model/explanation artefacts", "why_it_matters": "supports traceability and reviewer inspection"},
            {"logged_item": "answer and safety warning", "why_it_matters": "supports export, evaluation and audit review"},
        ])
        return CopilotResponse(
            "The audit trail proves traceability: which dataset was used, what question was asked, how it was interpreted, what evidence/columns/model artefacts were used, and what safety warning was returned. Copilot answers are stored in current-session chat history and export/audit files. Optional future semantic-memory logging is separate and disabled by default. " + source,
            table=table,
            interpreted_question=q,
            context={"topic": "audit_logging"},
        )

    if ("real" in tokens and "evidence" in tokens) or ("dataset" in tokens and "evidence" in tokens and ("suggestions" in tokens or "recommendation" in tokens or "trust" in tokens)):
        evidence_sources = ["active dataset rows/columns", "readiness checks", "EDA statistics"]
        if target_column:
            evidence_sources.append(f"model target `{target_column}`")
        table = pd.DataFrame([{"evidence_source": s, "status": "available"} for s in evidence_sources])
        return CopilotResponse(
            "Yes, supported suggestions are based on the active integrated dataset and available artefacts. The answer frame shows the dataset name, readiness status, columns/artefacts used, model evidence when available, limitations and safety warning. Unsupported evidence is refused instead of guessed. " + source,
            table=table,
            interpreted_question=q,
            context={"topic": "grounding_evidence", "target_column": target_column},
        )

    return None


def _categorical_target_relationship_response(
    q: str,
    df: pd.DataFrame,
    target_column: Optional[str],
    positive_label: Optional[Any],
    dataset_name: str,
) -> Optional[CopilotResponse]:
    """Compare a categorical predictor with the selected target, e.g. payment method vs churn."""
    if not target_column or target_column not in df.columns:
        return None
    tokens = _query_tokens(q)
    relationship_terms = {"linked", "link", "higher", "highest", "lower", "relate", "relates", "related", "compared", "compare", "by", "segment", "segments", "influence", "influences", "influenced", "association", "associated"}
    if not (tokens & relationship_terms):
        return None
    explicit = _explicit_column_mentions(q, list(df.columns))
    id_like = set(_find_id_like_columns(df))
    cat_cols = [c for c in explicit if c != target_column and c not in id_like and c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    if not cat_cols:
        # Try a safe column match for questions like "payment method linked with churn".
        candidate = _safe_best_column(q, df, allow_target=False, target_column=target_column)
        if candidate and candidate not in id_like and candidate != target_column and not pd.api.types.is_numeric_dtype(df[candidate]):
            cat_cols = [candidate]
    if not cat_cols:
        return None
    group_col = cat_cols[0]
    temp = df[[group_col, target_column]].dropna()
    if temp.empty:
        return None
    pos = positive_label if positive_label is not None else temp[target_column].value_counts().index[0]
    temp["__positive"] = (temp[target_column].astype(str) == str(pos)).astype(int)
    table = temp.groupby(group_col, dropna=False)["__positive"].agg(["mean", "count"]).reset_index()
    table["positive_rate_percent"] = (table["mean"] * 100).round(2)
    table = table.rename(columns={"count": "record_count"}).sort_values("positive_rate_percent", ascending=False).head(30)
    chart = px.bar(table, x=group_col, y="positive_rate_percent", hover_data=["record_count"], title=f"{target_column} positive rate by {group_col}")
    top = table.iloc[0]
    return CopilotResponse(
        f"I compared `{group_col}` against `{target_column}`. The highest observed positive-rate group is `{top[group_col]}` with {top['positive_rate_percent']:.2f}% positive rate across {int(top['record_count']):,} records. This is an observed dataset pattern, not proof of causation. {_dataset_source_sentence(dataset_name)}",
        table=table,
        chart=chart,
        interpreted_question=q,
        safety_warning="Group differences are decision-support evidence only. Investigate causes and fairness/compliance before acting.",
        context={"topic": "target_relationship", "group_col": group_col, "target_column": target_column},
    )


def _retention_or_churn_action_response(
    q: str,
    df: pd.DataFrame,
    dataset_name: str,
    target_column: Optional[str],
    positive_label: Optional[Any],
    model_output: Optional[Any],
    explanation_output: Optional[Any],
) -> Optional[CopilotResponse]:
    """High-standard churn/retention actions using segments, feedback and model evidence."""
    tokens = _query_tokens(q)
    if not target_column or target_column not in df.columns:
        return None
    if not ("retention" in tokens or "prioritised" in tokens or "prioritized" in tokens or "prioritise" in tokens or "prioritize" in tokens
            or (bool(tokens & {"customer", "customers"}) and "risk" in tokens)
            or ("action" in tokens and bool(tokens & {"churn", "risk"}))):
        return None

    source = _dataset_source_sentence(dataset_name)
    seg = _highest_target_segments(df, target_column, positive_label, dataset_name, q + " highest risk segment churn")
    feedback_cols = _feedback_columns(df)
    rows: List[Dict[str, Any]] = []
    if seg is not None and seg.table is not None and not seg.table.empty:
        for _, r in seg.table.head(5).iterrows():
            rows.append({
                "priority_type": "high churn segment",
                "evidence": f"{r['segment_column']} = {r['segment_value']}",
                "positive_rate_percent": r["positive_rate_percent"],
                "record_count": r["record_count"],
                "recommended_action": "Prioritise this segment for human-reviewed retention investigation and targeted communication.",
            })
    if feedback_cols:
        ftable = _feedback_business_action_table(df, feedback_cols[0], target_column=target_column, positive_label=positive_label)
        for _, r in ftable.head(3).iterrows():
            rows.append({
                "priority_type": "feedback improvement",
                "evidence": r["theme"],
                "positive_rate_percent": "",
                "record_count": r.get("negative_evidence_rows", r.get("evidence_rows", "")),
                "recommended_action": r["recommended_action"],
            })
    table = pd.DataFrame(rows)
    if table.empty:
        return None
    chart = None
    if "positive_rate_percent" in table.columns:
        numeric_part = table[pd.to_numeric(table["positive_rate_percent"], errors="coerce").notna()].copy()
        if not numeric_part.empty:
            numeric_part["positive_rate_percent"] = pd.to_numeric(numeric_part["positive_rate_percent"])
            chart = px.bar(numeric_part, x="positive_rate_percent", y="evidence", orientation="h", title="Retention priority evidence")
    model_line = _model_summary(model_output) if model_output is not None else "No trained model is currently available, so prioritisation is based on observed dataset segments and feedback evidence."
    answer = (
        "For retention, prioritise customers or segments where churn evidence is strongest, but only after human review. "
        f"{model_line} The table lists high churn segments and/or feedback improvement priorities. {source}"
    )
    return CopilotResponse(
        answer,
        table=table,
        chart=chart,
        interpreted_question=q,
        safety_warning="Retention actions must be reviewed by a human. Do not automatically contact, penalise, cancel, or treat customers only because of model output.",
        context={"topic": "retention_actions", "target_column": target_column},
    )


def _prediction_evidence_response(
    q: str,
    df: pd.DataFrame,
    dataset_name: str,
    target_column: Optional[str],
    positive_label: Optional[Any],
    model_output: Optional[Any],
    explanation_output: Optional[Any],
) -> Optional[CopilotResponse]:
    tokens = _query_tokens(q)
    if not (("prediction" in tokens and "evidence" in tokens) or ("model" in tokens and "evidence" in tokens)):
        return None
    source = _dataset_source_sentence(dataset_name)
    rows: List[Dict[str, Any]] = []
    if model_output is not None and getattr(model_output, "best_result", None) is not None:
        best = model_output.best_result
        metrics = getattr(best, "metrics", {}) or {}
        for key in ["f1", "roc_auc", "recall", "precision", "accuracy"]:
            val = metrics.get(key)
            if val is not None:
                try:
                    rows.append({"evidence_type": "model_metric", "name": key, "value": round(float(val), 4), "meaning": "predictive performance evidence"})
                except Exception:
                    pass
        rows.append({"evidence_type": "model", "name": "best_model", "value": best.model_name, "meaning": "selected model used for decision-support evidence"})
    if explanation_output is not None and getattr(explanation_output, "global_importance", None) is not None:
        imp = explanation_output.global_importance.head(5)
        for _, r in imp.iterrows():
            rows.append({"evidence_type": "explanation_driver", "name": str(r.get("feature", "")), "value": r.get("importance", ""), "meaning": "feature driver evidence"})
    else:
        rows.append({"evidence_type": "explanation_status", "name": "SHAP/fallback", "value": "not generated", "meaning": "generate explanation in the Model page to add driver evidence"})
    if target_column:
        rows.append({"evidence_type": "target", "name": "target_column", "value": target_column, "meaning": "prediction target used by the trained model"})
    table = pd.DataFrame(rows)
    if table.empty:
        return CopilotResponse("No prediction evidence is available yet. Train a model first, then generate explanation evidence in the Model page. " + source, interpreted_question=q, context={"topic": "model"})
    return CopilotResponse(
        "Prediction evidence comes from the selected target, trained model metrics and SHAP/fallback explanation artefacts when generated. This evidence supports recommendations only as decision support, not as an automatic decision rule. " + source,
        table=table,
        interpreted_question=q,
        safety_warning="Prediction evidence is probabilistic and must be checked with business/customer context before action.",
        context={"topic": "model", "target_column": target_column},
    )
def _model_evidence_note(model_output: Optional[Any], explanation_output: Optional[Any], context: Dict[str, Any]) -> str:
    """Explain where modelling is used in the answer."""
    topic = str((context or {}).get("topic", "")).lower()
    if model_output is None or getattr(model_output, "best_result", None) is None:
        return "No trained model evidence is available for this answer. The response is based on dataset-level evidence only."
    best = model_output.best_result
    metrics = getattr(best, "metrics", {}) or {}
    target = getattr(model_output, "target_column", getattr(best, "target_column", "target"))
    parts = [f"preferred F1-selected model `{best.model_name}`", f"target `{target}`"]
    for key, label in [("f1", "F1"), ("roc_auc", "ROC-AUC"), ("recall", "Recall"), ("precision", "Precision")]:
        val = metrics.get(key)
        try:
            if val is not None and not np.isnan(float(val)):
                parts.append(f"{label}={float(val):.3f}")
        except Exception:
            pass
    if explanation_output is not None and getattr(explanation_output, "global_importance", None) is not None:
        drivers = getattr(explanation_output, "global_importance").head(3)
        if drivers is not None and not drivers.empty and "feature" in drivers.columns:
            parts.append("top drivers: " + ", ".join(map(str, drivers["feature"].head(3).tolist())))
        else:
            parts.append("SHAP/fallback explanation available")
    else:
        parts.append("generate explanation in the model page for SHAP/fallback drivers")
    if topic in {"model", "drivers", "recommendation", "business_insights", "feedback_business_actions", "feedback_themes", "feedback_technical_issues"}:
        return "Model evidence considered: " + "; ".join(parts) + "."
    return "Model evidence available for model/prediction questions: " + "; ".join(parts) + "."



def _confidence_summary(response: CopilotResponse, answer_status: str) -> str:
    """Reviewer-friendly confidence label based on evidence sufficiency."""
    ctx = response.context or {}
    topic = str(ctx.get("topic", "")).lower()
    if topic == "safe_refusal":
        return "HIGH — safe refusal is explicitly supported by the project's human-review/safety rule."
    if topic in {"data_readiness_fail", "supported_limitation"}:
        return "HIGH — the limitation is grounded in the absence of required evidence; no answer was guessed."
    if topic in {"fallback", "clarification"}:
        return "LOW — no supported answer operation was found; clarification is required."
    if answer_status == "PASS":
        return "HIGH — supported by the active dataset/model/explanation/session evidence or a transparent project rule."
    if answer_status == "WARNING":
        return "MEDIUM — partial evidence; human review is required."
    return "LOW — insufficient required evidence for the requested claim."


def _frame_copilot_answer(response: CopilotResponse, *, question: str, dataset_name: str, df: pd.DataFrame, model_output: Optional[Any] = None, explanation_output: Optional[Any] = None) -> CopilotResponse:
    """Make every answer explicit, evidence-based and export-friendly."""
    if response.answer.strip().startswith("### Copilot answer"):
        return response
    context = response.context or {}
    cols_used = []
    for key in ["target_column", "group_col", "rank_col", "text_col", "filter_col"]:
        value = context.get(key)
        if value and value not in cols_used:
            cols_used.append(str(value))
    if not cols_used:
        cols_used = _explicit_column_mentions(question, list(df.columns))[:4]
    cols_text = ", ".join(f"`{c}`" for c in cols_used) if cols_used else "dataset-level / project-rule evidence"

    if response.table is not None and not response.table.empty:
        evidence_count = f"Structured table returned: {len(response.table):,} row(s)."
    elif response.chart is not None:
        evidence_count = "Visual chart returned."
    else:
        evidence_count = "Controlled text/rule evidence returned; no supporting data table was required or available."

    target_for_readiness = context.get("target_column")
    if not target_for_readiness and model_output is not None:
        target_for_readiness = getattr(model_output, "target_column", None)
    readiness = _cached_readiness_report(df, dataset_name=dataset_name, target_column=target_for_readiness)
    readiness_status = readiness.overall_status
    readiness_score = readiness.quality_score
    core_nonpass = [g.gate for g in readiness.gates if g.status != "PASS" and not g.gate.startswith("Supplementary")]
    readiness_message = "All core dissertation gates pass." if not core_nonpass else "Non-PASS core gates: " + ", ".join(core_nonpass) + "."

    answer_status, answer_status_reason = _answer_grounding_status(response, df)
    confidence = _confidence_summary(response, answer_status)
    topic = str(context.get("topic", "") or "unknown")
    limitation_or_refusal = topic in {"fallback", "clarification", "data_readiness_fail", "safe_refusal", "supported_limitation"}

    if topic == "safe_refusal":
        response_type = "SAFE_REFUSAL"
    elif topic in {"data_readiness_fail", "supported_limitation"} or context.get("evidence_missing"):
        response_type = "SUPPORTED_LIMITATION"
    elif topic in {"fallback", "clarification"}:
        response_type = "ROUTING_FAILURE"
    elif answer_status == "WARNING":
        response_type = "PARTIAL_EVIDENCE"
    else:
        response_type = "GROUNDED_ANSWER"

    response.grounding_status = answer_status
    response.grounding_reason = answer_status_reason
    response.confidence = confidence
    response.intent = topic
    response.limitation_or_refusal = limitation_or_refusal
    response.response_type = response_type

    business_hint = _business_use_hint(context, question)
    limitation = (
        "This answer uses only the active integrated dataset, implemented short-term session context and available model/explanation artefacts or transparent project rules. "
        "It does not use external market, customer, HR, legal or operational evidence unless that source is explicitly connected. "
        "Long-term ChromaDB/document retrieval is a future/optional extension and is disabled by default in the assessed workflow."
    )
    model_note = _model_evidence_note(model_output, explanation_output, context)
    raw_answer = response.answer
    response.answer = (
        "### Copilot answer\n"
        f"**Question understood as:** {response.interpreted_question or question}\n\n"
        f"**Dataset used:** `{dataset_name}` ({len(df):,} rows, {df.shape[1]:,} columns).\n\n"
        f"**Data readiness:** {readiness_status} — quality score {readiness_score:.1f}/100. {readiness_message}\n\n"
        f"**Answer grounding:** {answer_status} — {answer_status_reason}\n\n"
        f"**Confidence:** {confidence}\n\n"
        f"**Response type:** {response_type}\n\n"
        f"**Intent/topic:** `{topic}`\n\n"
        f"**Columns/artefacts used:** {cols_text}.\n\n"
        f"**Model / prediction evidence:** {model_note}\n\n"
        f"**Evidence-grounded answer:** {raw_answer}\n\n"
        f"**Business / decision-support use:** {business_hint}\n\n"
        f"**Limitations:** {limitation}\n\n"
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
    return _frame_copilot_answer(response, question=question, dataset_name=dataset_name, df=df, model_output=model_output, explanation_output=explanation_output)
