from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class InterpretedQuestion:
    raw: str
    cleaned: str
    corrected: str
    changed: bool
    confidence: float
    matched_columns: List[str]


COMMON_TYPOS: Dict[str, str] = {
    # general question words
    "wat": "what",
    "wht": "what",
    "whats": "what is",
    "whta": "what",
    "hw": "how",
    "hwo": "how",
    "shw": "show",
    "sho": "show",
    "disply": "display",
    "graf": "graph",
    "grahp": "graph",
    "char": "chart",
    "chrt": "chart",
    "compar": "compare",
    "compair": "compare",
    "avg": "average",

    # project/domain words
    "chrun": "churn",
    "chrn": "churn",
    "churnn": "churn",
    "churnrate": "churn rate",
    "custmer": "customer",
    "cutomer": "customer",
    "contrct": "contract",
    "conract": "contract",
    "contrat": "contract",
    "tenur": "tenure",
    "montly": "monthly",
    "mothly": "monthly",
    "charg": "charges",
    "chargesrate": "charges",
    "paymnt": "payment",
    "methd": "method",
    "internt": "internet",
    "servce": "service",
    "suport": "support",
    "securty": "security",

    # data quality words
    "colums": "columns",
    "colomns": "columns",
    "colmn": "column",
    "feild": "field",
    "feilds": "fields",
    "missng": "missing",
    "mising": "missing",
    "duplcate": "duplicate",
    "duplcates": "duplicates",
    "duplicte": "duplicate",

    # recommendation/explanation words
    "recomend": "recommend",
    "recommand": "recommend",
    "recomendation": "recommendation",
    "recommandation": "recommendation",
    "improv": "improve",
    "advce": "advice",
    "reasn": "reason",
    "explan": "explain",
    "explane": "explain",
    "explaination": "explanation",
    "risck": "risk",
    "higest": "highest",
    "highst": "highest",
    "lowst": "lowest",
    "summry": "summary",
}


PHRASE_FIXES: Dict[str, str] = {
    "what churn rate": "what is the churn rate",
    "what is chrun rate": "what is the churn rate",
    "show chrun by contrct": "show churn by contract",
    "show churn contract": "show churn by contract",
    "show churn monthly charge": "show churn by monthly charges",
    "check missng values": "check missing values",
    "what colums in file": "what columns are in this file",
    "what fields in file": "what columns are in this file",
    "why high risk": "why is this customer high risk",
    "why this customer high": "why is this customer high risk",
    "what should improve": "what should we improve",
    "what action": "what action should we consider",
    "can system decide automatic": "can this system make automatic decisions",
}


SYNONYM_MAP: Dict[str, str] = {
    "display": "show",
    "plot": "show",
    "graph": "show",
    "chart": "show",
    "percentage": "rate",
    "percent": "rate",
    "ratio": "rate",
    "client": "customer",
    "clients": "customers",
    "leave": "churn",
    "leaving": "churn",
    "loss": "churn",
    "price": "charges",
    "cost": "charges",
    "bill": "charges",
    "billing": "payment",
    "advice": "recommend",
    "suggest": "recommend",
    "suggestion": "recommend",
    "solution": "recommend",
    "improvement": "improve",
    "drivers": "reasons",
    "factors": "reasons",
}


def _clean_text(text: str) -> str:
    cleaned = (text or "").strip().lower()
    cleaned = re.sub(r"[^a-z0-9_\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _column_aliases(columns: Iterable[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for col in columns:
        raw = str(col)
        compact = re.sub(r"[^a-z0-9]", "", raw.lower())
        if compact:
            aliases[compact] = raw

        # Split common CamelCase names like MonthlyCharges.
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", raw)
        for part in parts:
            aliases[part.lower()] = raw

    manual = {
        "contract": "Contract",
        "tenure": "tenure",
        "monthly": "MonthlyCharges",
        "charges": "MonthlyCharges",
        "charge": "MonthlyCharges",
        "price": "MonthlyCharges",
        "cost": "MonthlyCharges",
        "total": "TotalCharges",
        "internet": "InternetService",
        "service": "InternetService",
        "payment": "PaymentMethod",
        "method": "PaymentMethod",
        "security": "OnlineSecurity",
        "support": "TechSupport",
        "customer": "customerID",
        "id": "customerID",
        "gender": "gender",
        "senior": "SeniorCitizen",
        "partner": "Partner",
        "dependent": "Dependents",
        "dependents": "Dependents",
    }

    existing_compact = {re.sub(r"[^a-z0-9]", "", str(c).lower()): str(c) for c in columns}
    for alias, target in manual.items():
        target_compact = re.sub(r"[^a-z0-9]", "", target.lower())
        if target in columns:
            aliases[alias] = target
        elif target_compact in existing_compact:
            aliases[alias] = existing_compact[target_compact]

    return aliases


def interpret_question(question: str, columns: Iterable[str]) -> InterpretedQuestion:
    raw = question or ""
    cleaned = _clean_text(raw)

    corrected = cleaned
    for wrong, right in PHRASE_FIXES.items():
        if wrong in corrected:
            corrected = corrected.replace(wrong, right)

    aliases = _column_aliases(columns)
    known_words = set(COMMON_TYPOS.keys()) | set(COMMON_TYPOS.values()) | set(SYNONYM_MAP.keys()) | set(SYNONYM_MAP.values()) | set(aliases.keys())
    corrected_tokens: List[str] = []

    for token in corrected.split():
        if token in COMMON_TYPOS:
            token = COMMON_TYPOS[token]

        # Expand possible multi-word typo result.
        expanded_tokens = str(token).split()
        for expanded in expanded_tokens:
            if expanded in SYNONYM_MAP:
                expanded = SYNONYM_MAP[expanded]

            if expanded not in known_words and len(expanded) > 3:
                close = difflib.get_close_matches(expanded, list(known_words), n=1, cutoff=0.80)
                if close:
                    expanded = COMMON_TYPOS.get(close[0], SYNONYM_MAP.get(close[0], close[0]))

            corrected_tokens.append(expanded)

    corrected = " ".join(corrected_tokens)
    corrected = re.sub(r"\s+", " ", corrected).strip()

    matched_columns: List[str] = []
    compact_question = re.sub(r"[^a-z0-9]", "", corrected)

    for alias, col in aliases.items():
        if alias and alias in compact_question and col not in matched_columns:
            matched_columns.append(col)

    changed = corrected != cleaned
    confidence = 0.75
    if changed:
        confidence += 0.10
    if matched_columns:
        confidence += 0.10
    confidence = min(confidence, 0.95)

    return InterpretedQuestion(
        raw=raw,
        cleaned=cleaned,
        corrected=corrected,
        changed=changed,
        confidence=confidence,
        matched_columns=matched_columns,
    )


def best_column_match(question: str, columns: Iterable[str]) -> Optional[str]:
    interpretation = interpret_question(question, columns)
    if interpretation.matched_columns:
        # Avoid returning customerID as a grouping column unless explicitly about ID.
        non_id = [c for c in interpretation.matched_columns if "id" not in str(c).lower()]
        return non_id[0] if non_id else interpretation.matched_columns[0]

    aliases = _column_aliases(columns)
    candidates = list(aliases.keys())
    for token in interpretation.corrected.split():
        match = difflib.get_close_matches(token, candidates, n=1, cutoff=0.76)
        if match:
            return aliases[match[0]]
    return None


def correction_prefix(interpretation: InterpretedQuestion) -> str:
    if interpretation.changed and interpretation.cleaned != interpretation.corrected:
        return f"I interpreted your question as: **{interpretation.corrected}**.\n\n"
    return ""
