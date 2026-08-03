"""Measurable PASS/WARNING/FAIL readiness criteria for the research prototype.

The purpose of this module is to make the readiness contribution explicit and
reproducible.  These thresholds are intentionally transparent rather than
hidden inside a black-box score.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd

PASS = "PASS"
WARNING = "WARNING"
FAIL = "FAIL"

READINESS_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "missing_cells_percent": {"pass_max": 5.0, "warning_max": 20.0, "fail_over": 20.0},
    "duplicate_rows_percent": {"pass_max": 5.0, "warning_max": 15.0, "fail_over": 15.0},
    "row_count_for_modelling": {"pass_min": 500, "warning_min": 100, "fail_below": 100},
    "minority_class_percent": {"pass_min": 10.0, "warning_min": 5.0, "fail_below": 5.0},
    "column_count": {"pass_min": 2, "fail_below": 2},
}

@dataclass
class ReadinessCriterion:
    criterion: str
    pass_condition: str
    warning_condition: str
    fail_condition: str
    why_it_matters: str


def readiness_criteria_table() -> pd.DataFrame:
    """Return a dissertation-ready explanation of the readiness framework."""
    rows = [
        ReadinessCriterion(
            "File/data availability",
            "Dataset loads and has at least one row and two columns",
            "Dataset loads but has very small scope or weak structure",
            "Dataset cannot be read, is empty, or has fewer than two columns",
            "The Copilot cannot provide evidence-grounded answers without usable records and fields.",
        ),
        ReadinessCriterion(
            "Schema readability",
            "Column names are readable and not mostly generated names",
            "Some columns are unnamed, duplicated, or weakly named",
            "Schema is unusable for matching questions to columns",
            "Column names are needed for question interpretation and evidence traceability.",
        ),
        ReadinessCriterion(
            "Missing cells",
            "0-5% missing cells",
            "More than 5% and up to 20% missing cells",
            "More than 20% missing cells",
            "High missingness can distort summaries, model training, and business recommendations.",
        ),
        ReadinessCriterion(
            "Duplicate rows",
            "0-5% duplicate rows",
            "More than 5% and up to 15% duplicate rows",
            "More than 15% duplicate rows",
            "Duplicates can inflate patterns, customer counts, and model performance.",
        ),
        ReadinessCriterion(
            "Modelling row count",
            "500 or more rows for automatic modelling",
            "100-499 rows; modelling allowed with caution",
            "Fewer than 100 rows; modelling not reliable",
            "Very small datasets can produce unstable model metrics and explanations.",
        ),
        ReadinessCriterion(
            "Target readiness",
            "Target exists and has at least two valid classes/values",
            "Target needs user confirmation or class balance is weak",
            "Target missing, constant, or not usable",
            "Prediction and explanation require a valid target variable.",
        ),
        ReadinessCriterion(
            "Class balance",
            "Minority class is at least 10%",
            "Minority class is 5-10%",
            "Minority class is below 5%",
            "Severe imbalance can make accuracy misleading and reduce recall for important cases.",
        ),
        ReadinessCriterion(
            "Privacy/security screen",
            "No obvious personal identifiers detected",
            "Possible personal identifiers detected",
            "High-risk sensitive fields detected or access is not authorised",
            "Decision support must protect privacy and avoid unsafe automated use of sensitive data.",
        ),
        ReadinessCriterion(
            "Question evidence availability",
            "Required columns/model/explanation artefacts are available",
            "Partial or approximate evidence is available",
            "Required evidence is missing",
            "The Copilot must refuse rather than invent if evidence is not present.",
        ),
    ]
    return pd.DataFrame([asdict(r) for r in rows])


def status_from_percent(value: float, pass_max: float, warning_max: float) -> str:
    if value <= pass_max:
        return PASS
    if value <= warning_max:
        return WARNING
    return FAIL


def row_count_status(rows: int) -> str:
    if rows >= int(READINESS_THRESHOLDS["row_count_for_modelling"]["pass_min"]):
        return PASS
    if rows >= int(READINESS_THRESHOLDS["row_count_for_modelling"]["warning_min"]):
        return WARNING
    return FAIL


def class_balance_status(series: pd.Series) -> tuple[str, float, str]:
    values = series.dropna()
    if values.nunique() < 2:
        return FAIL, 0.0, "Target has fewer than two classes/values."
    rates = values.astype(str).value_counts(normalize=True) * 100
    minority = float(rates.min())
    if minority >= float(READINESS_THRESHOLDS["minority_class_percent"]["pass_min"]):
        return PASS, minority, f"Minority class is {minority:.2f}% of records."
    if minority >= float(READINESS_THRESHOLDS["minority_class_percent"]["warning_min"]):
        return WARNING, minority, f"Minority class is only {minority:.2f}%; use recall/F1 and report imbalance."
    return FAIL, minority, f"Minority class is {minority:.2f}%, which is too imbalanced for reliable automatic modelling."


def answer_readiness_status(*, has_required_columns: bool, has_evidence: bool, has_model_or_eda: bool = True, safety_ok: bool = True) -> str:
    """Reusable answer-level PASS/WARNING/FAIL decision.

    PASS: required columns and evidence are present.
    WARNING: partial evidence is available but one support artefact is weak.
    FAIL: required columns/evidence are missing or the request is unsafe.
    """
    if not safety_ok or not has_required_columns or not has_evidence:
        return FAIL
    if not has_model_or_eda:
        return WARNING
    return PASS
