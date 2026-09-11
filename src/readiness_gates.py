"""Measurable PASS/WARNING/FAIL readiness criteria for the dissertation prototype.

The core gates mirror Table 3.1 in the dissertation methodology:
file loading, dataset size, missing values, duplicate rows, target-column
availability, target validity, class balance, data types and evidence
availability.  A privacy/security screen is retained as a supplementary safety
check, but it is not used to redefine the dissertation's data-readiness result.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

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

CORE_READINESS_GATES = {
    "File loading / data availability",
    "Dataset size",
    "Missing values",
    "Duplicate rows",
    "Target column",
    "Target validity",
    "Class balance",
    "Data types",
    "Evidence availability",
}


@dataclass
class ReadinessCriterion:
    check_area: str
    pass_condition: str
    warning_condition: str
    fail_condition: str
    why_it_matters: str


def readiness_criteria_table() -> pd.DataFrame:
    """Return the dissertation Table 3.1 criteria plus one labelled extra check."""
    rows = [
        ReadinessCriterion(
            "File loading / data availability",
            "File loads correctly and contains usable records/columns",
            "File loads with minor structural concerns",
            "File cannot be loaded, is empty, or has too few columns",
            "The workflow requires usable records and fields before analysis can be grounded.",
        ),
        ReadinessCriterion(
            "Dataset size",
            "500 or more rows",
            "100-499 rows",
            "Fewer than 100 rows",
            "Very small datasets can produce unstable model metrics and explanations.",
        ),
        ReadinessCriterion(
            "Missing values",
            "0-5% missing values",
            "More than 5% and up to 20% missing values",
            "More than 20% missing values",
            "High missingness can distort summaries, models and recommendations.",
        ),
        ReadinessCriterion(
            "Duplicate rows",
            "0-5% duplicate rows",
            "More than 5% and up to 15% duplicate rows",
            "More than 15% duplicate rows",
            "Duplicates can inflate patterns and model performance.",
        ),
        ReadinessCriterion(
            "Target column",
            "Valid target column is available",
            "Possible target requires user confirmation",
            "No suitable target column is available",
            "Prediction and explanation require a clearly selected target.",
        ),
        ReadinessCriterion(
            "Target validity",
            "Target has at least two usable classes/values",
            "Target requires cleaning or confirmation",
            "Target has one class or unusable values",
            "A supervised model cannot be trained on a constant or unusable target.",
        ),
        ReadinessCriterion(
            "Class balance",
            "Minority class is 10% or higher",
            "Minority class is between 5% and 10%",
            "Minority class is below 5%",
            "Severe imbalance can make accuracy misleading and reduce minority-class recall.",
        ),
        ReadinessCriterion(
            "Data types",
            "Columns can be processed after standard encoding/scaling",
            "Some columns require cleaning or conversion",
            "Major data-type issues prevent modelling",
            "The modelling pipeline needs scalar numeric/categorical/date-like fields.",
        ),
        ReadinessCriterion(
            "Evidence availability",
            "Enough evidence exists for charts, model explanation and Copilot answers",
            "Partial evidence exists, so limitations are required",
            "Required evidence is missing",
            "The Copilot must refuse or limit answers when supporting evidence is absent.",
        ),
        ReadinessCriterion(
            "Supplementary privacy/security screen",
            "No obvious personal identifiers detected",
            "Possible personal identifiers detected",
            "High-risk sensitive fields or unauthorised access would require restriction",
            "This is an additional responsible-use control, not a replacement for Table 3.1.",
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
    return FAIL, minority, f"Minority class is {minority:.2f}%, below the 5% FAIL threshold."


def answer_readiness_status(*, has_required_columns: bool, has_evidence: bool, has_model_or_eda: bool = True, safety_ok: bool = True) -> str:
    """Reusable answer-level PASS/WARNING/FAIL decision."""
    if not safety_ok or not has_required_columns or not has_evidence:
        return FAIL
    if not has_model_or_eda:
        return WARNING
    return PASS
