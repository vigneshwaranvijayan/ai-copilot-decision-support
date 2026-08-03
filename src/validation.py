"""Dataset readiness, quality-gate and trust evidence utilities.

The validation layer implements explicit PASS/WARNING/FAIL thresholds so the
research contribution is measurable and explainable in the dissertation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import hashlib
import json
import time

import pandas as pd

from .readiness_gates import (
    FAIL,
    PASS,
    WARNING,
    READINESS_THRESHOLDS,
    class_balance_status,
    readiness_criteria_table,
    row_count_status,
    status_from_percent,
)


@dataclass
class GateResult:
    gate: str
    status: str  # PASS / WARNING / FAIL
    message: str
    evidence: str = ""


@dataclass
class ReadinessReport:
    dataset_id: str
    dataset_name: str
    rows: int
    columns: int
    memory_mb: float
    missing_cells: int
    duplicate_rows: int
    quality_score: float
    overall_status: str
    gates: List[GateResult]
    generated_at: float

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["gates"] = [asdict(g) for g in self.gates]
        return data

    def gates_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(g) for g in self.gates])


def dataset_fingerprint(df: pd.DataFrame, name: str = "dataset") -> str:
    """Create a stable light-weight fingerprint for caching/audit evidence."""
    cols = [str(c) for c in df.columns]
    payload = {
        "name": name,
        "shape": list(df.shape),
        "columns": cols,
        "dtypes": [str(df[c].dtype) for c in df.columns],
    }
    try:
        sample = df.head(25).astype("string").fillna("<NA>").to_csv(index=False)
        payload["sample"] = sample
    except Exception:
        payload["sample"] = "unavailable"
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _status_rank(status: str) -> int:
    return {PASS: 0, WARNING: 1, FAIL: 2}.get(status, 1)


def _infer_sensitive_columns(df: pd.DataFrame) -> List[str]:
    tokens = ["name", "email", "phone", "mobile", "address", "postcode", "zip", "dob", "birth", "ssn", "national", "passport"]
    matches = []
    for c in df.columns:
        lower = str(c).lower().replace("_", "")
        if any(t in lower for t in tokens):
            matches.append(str(c))
    return matches


def build_readiness_report(df: pd.DataFrame, dataset_name: str = "active_dataset", target_column: Optional[str] = None) -> ReadinessReport:
    rows, cols = int(df.shape[0]), int(df.shape[1])
    total_cells = max(rows * cols, 1)
    missing_cells = int(df.isna().sum().sum()) if not df.empty else 0
    missing_pct = missing_cells / total_cells * 100
    duplicate_rows = int(df.duplicated().sum()) if not df.empty else 0
    duplicate_pct = duplicate_rows / max(rows, 1) * 100
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024)) if not df.empty else 0.0

    gates: List[GateResult] = []

    if rows == 0 or cols == 0:
        gates.append(GateResult("Data availability", FAIL, "Dataset is empty or unavailable.", f"rows={rows}, columns={cols}"))
    elif cols < int(READINESS_THRESHOLDS["column_count"]["pass_min"]):
        gates.append(GateResult("Data availability", FAIL, "Dataset has too few columns for analysis.", f"rows={rows:,}, columns={cols:,}"))
    else:
        gates.append(GateResult("Data availability", PASS, "Dataset contains records and usable columns.", f"rows={rows:,}, columns={cols:,}"))

    unnamed_cols = [c for c in df.columns if str(c).strip() == "" or str(c).lower().startswith("unnamed")]
    duplicate_col_names = len(set(map(str, df.columns))) != len(df.columns)
    if duplicate_col_names:
        gates.append(GateResult("Schema readability", FAIL, "Duplicate column names make evidence tracing unsafe.", "duplicate column names detected"))
    elif unnamed_cols:
        gates.append(GateResult("Schema readability", WARNING, "Some columns have weak or generated names.", ", ".join(map(str, unnamed_cols[:8]))))
    else:
        gates.append(GateResult("Schema readability", PASS, "Column names are readable.", f"{cols:,} columns"))

    miss_status = status_from_percent(
        missing_pct,
        READINESS_THRESHOLDS["missing_cells_percent"]["pass_max"],
        READINESS_THRESHOLDS["missing_cells_percent"]["warning_max"],
    )
    if miss_status == PASS:
        miss_msg = "Missing values are within the PASS threshold of 0-5%."
    elif miss_status == WARNING:
        miss_msg = "Missing values are within the WARNING range of >5-20%; review before decisions."
    else:
        miss_msg = "Missing values exceed 20%, so reliable automatic analysis is not safe."
    gates.append(GateResult("Missing-value quality", miss_status, miss_msg, f"{missing_pct:.2f}% missing cells"))

    dup_status = status_from_percent(
        duplicate_pct,
        READINESS_THRESHOLDS["duplicate_rows_percent"]["pass_max"],
        READINESS_THRESHOLDS["duplicate_rows_percent"]["warning_max"],
    )
    if dup_status == PASS:
        dup_msg = "Duplicate rows are within the PASS threshold of 0-5%."
    elif dup_status == WARNING:
        dup_msg = "Duplicate rows are within the WARNING range of >5-15%; review possible repeated records."
    else:
        dup_msg = "Duplicate rows exceed 15%, so summaries/models may be distorted."
    gates.append(GateResult("Duplicate-row quality", dup_status, dup_msg, f"{duplicate_pct:.2f}% duplicate rows"))

    row_status = row_count_status(rows)
    row_msg = {
        PASS: "Dataset has enough rows for prototype automatic modelling.",
        WARNING: "Dataset is small for modelling; use results with caution.",
        FAIL: "Dataset has fewer than 100 rows, so automatic modelling is not reliable.",
    }[row_status]
    gates.append(GateResult("Modelling row-count readiness", row_status, row_msg, f"{rows:,} rows"))

    if target_column:
        if target_column not in df.columns:
            gates.append(GateResult("Model target readiness", FAIL, "Selected target column is not present in the dataset.", str(target_column)))
        elif df[target_column].dropna().nunique() < 2:
            gates.append(GateResult("Model target readiness", FAIL, "Target column has fewer than two classes/values.", str(target_column)))
        else:
            gates.append(GateResult("Model target readiness", PASS, "Target column is present and usable for model training.", str(target_column)))
            if df[target_column].dropna().nunique() <= 20:
                bal_status, minority, bal_msg = class_balance_status(df[target_column])
                gates.append(GateResult("Class-balance readiness", bal_status, bal_msg, f"minority_class={minority:.2f}%"))
    else:
        # Dataset-level analysis can still be PASS. Model-specific questions will fail/warn at answer level.
        gates.append(GateResult("Model target readiness", WARNING, "No target selected yet; EDA and Copilot summary can still work, but prediction requires a target.", "classification/regression requires target selection"))

    sensitive = _infer_sensitive_columns(df)
    if sensitive:
        gates.append(GateResult("Security/privacy readiness", WARNING, "Possible personal/sensitive fields detected; outputs must be handled carefully.", ", ".join(sensitive[:8])))
    else:
        gates.append(GateResult("Security/privacy readiness", PASS, "No obvious personal identifier columns detected by the simple prototype check.", "keyword-based scan"))

    # Transparent quality score: missingness and duplicates dominate; empty/invalid data heavily penalised.
    penalty = min(missing_pct * 2.0, 55) + min(duplicate_pct * 1.2, 30)
    if rows == 0 or cols < 2:
        penalty += 60
    if row_status == WARNING:
        penalty += 5
    elif row_status == FAIL:
        penalty += 15
    quality_score = max(0.0, round(100 - penalty, 2))

    hard_fail_gates = {"Data availability", "Schema readability", "Missing-value quality", "Duplicate-row quality"}
    if any(g.status == FAIL and g.gate in hard_fail_gates for g in gates):
        overall = FAIL
    elif any(g.status == WARNING and g.gate != "Model target readiness" for g in gates):
        overall = WARNING
    else:
        overall = PASS

    return ReadinessReport(
        dataset_id=dataset_fingerprint(df, dataset_name),
        dataset_name=dataset_name,
        rows=rows,
        columns=cols,
        memory_mb=round(memory_mb, 2),
        missing_cells=missing_cells,
        duplicate_rows=duplicate_rows,
        quality_score=quality_score,
        overall_status=overall,
        gates=gates,
        generated_at=time.time(),
    )


def status_badge_html(status: str) -> str:
    colours = {PASS: "#047857", WARNING: "#b45309", FAIL: "#b91c1c"}
    bg = {PASS: "#ecfdf5", WARNING: "#fffbeb", FAIL: "#fef2f2"}
    color = colours.get(status, "#475569")
    back = bg.get(status, "#f8fafc")
    return f"<span style='background:{back}; color:{color}; padding:0.2rem 0.55rem; border-radius:999px; font-weight:700; font-size:0.8rem'>{status}</span>"
