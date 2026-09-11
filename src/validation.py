"""Dataset readiness, quality-gate and trust evidence utilities.

The core PASS/WARNING/FAIL gates intentionally mirror Table 3.1 of the
methodology so exported evidence and dissertation wording stay consistent.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import hashlib
import json
import time

import numpy as np
import pandas as pd

from .readiness_gates import (
    FAIL,
    PASS,
    WARNING,
    CORE_READINESS_GATES,
    READINESS_THRESHOLDS,
    class_balance_status,
    row_count_status,
    status_from_percent,
)


@dataclass
class GateResult:
    gate: str
    status: str
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
    cols = [str(c) for c in df.columns]
    payload = {
        "name": name,
        "shape": list(df.shape),
        "columns": cols,
        "dtypes": [str(df[c].dtype) for c in df.columns],
    }
    try:
        payload["sample"] = df.head(25).astype("string").fillna("<NA>").to_csv(index=False)
    except Exception:
        payload["sample"] = "unavailable"
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _infer_sensitive_columns(df: pd.DataFrame) -> List[str]:
    tokens = ["name", "email", "phone", "mobile", "address", "postcode", "zip", "dob", "birth", "ssn", "national", "passport"]
    service_feature_exceptions = {
        "phoneservice", "multiplelines", "internetservice", "onlinesecurity", "onlinebackup",
        "deviceprotection", "techsupport", "streamingtv", "streamingmovies",
    }
    matches = []
    for c in df.columns:
        lower = str(c).lower().replace("_", "").replace(" ", "")
        if lower in service_feature_exceptions:
            continue
        if any(t in lower for t in tokens):
            matches.append(str(c))
    return matches


def _data_type_gate(df: pd.DataFrame, target_column: Optional[str]) -> GateResult:
    """Assess whether fields can enter the standard encoding/scaling pipeline."""
    if df is None or df.empty:
        return GateResult("Data types", FAIL, "No usable columns are available for type checking.", "empty dataset")

    problematic: List[str] = []
    conversion_candidates: List[str] = []
    for col in df.columns:
        if col == target_column:
            continue
        s = df[col]
        if pd.api.types.is_complex_dtype(s):
            problematic.append(str(col))
            continue
        if pd.api.types.is_object_dtype(s):
            sample = s.dropna().head(100)
            if any(isinstance(v, (dict, list, set, tuple)) for v in sample):
                problematic.append(str(col))
                continue
            # Numeric-looking text is processable but should be cleaned/converted.
            if len(sample):
                numeric_rate = pd.to_numeric(sample, errors="coerce").notna().mean()
                if 0.8 <= numeric_rate < 1.0:
                    conversion_candidates.append(str(col))

    predictor_count = len([c for c in df.columns if c != target_column])
    if predictor_count < 1 or len(problematic) >= predictor_count:
        return GateResult("Data types", FAIL, "Major data-type issues prevent modelling.", ", ".join(problematic[:8]) or "no predictor columns")
    if problematic or conversion_candidates:
        details = []
        if problematic:
            details.append("non-scalar: " + ", ".join(problematic[:6]))
        if conversion_candidates:
            details.append("conversion suggested: " + ", ".join(conversion_candidates[:6]))
        return GateResult("Data types", WARNING, "Some columns require cleaning or conversion before reliable modelling.", "; ".join(details))
    return GateResult("Data types", PASS, "Columns can be processed after standard encoding/scaling.", f"{predictor_count} predictor column(s) checked")


def build_readiness_report(df: pd.DataFrame, dataset_name: str = "active_dataset", target_column: Optional[str] = None) -> ReadinessReport:
    rows, cols = int(df.shape[0]), int(df.shape[1])
    total_cells = max(rows * cols, 1)
    missing_cells = int(df.isna().sum().sum()) if not df.empty else 0
    missing_pct = missing_cells / total_cells * 100
    duplicate_rows = int(df.duplicated().sum()) if not df.empty else 0
    duplicate_pct = duplicate_rows / max(rows, 1) * 100
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024)) if not df.empty else 0.0

    gates: List[GateResult] = []

    # 1. File loading / data availability
    if rows == 0 or cols < int(READINESS_THRESHOLDS["column_count"]["pass_min"]):
        gates.append(GateResult("File loading / data availability", FAIL, "Dataset is empty or has too few usable columns.", f"rows={rows}, columns={cols}"))
    else:
        gates.append(GateResult("File loading / data availability", PASS, "File/data loaded correctly for the in-memory workflow.", f"rows={rows:,}, columns={cols:,}"))

    # 2. Dataset size
    row_status = row_count_status(rows)
    gates.append(GateResult(
        "Dataset size",
        row_status,
        {PASS: "Dataset has 500 or more rows.", WARNING: "Dataset has 100-499 rows; modelling can continue with caution.", FAIL: "Dataset has fewer than 100 rows."}[row_status],
        f"{rows:,} rows",
    ))

    # 3. Missing values
    miss_status = status_from_percent(missing_pct, READINESS_THRESHOLDS["missing_cells_percent"]["pass_max"], READINESS_THRESHOLDS["missing_cells_percent"]["warning_max"])
    gates.append(GateResult(
        "Missing values",
        miss_status,
        {PASS: "Missing values are within 0-5%.", WARNING: "Missing values are above 5% and up to 20%; limitations must be visible.", FAIL: "Missing values exceed 20%."}[miss_status],
        f"{missing_pct:.2f}% missing cells",
    ))

    # 4. Duplicate rows
    dup_status = status_from_percent(duplicate_pct, READINESS_THRESHOLDS["duplicate_rows_percent"]["pass_max"], READINESS_THRESHOLDS["duplicate_rows_percent"]["warning_max"])
    gates.append(GateResult(
        "Duplicate rows",
        dup_status,
        {PASS: "Duplicate rows are within 0-5%.", WARNING: "Duplicate rows are above 5% and up to 15%; review repeated records.", FAIL: "Duplicate rows exceed 15%."}[dup_status],
        f"{duplicate_pct:.2f}% duplicate rows",
    ))

    # 5-7. Target column, target validity and class balance
    target_exists = bool(target_column and target_column in df.columns)
    target_valid = False
    if not target_column:
        gates.append(GateResult("Target column", WARNING, "No target is selected yet; EDA can continue but prediction requires confirmation.", "target not selected"))
        gates.append(GateResult("Target validity", WARNING, "Target validity cannot be checked until a target is selected.", "target not selected"))
        gates.append(GateResult("Class balance", WARNING, "Class balance cannot be checked until a classification target is selected.", "target not selected"))
    elif not target_exists:
        gates.append(GateResult("Target column", FAIL, "Selected target column is not present in the dataset.", str(target_column)))
        gates.append(GateResult("Target validity", FAIL, "Target cannot be validated because the selected column is missing.", str(target_column)))
        gates.append(GateResult("Class balance", FAIL, "Class balance cannot be computed because the selected target is missing.", str(target_column)))
    else:
        gates.append(GateResult("Target column", PASS, "Selected target column is available.", str(target_column)))
        target_values = df[target_column].dropna()
        nunique = int(target_values.nunique())
        if nunique < 2:
            gates.append(GateResult("Target validity", FAIL, "Target has only one usable class/value.", f"unique_values={nunique}"))
            gates.append(GateResult("Class balance", FAIL, "Class balance is not meaningful for a one-class target.", f"unique_values={nunique}"))
        else:
            target_valid = True
            gates.append(GateResult("Target validity", PASS, "Target has at least two usable classes/values.", f"unique_values={nunique}"))
            if nunique <= 20:
                bal_status, minority, bal_msg = class_balance_status(target_values)
                gates.append(GateResult("Class balance", bal_status, bal_msg, f"minority_class={minority:.2f}%"))
            else:
                gates.append(GateResult("Class balance", WARNING, "Selected target has many unique values; class-balance classification criteria are not directly applicable.", f"unique_values={nunique}"))

    # 8. Data types
    dtype_gate = _data_type_gate(df, target_column)
    gates.append(dtype_gate)

    # 9. Evidence availability
    if rows == 0 or cols < 2:
        evidence_gate = GateResult("Evidence availability", FAIL, "Required evidence is missing.", "no usable dataset evidence")
    elif target_column and (not target_exists or not target_valid):
        evidence_gate = GateResult("Evidence availability", FAIL, "Required modelling/explanation evidence is missing because the target is unusable.", "target evidence unavailable")
    elif target_column and row_status == FAIL:
        evidence_gate = GateResult("Evidence availability", FAIL, "There are too few rows for reliable modelling/explanation evidence.", f"rows={rows}")
    elif target_column:
        evidence_gate = GateResult("Evidence availability", PASS, "Enough dataset evidence exists for charts, modelling, explanation and grounded Copilot answers.", "dataset + selected target available")
    else:
        evidence_gate = GateResult("Evidence availability", WARNING, "Dataset/visual evidence is available, but prediction/explanation evidence requires target selection.", "partial evidence: EDA only")
    gates.append(evidence_gate)

    # Supplementary privacy/security screen (does not redefine Table 3.1 status).
    sensitive = _infer_sensitive_columns(df)
    if sensitive:
        gates.append(GateResult("Supplementary privacy/security screen", WARNING, "Possible personal identifier fields detected; handle outputs carefully.", ", ".join(sensitive[:8])))
    else:
        gates.append(GateResult("Supplementary privacy/security screen", PASS, "No obvious personal identifier columns detected by the simple keyword screen.", "supplementary check"))

    # Transparent quality score. Core fail/warning states still determine the final label.
    penalty = min(missing_pct * 2.0, 55) + min(duplicate_pct * 1.2, 30)
    if rows == 0 or cols < 2:
        penalty += 60
    if row_status == WARNING:
        penalty += 5
    elif row_status == FAIL:
        penalty += 15
    if dtype_gate.status == WARNING:
        penalty += 3
    elif dtype_gate.status == FAIL:
        penalty += 15
    quality_score = max(0.0, round(100 - penalty, 2))

    core = [g for g in gates if g.gate in CORE_READINESS_GATES]
    if any(g.status == FAIL for g in core):
        overall = FAIL
    elif any(g.status == WARNING for g in core):
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
