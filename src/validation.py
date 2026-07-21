"""Dataset readiness, quality-gate and trust evidence utilities.

The validation layer is intentionally simple and transparent so that the
prototype can explain why an answer/model is allowed, warned, or refused.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import hashlib
import json
import time

import pandas as pd


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
        # Add a tiny sample hash without hashing the full dataset.
        sample = df.head(25).astype("string").fillna("<NA>").to_csv(index=False)
        payload["sample"] = sample
    except Exception:
        payload["sample"] = "unavailable"
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _status_rank(status: str) -> int:
    return {"PASS": 0, "WARNING": 1, "FAIL": 2}.get(status, 1)


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
        gates.append(GateResult("Data availability", "FAIL", "Dataset is empty.", f"rows={rows}, columns={cols}"))
    else:
        gates.append(GateResult("Data availability", "PASS", "Dataset contains records and columns.", f"rows={rows:,}, columns={cols:,}"))

    unnamed_cols = [c for c in df.columns if str(c).strip() == "" or str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        gates.append(GateResult("Schema readability", "WARNING", "Some columns have weak or generated names.", ", ".join(map(str, unnamed_cols[:8]))))
    else:
        gates.append(GateResult("Schema readability", "PASS", "Column names are readable.", f"{cols:,} columns"))

    if missing_pct >= 40:
        gates.append(GateResult("Missing-value quality", "FAIL", "Missing values are too high for reliable automatic analysis.", f"{missing_pct:.2f}% missing cells"))
    elif missing_pct >= 10:
        gates.append(GateResult("Missing-value quality", "WARNING", "Missing values exist and should be reviewed before decision-making.", f"{missing_pct:.2f}% missing cells"))
    else:
        gates.append(GateResult("Missing-value quality", "PASS", "Missing values are within an acceptable prototype range.", f"{missing_pct:.2f}% missing cells"))

    if duplicate_pct >= 25:
        gates.append(GateResult("Duplicate-row quality", "WARNING", "High duplicate-row rate may distort summaries or models.", f"{duplicate_pct:.2f}% duplicate rows"))
    else:
        gates.append(GateResult("Duplicate-row quality", "PASS", "Duplicate-row rate is acceptable for prototype analysis.", f"{duplicate_pct:.2f}% duplicate rows"))

    if target_column:
        if target_column not in df.columns:
            gates.append(GateResult("Model target readiness", "FAIL", "Selected target column is not present in the dataset.", str(target_column)))
        elif df[target_column].dropna().nunique() < 2:
            gates.append(GateResult("Model target readiness", "FAIL", "Target column has fewer than two classes/values.", str(target_column)))
        else:
            gates.append(GateResult("Model target readiness", "PASS", "Target column is present and usable for model training.", str(target_column)))
    else:
        gates.append(GateResult("Model target readiness", "WARNING", "No target column has been selected yet; EDA and Copilot can still work.", "classification/regression requires a target"))

    sensitive = _infer_sensitive_columns(df)
    if sensitive:
        gates.append(GateResult("Security/privacy readiness", "WARNING", "Possible personal/sensitive fields detected; outputs must be handled carefully.", ", ".join(sensitive[:8])))
    else:
        gates.append(GateResult("Security/privacy readiness", "PASS", "No obvious personal identifier columns detected by the simple prototype check.", "keyword-based scan"))

    penalty = min(missing_pct * 1.2, 50) + min(duplicate_pct * 0.5, 20)
    if rows == 0 or cols == 0:
        penalty += 50
    quality_score = max(0.0, round(100 - penalty, 2))

    worst = max((_status_rank(g.status) for g in gates), default=1)
    overall = "FAIL" if any(g.status == "FAIL" for g in gates if g.gate in {"Data availability", "Missing-value quality"}) else ("WARNING" if worst >= 1 else "PASS")

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
    colours = {"PASS": "#047857", "WARNING": "#b45309", "FAIL": "#b91c1c"}
    bg = {"PASS": "#ecfdf5", "WARNING": "#fffbeb", "FAIL": "#fef2f2"}
    color = colours.get(status, "#475569")
    back = bg.get(status, "#f8fafc")
    return f"<span style='background:{back}; color:{color}; padding:0.2rem 0.55rem; border-radius:999px; font-weight:700; font-size:0.8rem'>{status}</span>"
