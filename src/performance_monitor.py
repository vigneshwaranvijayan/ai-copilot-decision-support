"""Runtime, process-memory and evaluation evidence helpers.

The dissertation prototype should be able to export evidence about how long
computational stages took and what memory was allocated to the active artefacts.
These helpers deliberately distinguish:

* dataset/artefact memory (measured from pandas/numpy objects or serialised size),
* process RSS/peak RSS (whole Python process memory), and
* short-term session-memory contents (active dataset, model/explanation state and
  chat context), which is the implemented memory scope of the dissertation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import os
import pickle
import sys
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:  # optional but included in requirements for cross-platform RSS measurement
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def process_rss_mb() -> float:
    """Current resident-set size of the Python process in MiB."""
    try:
        if psutil is not None:
            return round(float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024), 3)
    except Exception:
        pass
    return float("nan")


def process_peak_rss_mb() -> float:
    """Best-effort peak resident memory in MiB.

    On Unix/macOS resource.ru_maxrss is used. On platforms where a reliable peak
    is unavailable, the current RSS is returned and labelled as best-effort by
    the caller.
    """
    try:
        import resource
        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        if sys.platform == "darwin":
            return round(raw / (1024 * 1024), 3)
        return round(raw / 1024, 3)
    except Exception:
        return process_rss_mb()


def performance_start() -> Dict[str, float]:
    """Capture a start sample for one timed operation."""
    import time
    return {"start_perf": time.perf_counter(), "start_rss_mb": process_rss_mb()}


def dataframe_memory_mb(df: Optional[pd.DataFrame]) -> float:
    if df is None:
        return 0.0
    try:
        return round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 4)
    except Exception:
        return 0.0


def _serialised_size_mb(obj: Any) -> float:
    if obj is None:
        return 0.0
    try:
        return round(len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)) / (1024 * 1024), 4)
    except Exception:
        return 0.0


def _model_runtime_memory_mb(model_output: Any) -> float:
    """Approximate retained model/evaluation artefact memory.

    This includes serialised fitted pipelines plus retained X_test/y_test,
    predictions/probabilities and leaderboard data. It is an artefact-memory
    estimate, not a full Python-heap measurement.
    """
    if model_output is None:
        return 0.0
    total = 0.0
    try:
        total += dataframe_memory_mb(getattr(model_output, "leaderboard", None))
        for result in getattr(model_output, "results", []) or []:
            total += _serialised_size_mb(getattr(result, "pipeline", None))
            total += dataframe_memory_mb(getattr(result, "X_test", None))
            y_test = getattr(result, "y_test", None)
            if isinstance(y_test, pd.Series):
                total += float(y_test.memory_usage(deep=True)) / (1024 * 1024)
            for arr_name in ["y_pred", "y_proba", "confusion"]:
                arr = getattr(result, arr_name, None)
                if arr is not None:
                    try:
                        total += float(np.asarray(arr).nbytes) / (1024 * 1024)
                    except Exception:
                        pass
    except Exception:
        pass
    return round(total, 4)


def build_memory_allocation_evidence(
    *,
    df: pd.DataFrame,
    readiness: Optional[Dict[str, Any]] = None,
    model_output: Any = None,
    explanation_output: Any = None,
    chat_df: Optional[pd.DataFrame] = None,
    audit_df: Optional[pd.DataFrame] = None,
    perf_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build reviewer-friendly memory-allocation evidence."""
    chat_df = chat_df if chat_df is not None else pd.DataFrame()
    audit_df = audit_df if audit_df is not None else pd.DataFrame()
    perf_df = perf_df if perf_df is not None else pd.DataFrame()
    readiness_df = pd.DataFrame((readiness or {}).get("gates", []))
    global_imp = getattr(explanation_output, "global_importance", pd.DataFrame()) if explanation_output is not None else pd.DataFrame()
    local_imp = getattr(explanation_output, "local_importance", pd.DataFrame()) if explanation_output is not None else pd.DataFrame()

    rows = [
        {
            "memory_scope": "implemented short-term session memory",
            "component": "active cleaned dataset",
            "measurement_type": "pandas deep memory",
            "memory_mb": dataframe_memory_mb(df),
            "record_count": int(len(df)),
            "notes": "Dataset retained in the current Streamlit session.",
        },
        {
            "memory_scope": "implemented short-term session memory",
            "component": "trained model/evaluation artefacts",
            "measurement_type": "retained/serialised artefact estimate",
            "memory_mb": _model_runtime_memory_mb(model_output),
            "record_count": int(len(getattr(model_output, "results", []) or [])) if model_output is not None else 0,
            "notes": "Approximate fitted pipelines + test/prediction artefacts; not total Python heap.",
        },
        {
            "memory_scope": "implemented short-term session memory",
            "component": "global explanation table",
            "measurement_type": "pandas deep memory",
            "memory_mb": dataframe_memory_mb(global_imp),
            "record_count": int(len(global_imp)) if global_imp is not None else 0,
            "notes": "SHAP/fallback global feature evidence.",
        },
        {
            "memory_scope": "implemented short-term session memory",
            "component": "local explanation table",
            "measurement_type": "pandas deep memory",
            "memory_mb": dataframe_memory_mb(local_imp),
            "record_count": int(len(local_imp)) if local_imp is not None else 0,
            "notes": "Currently explained record evidence.",
        },
        {
            "memory_scope": "implemented short-term session memory",
            "component": "Copilot chat/evaluation records",
            "measurement_type": "pandas deep memory",
            "memory_mb": dataframe_memory_mb(chat_df),
            "record_count": int(len(chat_df)),
            "notes": "Questions, answers, intent, grounding, confidence, response type and safety fields.",
        },
        {
            "memory_scope": "implemented short-term session memory",
            "component": "readiness evidence",
            "measurement_type": "pandas deep memory",
            "memory_mb": dataframe_memory_mb(readiness_df),
            "record_count": int(len(readiness_df)),
            "notes": "PASS/WARNING/FAIL gate evidence.",
        },
        {
            "memory_scope": "implemented short-term session memory",
            "component": "session audit records",
            "measurement_type": "pandas deep memory",
            "memory_mb": dataframe_memory_mb(audit_df),
            "record_count": int(len(audit_df)),
            "notes": "Traceability/audit events retained for export.",
        },
        {
            "memory_scope": "implemented short-term session memory",
            "component": "runtime/process records",
            "measurement_type": "pandas deep memory",
            "memory_mb": dataframe_memory_mb(perf_df),
            "record_count": int(len(perf_df)),
            "notes": "Per-stage time and process-memory observations.",
        },
        {
            "memory_scope": "whole Python process",
            "component": "current process RSS",
            "measurement_type": "resident set size",
            "memory_mb": process_rss_mb(),
            "record_count": 1,
            "notes": "Includes Python interpreter, libraries, Streamlit, models, data and other live process objects.",
        },
        {
            "memory_scope": "whole Python process",
            "component": "peak process RSS",
            "measurement_type": "best-effort peak resident set size",
            "memory_mb": process_peak_rss_mb(),
            "record_count": 1,
            "notes": "Best-effort process peak; platform implementation may differ.",
        },
    ]
    out = pd.DataFrame(rows)
    tracked_mask = out["memory_scope"].eq("implemented short-term session memory")
    tracked_total = round(float(out.loc[tracked_mask, "memory_mb"].fillna(0).sum()), 4)
    out = pd.concat([out, pd.DataFrame([{
        "memory_scope": "implemented short-term session memory",
        "component": "TOTAL tracked artefact memory",
        "measurement_type": "sum of tracked components",
        "memory_mb": tracked_total,
        "record_count": int(out.loc[tracked_mask, "record_count"].fillna(0).sum()),
        "notes": "Does not equal process RSS because Python/library/runtime overhead is excluded.",
    }])], ignore_index=True)
    return out


def build_runtime_summary(perf_df: pd.DataFrame, model_output: Any = None) -> pd.DataFrame:
    """Summarise computational process time; user typing/idle time is excluded."""
    if perf_df is None or perf_df.empty or "elapsed_seconds" not in perf_df.columns:
        return pd.DataFrame([{
            "metric": "recorded_computational_time_seconds",
            "value": 0.0,
            "notes": "No performance events were recorded yet.",
        }])
    work = perf_df.copy()
    work["elapsed_seconds"] = pd.to_numeric(work["elapsed_seconds"], errors="coerce").fillna(0.0)
    chat = work[work.get("step", pd.Series(index=work.index, dtype=str)).astype(str) == "copilot_answer_generation"]
    # auto_train_and_explain already contains model training, so model training
    # must not be added a second time to the computational-event total.
    rows = [
        {"metric": "recorded_event_count", "value": int(len(work)), "notes": "All timed computational events currently retained."},
        {"metric": "recorded_computational_time_seconds", "value": round(float(work["elapsed_seconds"].sum()), 3), "notes": "Sum of recorded non-idle workflow events; excludes time while the user is reading/typing."},
        {"metric": "copilot_question_count", "value": int(len(chat)), "notes": "Number of individually timed Copilot answer operations."},
        {"metric": "copilot_total_seconds", "value": round(float(chat["elapsed_seconds"].sum()), 3), "notes": "Total Copilot computational answer time."},
        {"metric": "copilot_average_seconds", "value": round(float(chat["elapsed_seconds"].mean()), 3) if len(chat) else 0.0, "notes": "Mean answer-generation time per question."},
        {"metric": "copilot_max_seconds", "value": round(float(chat["elapsed_seconds"].max()), 3) if len(chat) else 0.0, "notes": "Slowest recorded Copilot answer."},
    ]
    for step, label in [
        ("load_clean_validate_dataset", "load_clean_validate_seconds"),
        ("auto_train_and_explain", "auto_train_and_explain_seconds"),
        ("manual_classification_training", "manual_classification_training_seconds"),
        ("manual_shap_or_fallback_explanation", "manual_explanation_seconds"),
        ("build_full_results_export_package", "export_build_seconds"),
    ]:
        vals = work.loc[work["step"].astype(str) == step, "elapsed_seconds"] if "step" in work.columns else pd.Series(dtype=float)
        if len(vals):
            rows.append({"metric": label, "value": round(float(vals.sum()), 3), "notes": f"Sum of `{step}` events."})
    if model_output is not None:
        rows.append({
            "metric": "four_model_training_seconds",
            "value": round(float(getattr(model_output, "total_training_seconds", 0.0) or 0.0), 3),
            "notes": "Internal training time for the assessed four-model classification set; nested inside the auto/manual training event.",
        })
    return pd.DataFrame(rows)


def build_copilot_evaluation_summary(chat_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise actual exported Copilot evaluation outcomes."""
    if chat_df is None or chat_df.empty:
        return pd.DataFrame([{"metric": "copilot_evaluation_records", "value": 0, "notes": "No Copilot answers saved yet."}])
    statuses = chat_df.get("grounding_status", pd.Series(dtype=str)).astype(str)
    types = chat_df.get("response_type", pd.Series(dtype=str)).astype(str)
    total = int(len(chat_df))
    pass_count = int((statuses == "PASS").sum())
    warning_count = int((statuses == "WARNING").sum())
    fail_count = int((statuses == "FAIL").sum())
    rows = [
        {"metric": "copilot_evaluation_records", "value": total, "notes": "Saved question/answer evaluation rows for the active dataset."},
        {"metric": "grounding_PASS", "value": pass_count, "notes": "Grounded answer or correctly grounded safety/limitation behaviour."},
        {"metric": "grounding_WARNING", "value": warning_count, "notes": "Partial evidence; reviewer caution required."},
        {"metric": "grounding_FAIL", "value": fail_count, "notes": "Routing/evidence failure that requires correction or clarification."},
        {"metric": "pass_rate_percent", "value": round(pass_count / max(total, 1) * 100, 2), "notes": "PASS / total saved evaluation records."},
        {"metric": "safe_refusal_count", "value": int((types == "SAFE_REFUSAL").sum()), "notes": "Correct safety refusals are separated from failures."},
        {"metric": "supported_limitation_count", "value": int((types == "SUPPORTED_LIMITATION").sum()), "notes": "Correct evidence-unavailable/limitation responses."},
        {"metric": "routing_failure_count", "value": int((types == "ROUTING_FAILURE").sum()), "notes": "Questions that could not be mapped to a supported intent."},
    ]
    return pd.DataFrame(rows)
