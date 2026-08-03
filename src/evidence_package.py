"""Evidence package generation for Copilot answers and exports."""
from __future__ import annotations
from typing import Any, Dict, Optional
import pandas as pd


def build_evidence_package(
    *,
    dataset_name: str,
    df: pd.DataFrame,
    question: str = "",
    columns_used: Optional[list[str]] = None,
    readiness: Optional[Dict[str, Any]] = None,
    model_output: Optional[Any] = None,
    explanation_output: Optional[Any] = None,
    response_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a traceable evidence package for one answer or dataset state."""
    model_info: Dict[str, Any] = {}
    if model_output is not None and getattr(model_output, "best_result", None) is not None:
        best = model_output.best_result
        model_info = {
            "target_column": getattr(model_output, "target_column", None),
            "positive_label": getattr(model_output, "positive_label", None),
            "best_model": getattr(best, "model_name", None),
            "metrics": getattr(best, "metrics", {}),
        }
    explanation_info: Dict[str, Any] = {}
    if explanation_output is not None:
        gi = getattr(explanation_output, "global_importance", None)
        explanation_info = {
            "method": getattr(explanation_output, "method", "SHAP/fallback"),
            "top_features": gi.head(10).to_dict("records") if gi is not None and not gi.empty else [],
        }
    return {
        "dataset_name": dataset_name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "question": question,
        "columns_used": columns_used or [],
        "readiness": readiness or {},
        "model": model_info,
        "explanation": explanation_info,
        "response_context": response_context or {},
    }


def evidence_package_to_text(package: Dict[str, Any]) -> str:
    """Make a concise memory document from an evidence package."""
    parts = [
        f"Dataset: {package.get('dataset_name')}",
        f"Rows: {package.get('rows')}, columns: {package.get('columns')}",
    ]
    if package.get("question"):
        parts.append(f"Question: {package.get('question')}")
    readiness = package.get("readiness") or {}
    if readiness:
        parts.append(f"Readiness: {readiness.get('overall_status')} quality={readiness.get('quality_score')}")
    model = package.get("model") or {}
    if model:
        parts.append(f"Model: target={model.get('target_column')} best={model.get('best_model')} metrics={model.get('metrics')}")
    explanation = package.get("explanation") or {}
    if explanation.get("top_features"):
        parts.append(f"Top explanation features: {explanation.get('top_features')[:3]}")
    return " | ".join(parts)
