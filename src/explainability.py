"""SHAP and fallback feature importance utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .modeling import ModelResult, get_feature_names, transformed_matrix

# SHAP is intentionally imported lazily. Importing it during Streamlit module
# startup adds avoidable delay before the website becomes available.
shap = None
SHAP_AVAILABLE = None

def _load_shap():
    global shap, SHAP_AVAILABLE
    if SHAP_AVAILABLE is not None:
        return shap
    try:
        import importlib
        shap = importlib.import_module("shap")
        SHAP_AVAILABLE = True
        return shap
    except Exception:  # pragma: no cover - optional/runtime environment
        shap = None
        SHAP_AVAILABLE = False
        return None


@dataclass
class ExplanationOutput:
    method: str
    global_importance: pd.DataFrame
    local_importance: pd.DataFrame
    base_value: Optional[float] = None
    warning: str = ""
    # Prediction metadata for the exact row explained.  This lets the Copilot
    # answer local questions such as "What is this customer's churn risk?"
    # from the same evidence object used for local SHAP.
    predicted_label: Optional[object] = None
    predicted_probability: Optional[float] = None
    explained_row_index: Optional[object] = None


def _extract_shap_values(values):
    """Handle shap outputs from binary classifiers across versions."""
    arr = values
    if isinstance(arr, list):
        arr = arr[1] if len(arr) > 1 else arr[0]
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # samples × features × classes
        arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


def _feature_importance_from_model(result: ModelResult) -> pd.DataFrame:
    model = result.pipeline.named_steps.get("model")
    names = get_feature_names(result.pipeline)
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        n = min(len(names), len(importances))
        return pd.DataFrame({"feature": names[:n], "importance": importances[:n]}).sort_values("importance", ascending=False)
    if hasattr(model, "coef_"):
        coefs = np.abs(np.asarray(model.coef_).ravel())
        n = min(len(names), len(coefs))
        return pd.DataFrame({"feature": names[:n], "importance": coefs[:n]}).sort_values("importance", ascending=False)
    return pd.DataFrame(columns=["feature", "importance"])


def explain_model(result: ModelResult, row: Optional[pd.DataFrame] = None, max_background: int = 250) -> ExplanationOutput:
    """Compute SHAP explanation when possible, otherwise fall back safely."""
    X_sample = result.X_test.sample(min(max_background, len(result.X_test)), random_state=42) if len(result.X_test) else result.X_test
    row_df = row[result.feature_columns] if row is not None else result.X_test.head(1)

    predicted_label = None
    predicted_probability: Optional[float] = None
    explained_row_index = row_df.index[0] if row_df is not None and len(row_df) else None
    try:
        if row_df is not None and len(row_df):
            predicted_label = result.pipeline.predict(row_df)[0]
            if hasattr(result.pipeline, "predict_proba"):
                probs = np.asarray(result.pipeline.predict_proba(row_df))[0]
                classes = list(getattr(result.pipeline, "classes_", []))
                idx = next((i for i, c in enumerate(classes) if str(c) == str(result.positive_label)), 1 if len(probs) > 1 else 0)
                predicted_probability = float(probs[idx])
    except Exception:
        predicted_label = None
        predicted_probability = None

    shap_mod = _load_shap()
    if shap_mod is not None and len(X_sample) > 1:
        try:
            model = result.pipeline.named_steps["model"]
            X_trans = np.asarray(transformed_matrix(result.pipeline, X_sample))
            row_trans = np.asarray(transformed_matrix(result.pipeline, row_df))
            feature_names = get_feature_names(result.pipeline)

            if hasattr(model, "feature_importances_"):
                explainer = shap_mod.TreeExplainer(model)
                shap_values = _extract_shap_values(explainer.shap_values(X_trans))
                row_values = _extract_shap_values(explainer.shap_values(row_trans))
                base = explainer.expected_value
                if isinstance(base, (list, np.ndarray)):
                    base_value = float(np.asarray(base).ravel()[-1])
                else:
                    base_value = float(base)
            elif hasattr(model, "coef_"):
                # Logistic Regression / linear models: exact Linear SHAP is
                # substantially faster than the generic model-agnostic explainer.
                background = X_trans[: min(100, X_trans.shape[0])]
                explainer = shap_mod.LinearExplainer(model, background)
                eval_matrix = X_trans[: min(160, X_trans.shape[0])]
                shap_values = _extract_shap_values(explainer.shap_values(eval_matrix))
                row_values = _extract_shap_values(explainer.shap_values(row_trans))
                base = getattr(explainer, "expected_value", None)
                if isinstance(base, (list, np.ndarray)):
                    base_value = float(np.asarray(base).ravel()[-1])
                elif base is not None:
                    base_value = float(base)
                else:
                    base_value = None
            else:
                # MLP/other model-agnostic path: keep the background/evaluation
                # deliberately compact so the interactive app remains responsive.
                background = shap_mod.sample(X_trans, min(30, X_trans.shape[0]), random_state=42)
                explainer = shap_mod.Explainer(model.predict_proba, background, feature_names=feature_names)
                shap_values_obj = explainer(X_trans[: min(50, X_trans.shape[0])])
                values = shap_values_obj.values
                if values.ndim == 3:
                    shap_values = values[:, :, 1]
                else:
                    shap_values = values
                row_values_obj = explainer(row_trans)
                row_values = row_values_obj.values
                if row_values.ndim == 3:
                    row_values = row_values[:, :, 1]
                base_value = None

            shap_values = np.asarray(shap_values)
            row_values = np.asarray(row_values)
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)
            if row_values.ndim == 1:
                row_values = row_values.reshape(1, -1)
            n = min(len(feature_names), shap_values.shape[1])
            global_df = pd.DataFrame({
                "feature": feature_names[:n],
                "importance": np.abs(shap_values[:, :n]).mean(axis=0),
            }).sort_values("importance", ascending=False)
            local_df = pd.DataFrame({
                "feature": feature_names[:n],
                "contribution": row_values[0, :n],
                "absolute_contribution": np.abs(row_values[0, :n]),
            }).sort_values("absolute_contribution", ascending=False)
            return ExplanationOutput(
                "SHAP", global_df, local_df, base_value=base_value,
                predicted_label=predicted_label,
                predicted_probability=predicted_probability,
                explained_row_index=explained_row_index,
            )
        except Exception as exc:
            fallback = _feature_importance_from_model(result)
            return ExplanationOutput(
                method="Fallback importance",
                global_importance=fallback,
                local_importance=fallback.rename(columns={"importance": "absolute_contribution"}).assign(contribution=np.nan),
                warning=f"SHAP could not be computed in this run, so fallback model importance is shown. Detail: {exc}",
                predicted_label=predicted_label,
                predicted_probability=predicted_probability,
                explained_row_index=explained_row_index,
            )

    fallback = _feature_importance_from_model(result)
    return ExplanationOutput(
        method="Fallback importance",
        global_importance=fallback,
        local_importance=fallback.rename(columns={"importance": "absolute_contribution"}).assign(contribution=np.nan),
        warning="SHAP package is not available or there is insufficient test data, so fallback model importance is shown.",
        predicted_label=predicted_label,
        predicted_probability=predicted_probability,
        explained_row_index=explained_row_index,
    )


def top_driver_sentences(local_importance: pd.DataFrame, top_n: int = 5) -> List[str]:
    if local_importance is None or local_importance.empty:
        return []
    rows = local_importance.head(top_n)
    sentences: List[str] = []
    for _, row in rows.iterrows():
        feature = str(row.get("feature", "feature")).replace("_", " ")
        contribution = row.get("contribution", np.nan)
        if pd.isna(contribution):
            sentences.append(f"{feature} is an important driver in the selected model.")
        elif float(contribution) >= 0:
            sentences.append(f"{feature} increases the predicted risk for this record.")
        else:
            sentences.append(f"{feature} reduces the predicted risk for this record.")
    return sentences
