from __future__ import annotations

import numpy as np
import pandas as pd


def _get_transformed_feature_names(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"feature_{i}" for i in range(1000)]


def get_shap_explanation_safe(trained_result, row_index):
    """
    Attempts SHAP explanation.
    If SHAP fails due to environment/model compatibility, returns a safe fallback.
    """
    try:
        import shap

        prepared = trained_result["prepared"]
        pipe = trained_result["best_model"]
        model = pipe.named_steps["model"]
        preprocessor = pipe.named_steps["preprocessor"]

        X_train_transformed = preprocessor.transform(prepared.X_train)
        feature_names = _get_transformed_feature_names(pipe)

        X_all = pd.concat([prepared.X_train, prepared.X_test], axis=0)
        if row_index in X_all.index:
            selected_raw = X_all.loc[[row_index]]
        else:
            selected_raw = X_all.iloc[[0]]

        selected_transformed = preprocessor.transform(selected_raw)
        model_name = model.__class__.__name__.lower()

        if "forest" in model_name or "boost" in model_name or "xgb" in model_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(selected_transformed)

            if isinstance(shap_values, list):
                values = shap_values[1][0]
            else:
                arr = np.array(shap_values)
                if arr.ndim == 3:
                    values = arr[0, :, 1]
                elif arr.ndim == 2:
                    values = arr[0]
                else:
                    values = arr
        else:
            explainer = shap.LinearExplainer(model, X_train_transformed)
            shap_values = explainer.shap_values(selected_transformed)
            values = np.array(shap_values)[0]

        n = min(len(feature_names), len(values))
        top_df = pd.DataFrame(
            {
                "feature": feature_names[:n],
                "shap_value": values[:n],
                "impact": ["increases churn risk" if v > 0 else "reduces churn risk" for v in values[:n]],
            }
        )
        top_df["absolute_impact"] = top_df["shap_value"].abs()
        top_df = top_df.sort_values("absolute_impact", ascending=False).head(8)
        top_df["shap_value"] = top_df["shap_value"].round(4)
        top_df["absolute_impact"] = top_df["absolute_impact"].round(4)

        return {
            "available": True,
            "top_features": top_df[["feature", "shap_value", "impact", "absolute_impact"]],
            "message": "SHAP explanation generated successfully.",
        }

    except Exception as exc:
        return {
            "available": False,
            "top_features": pd.DataFrame(),
            "message": (
                "SHAP explanation is not available in this run. "
                f"Reason: {str(exc)}. "
                "The application can still provide model output, plain-English interpretation, and rule-based recommendations."
            ),
        }
