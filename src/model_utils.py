from __future__ import annotations

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def train_and_evaluate_models(prepared):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
    }

    trained_models = {}
    metric_rows = []

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", prepared.preprocessor),
                ("model", model),
            ]
        )

        pipe.fit(prepared.X_train, prepared.y_train)

        y_pred = pipe.predict(prepared.X_test)
        y_proba = pipe.predict_proba(prepared.X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_pred

        row = {
            "model": name,
            "accuracy": round(accuracy_score(prepared.y_test, y_pred), 4),
            "precision": round(precision_score(prepared.y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(prepared.y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(prepared.y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(prepared.y_test, y_proba), 4),
        }

        metric_rows.append(row)
        trained_models[name] = pipe

    metrics_table = pd.DataFrame(metric_rows).sort_values(
        by=["f1_score", "recall", "roc_auc"],
        ascending=False,
    )

    best_model_name = metrics_table.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    best_pred = best_model.predict(prepared.X_test)
    cm = confusion_matrix(prepared.y_test, best_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual class 0", "Actual class 1"],
        columns=["Predicted class 0", "Predicted class 1"],
    )

    return {
        "prepared": prepared,
        "trained_models": trained_models,
        "metrics_table": metrics_table,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "confusion_matrix": cm_df,
    }


def predict_single_record(trained_result, raw_df: pd.DataFrame, row_index):
    prepared = trained_result["prepared"]
    best_model = trained_result["best_model"]

    X_raw = raw_df.drop(columns=[prepared.target_col], errors="ignore")
    X_raw = X_raw[prepared.feature_columns]

    selected = X_raw.loc[[row_index]]
    prediction = int(best_model.predict(selected)[0])

    if hasattr(best_model, "predict_proba"):
        proba = float(best_model.predict_proba(selected)[0][1])
    else:
        proba = float(prediction)

    if proba >= 0.70:
        risk_label = "High positive-class risk"
    elif proba >= 0.40:
        risk_label = "Medium positive-class risk"
    else:
        risk_label = "Low positive-class risk"

    if str(prepared.target_col).lower() == "churn":
        risk_label = risk_label.replace("positive-class", "churn")

    return {
        "row_index": row_index,
        "prediction": prediction,
        "positive_probability": proba,
        "risk_label": risk_label,
        "target_col": prepared.target_col,
        "positive_label": prepared.positive_label,
    }
