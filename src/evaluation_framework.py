"""Evaluation metrics for the dissertation prototype."""
from __future__ import annotations
import pandas as pd


def evaluation_metrics_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"dimension": "Model performance", "metrics": "Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix", "how_measured": "Generated from held-out test split or sampled evaluation split during model training."},
        {"dimension": "Explanation quality", "metrics": "Explanation availability, driver coverage, explanation clarity, usefulness, stability", "how_measured": "SHAP/fallback availability plus participant Likert ratings and top-driver comparison."},
        {"dimension": "Grounding accuracy", "metrics": "Required-column match, evidence-source availability, unsupported-claim rate, refusal accuracy", "how_measured": "Test question set and manual review of whether each answer is supported by the active dataset/model/evidence."},
        {"dimension": "Trustworthiness", "metrics": "Evidence coverage, limitation coverage, safety-warning coverage, refusal behaviour", "how_measured": "Audit/export review and user questionnaire."},
        {"dimension": "Usability", "metrics": "Ease of use, clarity, task completion, perceived usefulness", "how_measured": "Likert questionnaire, optional SUS-style questions, comments and screenshots."},
        {"dimension": "Decision support", "metrics": "Understanding, decision confidence, action usefulness, human-review awareness", "how_measured": "Condition A prediction-only versus Condition B explanation + recommendation + warning."},
        {"dimension": "Performance/scalability", "metrics": "Rows processed, memory use, model runtime, chart generation behaviour", "how_measured": "Scale readiness page, runtime notes and exportable scale evidence."},
    ])
