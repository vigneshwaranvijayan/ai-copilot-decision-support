"""Research contribution statements used by the app and dissertation notes."""
from __future__ import annotations

from typing import Dict, List
import pandas as pd

CONTRIBUTIONS: List[Dict[str, str]] = [
    {
        "id": "C1",
        "contribution": "Measurable data-readiness framework",
        "description": "A transparent PASS/WARNING/FAIL gate system checks data availability, schema readability, missingness, duplicates, target readiness, class balance, security/privacy risk, and answer evidence before the Copilot responds.",
        "evidence": "Readiness gates table, quality score, refusal examples, exportable readiness report.",
    },
    {
        "id": "C2",
        "contribution": "Evidence-grounding mechanism",
        "description": "Each Copilot response is generated from an evidence package containing the active source, columns used, EDA statistics, model metrics, SHAP/fallback explanations, tables/visuals, and limitations.",
        "evidence": "Copilot answer contract, evidence rows, audit log, chat export.",
    },
    {
        "id": "C3",
        "contribution": "Explainability-to-action pipeline",
        "description": "Model outputs and SHAP/fallback drivers are translated into plain-English operational recommendations and human-review safety warnings rather than raw technical scores only.",
        "evidence": "Churn model drivers, recommendation table, safety-warning coverage.",
    },
    {
        "id": "C4",
        "contribution": "Memory-supported grounded Copilot architecture",
        "description": "Short-term session memory supports follow-up questions, while ChromaDB semantic memory stores dataset summaries, explanations, audit records and generated insights for retrieval-based grounding across sessions.",
        "evidence": "Session context, semantic memory records, retrieved evidence for later questions.",
    },
    {
        "id": "C5",
        "contribution": "Evaluation framework for decision-support quality",
        "description": "The prototype evaluates model performance, explanation quality, grounding accuracy, usability, trust, decision confidence and safety awareness using prediction-only versus explanation-supported conditions.",
        "evidence": "Evaluation workspace, questionnaire CSV, metrics summary and dissertation screenshots.",
    },
]


def contribution_table() -> pd.DataFrame:
    return pd.DataFrame(CONTRIBUTIONS)


def contribution_summary() -> str:
    return (
        "The project contribution is a dataset-grounded Explainable AI Copilot framework "
        "that combines measurable readiness gates, evidence-package generation, explainability-to-action translation, "
        "short-term and semantic long-term memory, and evaluation metrics for operational business decision support."
    )
