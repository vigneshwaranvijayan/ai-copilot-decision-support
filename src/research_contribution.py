"""Research contribution statements used by the app and dissertation notes."""
from __future__ import annotations

from typing import Dict, List
import pandas as pd

CONTRIBUTIONS: List[Dict[str, str]] = [
    {
        "id": "C1",
        "contribution": "Measurable data-readiness framework",
        "description": "A transparent PASS/WARNING/FAIL gate system mirrors the dissertation criteria for file/data availability, dataset size, missingness, duplicates, target availability/validity, class balance, data types and evidence availability.",
        "evidence": "Readiness gates table, quality score, limitation/refusal examples and exportable readiness report.",
    },
    {
        "id": "C2",
        "contribution": "Evidence-grounding mechanism",
        "description": "Each Copilot response is generated from the active dataset plus available EDA, model metrics, SHAP/fallback explanation artefacts, session context and transparent rules. Unsupported questions are limited or refused rather than guessed.",
        "evidence": "Copilot answer contract, grounding status, evidence rows, audit log and chat export.",
    },
    {
        "id": "C3",
        "contribution": "Explainability-to-action pipeline",
        "description": "Model outputs and SHAP/fallback drivers are translated into plain-English operational guidance and human-review safety warnings rather than being shown only as technical scores.",
        "evidence": "Global/local driver tables, recommendation evidence and safety-warning coverage.",
    },
    {
        "id": "C4",
        "contribution": "Short-term memory and traceable evidence management",
        "description": "Implemented Streamlit session memory keeps the active dataset, target/model context and recent question/answer context for follow-up questions. Long-term vector/database retrieval remains an explicitly labelled future/optional extension.",
        "evidence": "Session context, follow-up tests, chat history, audit records and exported evidence.",
    },
    {
        "id": "C5",
        "contribution": "Evaluation framework for decision-support quality",
        "description": "The prototype evaluates model performance, explanation quality, grounding accuracy, usability, trust, decision confidence, exportability and safety awareness.",
        "evidence": "Evaluation workspace, questionnaire CSV, model/runtime evidence and dissertation screenshots.",
    },
]


def contribution_table() -> pd.DataFrame:
    return pd.DataFrame(CONTRIBUTIONS)


def contribution_summary() -> str:
    return (
        "The project contribution is a dataset-grounded Explainable AI Copilot workflow that combines "
        "measurable readiness gates, visual/model evidence, SHAP-based explanation, controlled question answering, "
        "implemented short-term session memory, decision-support recommendations, safety warnings and exportable evidence. "
        "Long-term database/vector retrieval is presented only as a future optional architecture extension."
    )
