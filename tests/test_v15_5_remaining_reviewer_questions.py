
import pandas as pd
from types import SimpleNamespace

from src.copilot import answer_question


def _df():
    return pd.DataFrame({
        "contract": ["month-to-month", "one year", "two year", "month-to-month"],
        "monthlycharges": [85, 40, 30, 90],
        "tenure": [2, 40, 60, 5],
        "paymentmethod": ["electronic check", "bank transfer", "credit card", "electronic check"],
        "customerfeedback": ["slow internet", "good service", "happy", "connection outage"],
        "churn": ["Yes", "No", "No", "Yes"],
    })


def _model():
    best = SimpleNamespace(
        model_name="Random Forest",
        metrics={"f1": 0.65, "roc_auc": 0.86, "recall": 0.79, "precision": 0.55},
        target_column="churn",
    )
    return SimpleNamespace(best_result=best, target_column="churn", positive_label="Yes", leaderboard=pd.DataFrame([{"model":"Random Forest"}]))


def test_research_gap_is_answered_not_generic_refusal():
    r = answer_question("What research gap does this system address?", _df(), "demo", "churn", "Yes", model_output=_model())
    assert "research gap" in r.answer.lower()
    assert "could not safely map" not in r.answer.lower()


def test_chromadb_and_export_questions_are_answered():
    df = _df()
    r1 = answer_question("How is ChromaDB used in this project?", df, "demo", "churn", "Yes", model_output=_model())
    r2 = answer_question("Can I export the chat answers and evidence?", df, "demo", "churn", "Yes", model_output=_model())
    assert "semantic memory" in r1.answer.lower()
    assert "export data page" in r2.answer.lower()
    assert "could not safely map" not in r1.answer.lower() + r2.answer.lower()


def test_relationship_evidence_is_not_marked_fail_when_warning_says_not_automatic():
    r = answer_question("How does tenure relate to churn?", _df(), "demo", "churn", "Yes", model_output=_model())
    assert "I compared" in r.answer
    assert "**Answer grounding:** FAIL" not in r.answer


def test_competitor_pricing_is_safe_refusal():
    r = answer_question("What happens if I ask about competitor pricing?", _df(), "demo", "churn", "Yes", model_output=_model())
    assert "cannot answer competitor" in r.answer.lower()
    assert "refused" in r.answer.lower()
