import pandas as pd
from types import SimpleNamespace

from src.copilot import answer_question


def _df():
    return pd.DataFrame({
        "customerid": ["A", "B", "C", "D"],
        "churn": ["Yes", "No", "Yes", "No"],
        "tenure": [1, 60, 3, 70],
        "contract": ["month-to-month", "two year", "month-to-month", "two year"],
        "monthlycharges": [85, 40, 90, 35],
        "customerfeedback": ["slow internet", "happy", "connection outage", "good service"],
    })


def _model():
    best = SimpleNamespace(
        model_name="Random Forest",
        metrics={"f1": 0.72, "roc_auc": 0.84, "recall": 0.76, "precision": 0.69},
        target_column="churn",
    )
    return SimpleNamespace(best_result=best, target_column="churn", positive_label="Yes", leaderboard=pd.DataFrame([{"model": "Random Forest"}]))


def test_v15_4_confidence_line_added():
    r = answer_question("Is this model good enough for decision support?", _df(), "demo", "churn", "Yes", _model())
    assert "**Confidence:**" in r.answer
    assert "Model / prediction evidence" in r.answer
    assert "Random Forest" in r.answer


def test_v15_4_legal_refusal_has_safe_refusal_confidence():
    r = answer_question("What happens if I ask for legal advice?", _df(), "demo", "churn", "Yes", _model())
    assert "safe refusal" in r.answer.lower()
    assert "cannot provide legal advice" in r.answer.lower()
