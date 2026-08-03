
import pandas as pd
from types import SimpleNamespace

from src.copilot import answer_question


def test_copilot_answer_shows_model_evidence_when_available():
    df = pd.DataFrame({
        "tenure": [1,2,3,4],
        "monthlycharges": [80,70,40,35],
        "churn": ["Yes","Yes","No","No"],
    })
    best = SimpleNamespace(
        model_name="Random Forest",
        metrics={"f1": 0.75, "roc_auc": 0.80, "recall": 0.70, "precision": 0.78},
        target_column="churn",
    )
    output = SimpleNamespace(best_result=best, target_column="churn", positive_label="Yes", leaderboard=pd.DataFrame([{"model":"Random Forest"}]))
    response = answer_question("is this model good enough for decision support?", df, dataset_name="demo", target_column="churn", positive_label="Yes", model_output=output)
    assert "Model / prediction evidence" in response.answer
    assert "Random Forest" in response.answer
    assert "Answer grounding" in response.answer


def test_customer_id_is_not_used_for_technical_issue_question():
    df = pd.DataFrame({
        "customerid": ["A", "B", "C"],
        "customerfeedback": ["internet is slow", "good service", "support delay and connection problem"],
        "churn": ["Yes", "No", "Yes"],
    })
    response = answer_question("any technical issue customer found?", df, dataset_name="demo", target_column="churn", positive_label="Yes")
    assert "customerfeedback" in response.answer.lower()
    assert "customerid" not in response.answer.lower().split("columns/artefacts used:")[-1][:80]
