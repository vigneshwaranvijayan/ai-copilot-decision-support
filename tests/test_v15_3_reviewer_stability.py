
import pandas as pd
from types import SimpleNamespace

from src.copilot import answer_question


def _demo_df():
    return pd.DataFrame({
        "customerid": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "churn": ["Yes", "Yes", "No", "No", "Yes", "No", "Yes", "No"],
        "tenure": [1, 3, 40, 50, 2, 60, 6, 70],
        "paymentmethod": ["electronic check", "mailed check", "bank transfer", "credit card", "electronic check", "credit card", "electronic check", "bank transfer"],
        "contract": ["month-to-month", "month-to-month", "two year", "one year", "month-to-month", "two year", "month-to-month", "two year"],
        "customerfeedback": ["slow internet service", "good service", "happy", "billing issue", "connection outage", "loyal customer", "support delay", "good package"],
    })


def _model():
    best = SimpleNamespace(
        model_name="Random Forest",
        metrics={"f1": 0.648, "roc_auc": 0.864, "recall": 0.789, "precision": 0.549},
        target_column="churn",
    )
    return SimpleNamespace(best_result=best, target_column="churn", positive_label="Yes", leaderboard=pd.DataFrame([{"model": "Random Forest"}]))


def test_tenure_relationship_not_plain_target_rate():
    r = answer_question("How does tenure relate to churn?", _demo_df(), "demo", "churn", "Yes", _model())
    assert "I compared `tenure`" in r.answer
    assert r.table is not None and "positive_rate_percent" in r.table.columns


def test_payment_method_relationship_not_plain_target_rate():
    r = answer_question("Which payment method is linked with higher churn?", _demo_df(), "demo", "churn", "Yes", _model())
    assert "paymentmethod" in r.answer.lower()
    assert "positive-rate group" in r.answer


def test_safety_warning_question_returns_safety_not_churn_rate():
    r = answer_question("What safety warning applies before acting on churn predictions?", _demo_df(), "demo", "churn", "Yes", _model())
    assert "human reviewer" in r.answer.lower()
    assert "automatic" in r.safety_warning.lower()


def test_memory_audit_mongodb_meta_questions_supported():
    for q, expected in [
        ("What does short-term memory store in this session?", "session context"),
        ("Why did we remove MongoDB from the main design?", "ChromaDB"),
        ("What does the audit trail prove?", "traceability"),
        ("How are Copilot answers logged?", "audit trail"),
    ]:
        r = answer_question(q, _demo_df(), "demo", "churn", "Yes", _model())
        assert expected.lower() in r.answer.lower()


def test_legal_advice_refused():
    r = answer_question("What happens if I ask for legal advice?", _demo_df(), "demo", "churn", "Yes", _model())
    assert "cannot provide legal advice" in r.answer.lower()
    assert "refused" in r.safety_warning.lower()


def test_dataset_memory_usage_supported():
    r = answer_question("How much memory does this dataset consume?", _demo_df(), "demo", "churn", "Yes", _model())
    assert "mb" in r.answer.lower()
    assert r.table is not None
