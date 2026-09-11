import io
import zipfile
from types import SimpleNamespace

import pandas as pd

from src.business_insights import detect_business_domain
from src.copilot import answer_question
from src.explainability import ExplanationOutput
from src.report import build_results_export_package
from src.validation import build_readiness_report


def _df(n=600):
    base = pd.DataFrame({
        "customerid": [f"C{i:04d}" for i in range(n)],
        "contract": (["Month-to-month", "One year", "Two year"] * ((n + 2)//3))[:n],
        "monthlycharges": ([90.0, 55.0, 35.0, 75.0] * ((n + 3)//4))[:n],
        "tenure": ([2, 14, 48, 5, 60, 9] * ((n + 5)//6))[:n],
        "paymentmethod": (["Electronic check", "Bank transfer", "Credit card"] * ((n + 2)//3))[:n],
    })
    # 25% positive, so the dissertation class-balance gate should pass.
    base["churn"] = ["Yes" if i % 4 == 0 else "No" for i in range(n)]
    return base


def _model():
    lb = pd.DataFrame([
        {"model": "Logistic Regression", "accuracy": .738, "precision": .504, "recall": .783, "f1": .614, "roc_auc": .841},
        {"model": "Random Forest", "accuracy": .776, "precision": .569, "recall": .636, "f1": .601, "roc_auc": .834},
        {"model": "Gradient Boosting", "accuracy": .806, "precision": .674, "recall": .524, "f1": .590, "roc_auc": .843},
        {"model": "MLP Neural Network Baseline", "accuracy": .789, "precision": .611, "recall": .567, "f1": .588, "roc_auc": .835},
    ])
    best = SimpleNamespace(
        model_name="Logistic Regression",
        metrics={"accuracy": .738, "precision": .504, "recall": .783, "f1": .614, "roc_auc": .841},
        confusion=[[747, 288], [81, 293]],
        target_column="churn",
    )
    return SimpleNamespace(best_result=best, leaderboard=lb, target_column="churn", positive_label="Yes", total_training_seconds=4.2)


def _explanation():
    global_df = pd.DataFrame([
        {"feature": "tenure", "importance": .30},
        {"feature": "monthlycharges", "importance": .21},
        {"feature": "contract_Month-to-month", "importance": .18},
    ])
    local_df = pd.DataFrame([
        {"feature": "contract_Month-to-month", "contribution": .42, "absolute_contribution": .42},
        {"feature": "tenure", "contribution": -.31, "absolute_contribution": .31},
        {"feature": "monthlycharges", "contribution": .20, "absolute_contribution": .20},
    ])
    return ExplanationOutput("SHAP", global_df, local_df, base_value=.25)


def test_high_risk_does_not_route_to_hr_domain():
    df = _df()
    assert detect_business_domain(df, "What action for high-risk customers?") == "customer_churn"
    r = answer_question("What action should the business consider for high-risk customers?", df, "telco.csv", "churn", "Yes", _model(), _explanation())
    assert "employee" not in r.answer.lower()
    assert "retention" in r.answer.lower() or "customer" in r.answer.lower()


def test_readiness_matches_dissertation_core_gates_and_class_fail_propagates():
    rep = build_readiness_report(_df(), "telco.csv", "churn")
    names = {g.gate for g in rep.gates}
    expected = {
        "File loading / data availability", "Dataset size", "Missing values", "Duplicate rows",
        "Target column", "Target validity", "Class balance", "Data types", "Evidence availability",
    }
    assert expected.issubset(names)
    assert rep.overall_status == "PASS"

    bad = _df(600)
    bad["churn"] = ["Yes"] * 20 + ["No"] * 580  # 3.33% minority => FAIL
    rep_bad = build_readiness_report(bad, "bad.csv", "churn")
    assert any(g.gate == "Class balance" and g.status == "FAIL" for g in rep_bad.gates)
    assert rep_bad.overall_status == "FAIL"


def test_fallback_is_not_falsely_pass_high_confidence():
    r = answer_question("What is the best strategy for the moon tomorrow?", _df(), "telco.csv", "churn", "Yes", _model(), _explanation())
    assert r.grounding_status == "FAIL"
    assert "LOW" in r.confidence
    assert "**Answer grounding:** FAIL" in r.answer


def test_compare_models_and_f1_first_wording():
    r = answer_question("Compare Logistic Regression, Random Forest, Gradient Boosting and MLP.", _df(), "telco.csv", "churn", "Yes", _model(), _explanation())
    assert r.grounding_status == "PASS"
    assert "four assessed models" in r.answer.lower()
    assert "logistic regression" in r.answer.lower()
    # The preferred model is F1-first, while Gradient Boosting has stronger ROC-AUC.
    r2 = answer_question("Which model performed best?", _df(), "telco.csv", "churn", "Yes", _model(), _explanation())
    assert "f1-first" in r2.answer.lower()


def test_monthly_charges_relationship_uses_relationship_evidence():
    r = answer_question("Do monthly charges appear related to churn?", _df(), "telco.csv", "churn", "Yes", _model(), _explanation())
    assert "monthlycharges" in r.answer.lower()
    assert r.table is not None and "positive_rate_percent" in r.table.columns
    assert "not proof of causation" in r.answer.lower()


def test_local_prediction_and_directional_shap_answers():
    df = _df()
    exp = _explanation()
    r = answer_question("Why is this customer predicted to churn?", df, "telco.csv", "churn", "Yes", _model(), exp)
    assert r.context.get("topic") == "local_prediction_explanation"
    assert "currently explained record" in r.answer.lower()
    assert r.table is not None and "contribution" in r.table.columns

    inc = answer_question("Which features increase churn risk?", df, "telco.csv", "churn", "Yes", _model(), exp)
    assert (inc.table["contribution"] > 0).all()
    dec = answer_question("Which features reduce churn risk?", df, "telco.csv", "churn", "Yes", _model(), exp)
    assert (dec.table["contribution"] < 0).all()


def test_simple_business_language_followup_keeps_shap_context():
    df = _df()
    exp = _explanation()
    first = answer_question("What are the most important churn drivers?", df, "telco.csv", "churn", "Yes", _model(), exp)
    follow = answer_question("Can you explain that again in simple business language?", df, "telco.csv", "churn", "Yes", _model(), exp, previous_context=first.context)
    assert "shap" in follow.answer.lower()
    assert "simple business language" in follow.answer.lower()
    assert "positive rate for `churn`" not in follow.answer.lower()


def test_readiness_typo_is_answered_not_fallback():
    r = answer_question("is this datset good for modling?", _df(), "telco.csv", "churn", "Yes", _model(), _explanation())
    assert r.context.get("topic") == "readiness_evidence"
    assert r.grounding_status == "PASS"
    assert "data-readiness result" in r.answer.lower()


def test_limitations_evidence_export_and_grounding_meta_questions():
    df = _df()
    m = _model()
    e = _explanation()
    for q, needle in [
        ("What are the limitations of this prediction?", "probabilistic"),
        ("What evidence did you use for this answer?", "active cleaned dataset"),
        ("Show the data-readiness evidence.", "readiness"),
        ("What can be exported from this analysis?", "full results zip"),
        ("Why is this answer grounded in the dataset?", "controlled copilot"),
    ]:
        r = answer_question(q, df, "telco.csv", "churn", "Yes", m, e, previous_context={"topic": "model"})
        assert needle in r.answer.lower(), q
        assert "could not safely map" not in r.answer.lower()


def test_overreliance_and_external_evidence_are_explicitly_refused():
    df = _df()
    m = _model()
    e = _explanation()
    questions = [
        "Can you guarantee this customer will churn?",
        "Can this model replace a manager’s decision?",
        "Can you use external customer complaints that are not in this dataset?",
        "Can you recommend automatic cancellation for high-risk customers?",
    ]
    for q in questions:
        r = answer_question(q, df, "telco.csv", "churn", "Yes", m, e)
        assert r.grounding_status == "PASS", q
        assert r.limitation_or_refusal is True
        assert r.response_type == "SAFE_REFUSAL"
        assert "HIGH" in r.confidence


def test_export_package_has_explicit_chat_status_and_local_explanation_csv():
    df = _df()
    m = _model()
    e = _explanation()
    r = answer_question("Why is this customer predicted to churn?", df, "telco.csv", "churn", "Yes", m, e)
    readiness = build_readiness_report(df, "telco.csv", "churn").to_dict()
    blob = build_results_export_package(
        dataset_name="telco.csv", df=df, readiness=readiness, model_output=m,
        explanation_output=e, chat_history=[{"dataset": "telco.csv", "question": "Why is this customer predicted to churn?", "response": r}],
    )
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = set(zf.namelist())
        assert "local_prediction_explanation_drivers.csv" in names
        chat = pd.read_csv(zf.open("copilot_chat_answers.csv"))
        for col in ["intent", "grounding_status", "grounding_reason", "confidence", "limitation_or_refusal"]:
            assert col in chat.columns
        assert chat.loc[0, "grounding_status"] == "PASS"
