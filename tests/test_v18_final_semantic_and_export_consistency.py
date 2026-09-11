from types import SimpleNamespace
import io
import zipfile

import numpy as np
import pandas as pd

from src.copilot import answer_question
from src.explainability import ExplanationOutput
from src.modeling import train_models
from src.report import build_results_export_package
from src.validation import build_readiness_report


def _df(n=800):
    rows = []
    for i in range(n):
        contract = ["Month-to-month", "One year", "Two year"][i % 3]
        tenure = [2, 8, 18, 36, 60, 70][i % 6]
        monthly = [29.0, 49.0, 69.0, 89.0][i % 4]
        payment = ["Electronic check", "Bank transfer", "Credit card"][i % 3]
        churn = "Yes" if ((contract == "Month-to-month" and tenure <= 18) or (monthly >= 89 and i % 5 == 0)) else "No"
        rows.append({
            "customerid": f"C{i:05d}",
            "contract": contract,
            "tenure": tenure,
            "monthlycharges": monthly,
            "paymentmethod": payment,
            "totalcharges": monthly * max(tenure, 1),
            "churn": churn,
        })
    df = pd.DataFrame(rows)
    if (df["churn"] == "Yes").mean() < 0.10:
        df.loc[df.index[::5], "churn"] = "Yes"
    return df


def _model():
    lb = pd.DataFrame([
        {"model": "Logistic Regression", "accuracy": .7381, "precision": .5043, "recall": .7834, "f1": .6136, "roc_auc": .8413, "modeling_rows": 7043, "training_rows": 5634, "test_rows": 1409, "training_time_sec": .35},
        {"model": "Random Forest", "accuracy": .7757, "precision": .5694, "recall": .6364, "f1": .6010, "roc_auc": .8336, "modeling_rows": 7043, "training_rows": 5634, "test_rows": 1409, "training_time_sec": 1.55},
        {"model": "Gradient Boosting", "accuracy": .8062, "precision": .6735, "recall": .5241, "f1": .5895, "roc_auc": .8434, "modeling_rows": 7043, "training_rows": 5634, "test_rows": 1409, "training_time_sec": 1.60},
        {"model": "MLP Neural Network Baseline", "accuracy": .7892, "precision": .6110, "recall": .5668, "f1": .5881, "roc_auc": .8345, "modeling_rows": 7043, "training_rows": 5634, "test_rows": 1409, "training_time_sec": .66},
    ])
    best = SimpleNamespace(
        model_name="Logistic Regression",
        metrics={"accuracy": .7381, "precision": .5043, "recall": .7834, "f1": .6136, "roc_auc": .8413},
        confusion=np.array([[747, 288], [81, 293]]),
        target_column="churn",
    )
    return SimpleNamespace(
        best_result=best,
        results=[],
        leaderboard=lb,
        target_column="churn",
        positive_label="Yes",
        total_training_seconds=4.16,
    )


def _negative_explanation():
    global_df = pd.DataFrame([
        {"feature": "tenure", "importance": .30},
        {"feature": "monthlycharges", "importance": .21},
        {"feature": "contract_Month-to-month", "importance": .18},
    ])
    local_df = pd.DataFrame([
        {"feature": "tenure", "contribution": -.31, "absolute_contribution": .31},
        {"feature": "totalcharges", "contribution": .17, "absolute_contribution": .17},
        {"feature": "monthlycharges", "contribution": -.14, "absolute_contribution": .14},
        {"feature": "internetservice_Fiber optic", "contribution": .04, "absolute_contribution": .04},
    ])
    return ExplanationOutput(
        "SHAP", global_df, local_df, base_value=.25,
        predicted_label="No", predicted_probability=.1155, explained_row_index=0,
    )


def test_shorter_tenure_question_answers_relationship_not_overall_rate():
    r = answer_question(
        "Do customers with shorter tenure appear more likely to churn?",
        _df(), "telco.csv", "churn", "Yes", _model(), _negative_explanation(),
    )
    text = r.answer.lower()
    assert r.intent == "target_relationship"
    assert "shortest tenure" in text
    assert "more likely to churn" in text
    assert "association" in text


def test_global_shap_simple_language_defaults_to_global_not_local():
    r = answer_question(
        "Explain the SHAP results in simple business language.",
        _df(), "telco.csv", "churn", "Yes", _model(), _negative_explanation(),
    )
    text = r.answer.lower()
    assert "global shap" in text
    assert "across many records" in text
    assert "currently explained record" not in text


def test_customer_shap_simple_language_remains_local():
    r = answer_question(
        "Explain this customer's prediction in simple business language.",
        _df(), "telco.csv", "churn", "Yes", _model(), _negative_explanation(),
    )
    text = r.answer.lower()
    assert "local shap" in text
    assert "predicted as **no**" in text
    assert "11.55%" in text


def test_why_predicted_to_churn_corrects_false_premise_when_prediction_is_no():
    r = answer_question(
        "Why is this customer predicted to churn?",
        _df(), "telco.csv", "churn", "Yes", _model(), _negative_explanation(),
    )
    text = r.answer.lower()
    assert "actually predicted as **no**" in text
    assert "not** currently predicted to churn" in text or "not currently predicted to churn" in text
    assert "11.55%" in text


def test_low_risk_customer_recommendation_does_not_invent_high_risk_actions():
    r = answer_question(
        "What business action could be considered for this customer?",
        _df(), "telco.csv", "churn", "Yes", _model(), _negative_explanation(),
    )
    text = r.answer.lower()
    assert "non-churn/negative class" in text
    assert "does not justify treating this customer as high risk" in text
    assert "routine monitoring" in text
    assert "annual-contract incentives" not in text


def test_missing_columns_question_names_the_column_in_text():
    df = _df()
    df.loc[:4, "totalcharges"] = np.nan
    r = answer_question("Which columns contain missing values?", df, "telco.csv", "churn", "Yes", _model(), _negative_explanation())
    assert "`totalcharges`" in r.answer.lower()


def test_contract_influence_has_explicit_relationship_intent():
    r = answer_question("Does contract type influence churn prediction?", _df(), "telco.csv", "churn", "Yes", _model(), _negative_explanation())
    assert r.intent == "target_relationship"
    assert "month-to-month" in r.answer.lower()


def test_recommendation_evidence_followup_uses_previous_contract_context():
    df = _df()
    first = answer_question("What action could be considered for month-to-month customers?", df, "telco.csv", "churn", "Yes", _model(), _negative_explanation())
    second = answer_question("What evidence supports your recommendation?", df, "telco.csv", "churn", "Yes", _model(), _negative_explanation(), previous_context=first.context)
    text = second.answer.lower()
    assert "month-to-month customers have an observed churn rate" in text
    assert second.intent == "grounding_evidence"


def test_real_model_leaderboard_distinguishes_modeling_train_and_test_rows():
    out = train_models(_df(600), "churn", "Yes", include_xgboost=False, max_training_rows=600)
    lb = out.leaderboard
    assert {"modeling_rows", "training_rows", "test_rows"}.issubset(lb.columns)
    assert (lb["modeling_rows"] == lb["training_rows"] + lb["test_rows"]).all()
    assert (lb["test_rows"] > 0).all()
    assert (lb["training_rows"] < lb["modeling_rows"]).all()


def test_full_export_contains_official_100_and_software_and_user_eval_status():
    df = _df()
    model = _model()
    exp = _negative_explanation()
    r = answer_question("What dataset is currently loaded?", df, "telco.csv", "churn", "Yes", model, exp)
    chat = [{"dataset": "telco.csv", "question": "What dataset is currently loaded?", "response": r}]
    audit = [{"dataset": "telco.csv", "event": "test"}, {"dataset": "telco.csv", "event": "test2"}]
    perf = [{"dataset": "telco.csv", "step": "load_clean_validate_dataset", "status": "PASS", "elapsed_seconds": .1}]
    readiness = build_readiness_report(df, "telco.csv", "churn").to_dict()
    blob = build_results_export_package(
        dataset_name="telco.csv", df=df, readiness=readiness, model_output=model,
        explanation_output=exp, chat_history=chat, audit_events=audit, performance_events=perf,
    )
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = set(zf.namelist())
        for expected in [
            "copilot_100_question_results.csv",
            "copilot_100_question_summary.csv",
            "software_test_results.txt",
            "user_evaluation_questionnaire_template.csv",
            "user_evaluation_status.txt",
        ]:
            assert expected in names
        md = zf.read("full_results_report.md").decode("utf-8")
        assert "Official 100-question regression summary" in md
        assert "Packaged software test evidence" in md
        assert "User-centred evaluation status" in md
        # Markdown memory evidence must use the same audit input as the CSV/HTML.
        assert "session audit records" in md
        assert "| 2 | Traceability/audit events retained for export." in md
