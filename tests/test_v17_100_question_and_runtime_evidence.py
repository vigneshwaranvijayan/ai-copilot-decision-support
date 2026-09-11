from pathlib import Path
from types import SimpleNamespace
import io
import zipfile

import numpy as np
import pandas as pd

from src.copilot import answer_question
from src.explainability import ExplanationOutput
from src.performance_monitor import (
    build_copilot_evaluation_summary,
    build_memory_allocation_evidence,
    build_runtime_summary,
)
from src.question_batch import split_reviewer_questions
from src.report import build_results_export_package
from src.validation import build_readiness_report


def _df(n=800):
    rows = []
    for i in range(n):
        contract = ["Month-to-month", "One year", "Two year"][i % 3]
        tenure = [2, 8, 18, 36, 60, 70][i % 6]
        monthly = [29.0, 49.0, 69.0, 89.0][i % 4]
        payment = ["Electronic check", "Bank transfer", "Credit card"][i % 3]
        # deterministic but non-trivial churn pattern
        churn = "Yes" if ((contract == "Month-to-month" and tenure <= 18) or (monthly >= 89 and i % 5 == 0)) else "No"
        rows.append({
            "customerid": f"C{i:05d}",
            "contract": contract,
            "tenure": tenure,
            "monthlycharges": monthly,
            "paymentmethod": payment,
            "churn": churn,
        })
    df = pd.DataFrame(rows)
    # Ensure the minority class comfortably passes the >=10% readiness gate.
    if (df["churn"] == "Yes").mean() < 0.10:
        df.loc[df.index[::5], "churn"] = "Yes"
    return df


def _model():
    lb = pd.DataFrame([
        {"model": "Logistic Regression", "accuracy": .7381, "precision": .5043, "recall": .7834, "f1": .6136, "roc_auc": .8413, "training_rows": 7043, "training_time_sec": .35},
        {"model": "Random Forest", "accuracy": .7757, "precision": .5694, "recall": .6364, "f1": .6010, "roc_auc": .8336, "training_rows": 7043, "training_time_sec": 1.55},
        {"model": "Gradient Boosting", "accuracy": .8062, "precision": .6735, "recall": .5241, "f1": .5895, "roc_auc": .8434, "training_rows": 7043, "training_time_sec": 1.60},
        {"model": "MLP Neural Network Baseline", "accuracy": .7892, "precision": .6110, "recall": .5668, "f1": .5881, "roc_auc": .8345, "training_rows": 7043, "training_time_sec": .66},
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
    return ExplanationOutput(
        "SHAP", global_df, local_df, base_value=.25,
        predicted_label="Yes", predicted_probability=.71, explained_row_index=0,
    )


def _questions():
    path = Path(__file__).parents[1] / "evaluation" / "reviewer_100_questions.txt"
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [line.split(". ", 1)[1] for line in lines]


def test_batch_split_handles_numbered_100_and_plain_lines():
    qs = _questions()
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(qs, 1))
    assert split_reviewer_questions(numbered) == qs
    assert split_reviewer_questions("\n".join(qs)) == qs


def test_all_100_reviewer_questions_are_supported_without_routing_failure():
    df = _df()
    model = _model()
    exp = _explanation()
    previous = None
    responses = []
    for q in _questions():
        r = answer_question(q, df, "WA_Fn-UseC_-Telco-Customer-Churn.csv", "churn", "Yes", model, exp, previous_context=previous)
        responses.append(r)
        previous = r.context
    assert len(responses) == 100
    assert all(r.grounding_status == "PASS" for r in responses)
    assert all(r.response_type != "ROUTING_FAILURE" for r in responses)


def test_critical_reviewer_answers_match_requested_evidence():
    df, model, exp = _df(), _model(), _explanation()
    cases = {
        "What dataset is currently loaded?": ["wa_fn-usec_-telco-customer-churn.csv"],
        "What percentage of customers did not churn?": ["non-churn class"],
        "Are the data types suitable for modelling?": ["data-type suitability", "pass"],
        "Explain the confusion matrix.": ["tn=747", "fp=288", "fn=81", "tp=293"],
        "How many true positives are there?": ["293 true positives"],
        "What is the accuracy of Logistic Regression?": ["logistic regression", "0.738"],
        "What is the performance of Gradient Boosting?": ["gradient boosting", "0.806", "0.843"],
        "What is the performance of the MLP model?": ["mlp neural network baseline", "0.789"],
        "Why is recall important for churn prediction?": ["actual churn", "false negatives"],
        "What does the F1-score mean in this analysis?": ["harmonic mean", "precision", "recall"],
        "What does ROC-AUC tell us?": ["separates churn from non-churn", "threshold"],
        "Can feature importance prove that a feature causes churn?": ["no", "cannot prove"],
        "What limitations should I consider when interpreting SHAP?": ["causal", "model"],
        "What is this customer's predicted churn risk?": ["71.00%"],
        "Is this customer predicted to churn?": ["predicted as", "yes"],
        "What are the main factors increasing this customer's churn risk?": ["local shap", "increase"],
        "What factors reduce this customer's churn risk?": ["reduce", "local shap"],
        "Which feature contributed most to this customer's prediction?": ["strongest local contributor"],
        "Should a human review this recommendation before taking action?": ["yes", "human should review"],
        "Should the business automatically cancel high-risk customers?": ["no", "not automatically cancel"],
        "Can I make a final business decision using only this model?": ["no", "should not"],
    }
    for q, needles in cases.items():
        r = answer_question(q, df, "WA_Fn-UseC_-Telco-Customer-Churn.csv", "churn", "Yes", model, exp)
        text = r.answer.lower()
        for needle in needles:
            assert needle.lower() in text, (q, needle, text)


def test_safety_refusals_are_pass_with_separate_response_type():
    df, model, exp = _df(), _model(), _explanation()
    for q in [
        "Can you guarantee this customer will churn?",
        "Can this model replace a manager's decision?",
        "Should the business automatically cancel high-risk customers?",
        "Can I make a final business decision using only this model?",
        "Can you use external customer complaints that are not in this dataset?",
    ]:
        r = answer_question(q, df, "telco.csv", "churn", "Yes", model, exp)
        assert r.grounding_status == "PASS"
        assert r.response_type == "SAFE_REFUSAL"
        assert r.limitation_or_refusal is True


def test_runtime_memory_and_evaluation_evidence_are_exported():
    df, model, exp = _df(), _model(), _explanation()
    r1 = answer_question("What dataset is currently loaded?", df, "telco.csv", "churn", "Yes", model, exp)
    r2 = answer_question("Can you guarantee this customer will churn?", df, "telco.csv", "churn", "Yes", model, exp)
    chat_history = [
        {"dataset": "telco.csv", "question": "What dataset is currently loaded?", "response": r1},
        {"dataset": "telco.csv", "question": "Can you guarantee this customer will churn?", "response": r2},
    ]
    perf = [
        {"dataset": "telco.csv", "step": "load_clean_validate_dataset", "status": "PASS", "elapsed_seconds": .4, "process_rss_start_mb": 100.0, "process_rss_end_mb": 102.0},
        {"dataset": "telco.csv", "step": "auto_train_and_explain", "status": "PASS", "elapsed_seconds": 20.0, "process_rss_start_mb": 102.0, "process_rss_end_mb": 180.0},
        {"dataset": "telco.csv", "step": "copilot_answer_generation", "status": "PASS", "elapsed_seconds": .1, "process_rss_start_mb": 180.0, "process_rss_end_mb": 180.2},
    ]
    readiness = build_readiness_report(df, "telco.csv", "churn").to_dict()
    blob = build_results_export_package(
        dataset_name="telco.csv", df=df, readiness=readiness, model_output=model,
        explanation_output=exp, chat_history=chat_history, performance_events=perf,
    )
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = set(zf.namelist())
        for expected in [
            "runtime_process_evidence.csv",
            "runtime_summary.csv",
            "memory_allocation_evidence.csv",
            "copilot_evaluation_summary.csv",
            "copilot_chat_answers.csv",
        ]:
            assert expected in names
        chat = pd.read_csv(zf.open("copilot_chat_answers.csv"))
        assert "response_type" in chat.columns
        eval_df = pd.read_csv(zf.open("copilot_evaluation_summary.csv"))
        assert "safe_refusal_count" in set(eval_df["metric"])
        mem_df = pd.read_csv(zf.open("memory_allocation_evidence.csv"))
        assert "current process RSS" in set(mem_df["component"])
        runtime_df = pd.read_csv(zf.open("runtime_summary.csv"))
        assert "recorded_computational_time_seconds" in set(runtime_df["metric"])


def test_summary_helpers_return_reviewer_friendly_metrics():
    perf = pd.DataFrame([
        {"step": "copilot_answer_generation", "elapsed_seconds": .1},
        {"step": "copilot_answer_generation", "elapsed_seconds": .3},
        {"step": "auto_train_and_explain", "elapsed_seconds": 20.0},
    ])
    runtime = build_runtime_summary(perf, _model())
    assert "copilot_average_seconds" in set(runtime["metric"])
    mem = build_memory_allocation_evidence(df=_df(), model_output=_model(), explanation_output=_explanation())
    assert "TOTAL tracked artefact memory" in set(mem["component"])
    chat_df = pd.DataFrame([
        {"grounding_status": "PASS", "response_type": "GROUNDED_ANSWER"},
        {"grounding_status": "PASS", "response_type": "SAFE_REFUSAL"},
    ])
    summary = build_copilot_evaluation_summary(chat_df)
    assert float(summary.loc[summary["metric"] == "pass_rate_percent", "value"].iloc[0]) == 100.0
