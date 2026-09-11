from pathlib import Path

from src.copilot import answer_question
from tests.test_v18_final_semantic_and_export_consistency import _df, _model, _negative_explanation


def _core_answer(r):
    marker = "**Evidence-grounded answer:**"
    text = r.answer
    return text.split(marker, 1)[1] if marker in text else text


def test_sequential_customer_simple_language_keeps_local_context():
    df, model, exp = _df(), _model(), _negative_explanation()
    prev = answer_question("What factors reduce this customer's churn risk?", df, "telco.csv", "churn", "Yes", model, exp)
    r = answer_question("Explain this customer's prediction in simple business language.", df, "telco.csv", "churn", "Yes", model, exp, previous_context=prev.context)
    text = _core_answer(r).lower()
    assert "local shap" in text
    assert "predicted as **no**" in text
    assert "global shap shows" not in text


def test_short_tenure_business_consider_question_returns_action_not_only_relationship():
    r = answer_question("What should the business consider if short tenure is a major churn driver?", _df(), "telco.csv", "churn", "Yes", _model(), _negative_explanation())
    text = _core_answer(r).lower()
    assert r.intent == "recommendation"
    assert "strengthening onboarding" in text
    assert "early-life support" in text
    assert "do not assume tenure itself causes churn" in text


def test_recommendation_limitations_question_returns_limitations_not_repeated_action():
    df, model, exp = _df(), _model(), _negative_explanation()
    prev = answer_question("What action could be considered for month-to-month customers?", df, "telco.csv", "churn", "Yes", model, exp)
    r = answer_question("What limitations apply to this recommendation?", df, "telco.csv", "churn", "Yes", model, exp, previous_context=prev.context)
    text = _core_answer(r).lower()
    assert r.intent == "recommendation_limitations"
    assert "association/model evidence rather than causal proof" in text
    assert "model can make errors" in text
    assert "human must check" in text


def test_fast_launcher_skips_pip_on_normal_launch_and_core_requirements_exclude_future_packages():
    root = Path(__file__).parents[1]
    bat = (root / "run_app.bat").read_text(encoding="utf-8").lower()
    req = (root / "requirements.txt").read_text(encoding="utf-8").lower()
    assert ".runtime_ready" in bat
    assert "pip install --upgrade pip" not in bat
    assert "--server.filewatchertype none" in bat
    for optional in ["chromadb", "psycopg2", "sqlalchemy", "pypdf", "python-docx", "pytest"]:
        assert optional not in req


def test_silent_launcher_is_packaged():
    root = Path(__file__).parents[1]
    assert (root / "run_app_silent.vbs").exists()
