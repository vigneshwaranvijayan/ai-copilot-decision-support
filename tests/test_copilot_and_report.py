import sys
import types
import pandas as pd

# report_utils imports streamlit for the download button.
# This small mock keeps the report-building unit test lightweight.
if "streamlit" not in sys.modules:
    mock_streamlit = types.SimpleNamespace(download_button=lambda *args, **kwargs: None)
    sys.modules["streamlit"] = mock_streamlit

from src.recommendation_utils import (
    answer_copilot_question,
    generate_recommendations,
    humanise_feature_name,
)
from src.report_utils import build_html_report


def make_sample_df():
    return pd.DataFrame({
        "customerID": ["C001", "C002", "C003", "C004"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
        "MonthlyCharges": [80.0, 50.0, 90.0, 40.0],
        "TotalCharges": [160.0, 1200.0, 180.0, 3000.0],
        "Churn": ["Yes", "No", "Yes", "No"],
    })


def test_copilot_handles_typo_question():
    df = make_sample_df()
    answer = answer_copilot_question("shw chrun by contrct", df, target_col="Churn")
    assert "interpreted" in answer["text"].lower() or "highest" in answer["text"].lower()
    assert "chart" in answer


def test_generate_recommendations_returns_warning():
    recs = generate_recommendations(shap_result=None)
    assert any("warning" in r.lower() for r in recs)


def test_humanise_feature_name_for_encoded_contract():
    assert humanise_feature_name("Contract_Month-to-month") == "Contract is Month-to-month"


def test_build_html_report_contains_scope():
    df = make_sample_df()
    html = build_html_report(df, target_col="Churn")
    assert "AI Copilot Decision-Support Report" in html
    assert "Churn" in html
