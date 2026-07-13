import pandas as pd

from src.visual_analytics import build_visual_dashboard, data_quality_score


def test_data_quality_score_returns_expected_fields():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "x", "y"]})
    score = data_quality_score(df)
    assert "score" in score
    assert "band" in score
    assert 0 <= score["score"] <= 100


def test_visual_dashboard_builds_figures_for_binary_dataset():
    df = pd.DataFrame({
        "Churn": ["Yes", "No", "No", "Yes"],
        "Contract": ["Month-to-month", "Two year", "One year", "Month-to-month"],
        "MonthlyCharges": [80, 40, 50, 90],
        "customerfeedback": ["expensive service", "good value", "happy", "slow support"],
    })
    dash = build_visual_dashboard(df, dataset_name="test", target_col="Churn", positive_label="Yes")
    assert dash["cards"]
    assert dash["figures"]
    assert dash["quality"]["score"] >= 0
