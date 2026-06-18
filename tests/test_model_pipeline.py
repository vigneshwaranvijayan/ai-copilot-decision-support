import pandas as pd

from src.data_utils import prepare_features
from src.model_utils import train_and_evaluate_models, predict_single_record


def make_sample_df():
    return pd.DataFrame({
        "customerID": [f"C{i:03d}" for i in range(20)],
        "Contract": ["Month-to-month", "One year"] * 10,
        "tenure": [1, 34, 2, 45, 2, 8, 22, 10, 28, 4, 60, 5, 13, 72, 3, 40, 9, 55, 6, 20],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 89.10, 29.75, 104.80, 79.65, 64.50, 84.20, 49.95, 115.50, 75.35, 70.15, 81.25, 79.20, 45.10, 90.50],
        "TotalCharges": [29.85, 1889.5, 108.15, 1840.75, 151.65, 820.5, 1949.4, 301.9, 3046.05, 318.1, 3845.2, 421.0, 633.4, 8312.75, 228.2, 2845.15, 731.25, 4382.5, 270.6, 1810.0],
        "Churn": ["No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
    })


def test_prepare_features_outputs_train_test_sets():
    df = make_sample_df()
    prepared = prepare_features(df, "Churn", ["customerID"])
    assert prepared.X_train.shape[0] > 0
    assert prepared.X_test.shape[0] > 0
    assert "Churn" == prepared.target_col


def test_train_and_predict_pipeline_runs():
    df = make_sample_df()
    prepared = prepare_features(df, "Churn", ["customerID"])
    trained = train_and_evaluate_models(prepared)
    assert "metrics_table" in trained
    assert trained["best_model"] is not None
    pred = predict_single_record(trained, df, df.index[0])
    assert 0 <= pred["positive_probability"] <= 1
