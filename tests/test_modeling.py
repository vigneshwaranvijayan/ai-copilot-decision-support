import pandas as pd

from src.modeling import detect_binary_targets, train_models


def test_detect_binary_targets_prefers_churn():
    df = pd.DataFrame({'customer_id': [1,2,3], 'churn': ['Yes', 'No', 'No'], 'gender': ['M','F','F']})
    targets = detect_binary_targets(df)
    assert targets[0] == 'churn'


def test_train_models_on_small_binary_dataset():
    df = pd.DataFrame({
        'tenure': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        'contract': ['month', 'month', 'annual', 'annual'] * 5,
        'charges': [80,85,40,45,82,88,42,47,79,90,43,44,81,86,41,46,78,87,39,48],
        'churn': ['Yes','Yes','No','No'] * 5,
    })
    output = train_models(df, 'churn', 'Yes', include_xgboost=False, max_training_rows=1000)
    assert output.best_result is not None
    assert not output.leaderboard.empty
    assert 'f1' in output.leaderboard.columns
