import pandas as pd

from src.data_utils import detect_target_column, positive_mask


def test_detect_target_column_case_insensitive():
    df = pd.DataFrame({"CustomerID": [1, 2], "churn": ["Yes", "No"]})
    assert detect_target_column(df, "Churn") == "churn"


def test_positive_mask_yes_no():
    series = pd.Series(["Yes", "No", "1", "true", "False"])
    result = positive_mask(series).tolist()
    assert result == [True, False, True, True, False]
