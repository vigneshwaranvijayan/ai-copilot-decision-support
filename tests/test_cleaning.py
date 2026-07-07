import pandas as pd

from src.cleaning import clean_dataframe, clean_column_name


def test_clean_column_name():
    assert clean_column_name(' Monthly Charges (£) ') == 'monthly_charges'


def test_clean_dataframe_converts_numeric_and_drops_duplicates():
    df = pd.DataFrame({
        ' Customer ID ': [' A ', ' A ', 'B'],
        ' Total Charges ': ['1,200.50', '1,200.50', '300'],
        ' Empty ': [None, None, None],
    })
    cleaned, report = clean_dataframe(df)
    assert 'customer_id' in cleaned.columns
    assert 'total_charges' in cleaned.columns
    assert 'empty' not in cleaned.columns
    assert cleaned.shape[0] == 2
    assert 'total_charges' in report.converted_numeric_columns
