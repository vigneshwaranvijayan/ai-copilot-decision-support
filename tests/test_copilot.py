import pandas as pd

from src.copilot import answer_question


def test_copilot_overview_and_spelling_correction():
    df = pd.DataFrame({'contract': ['month', 'annual'], 'churn': ['Yes', 'No'], 'monthly_charges': [80, 40]})
    response = answer_question('show chrun by contrct', df, dataset_name='demo', target_column='churn', positive_label='Yes')
    assert 'demo' in response.answer
    assert response.table is not None
    assert response.corrections


def test_copilot_missing_values():
    df = pd.DataFrame({'a': [1, None], 'b': ['x', 'y']})
    response = answer_question('missng values', df, dataset_name='demo')
    assert 'missing' in response.answer.lower()
    assert response.table is not None


def test_copilot_y_target_does_not_trigger_on_type_word():
    df = pd.DataFrame({
        'education': ['primary', 'secondary', 'secondary', 'tertiary'],
        'y': ['no', 'yes', 'no', 'no'],
    })
    response = answer_question('what type of education in this dataset?', df, dataset_name='bank-full.csv', target_column='y', positive_label='yes')
    assert 'education' in response.answer.lower()
    assert response.table is not None
    assert 'secondary' in response.table['education'].astype(str).tolist()
    assert 'positive rate' not in response.answer.lower()


def test_copilot_highest_numeric_returns_ranked_rows():
    df = pd.DataFrame({
        'job': ['admin', 'tech', 'services'],
        'balance': [100, 9000, 250],
        'y': ['no', 'yes', 'no'],
    })
    response = answer_question('who have highest balance?', df, dataset_name='bank-full.csv', target_column='y', positive_label='yes')
    assert 'highest' in response.answer.lower()
    assert response.table is not None
    assert response.table.iloc[0]['balance'] == 9000
    assert response.chart is not None


def test_copilot_worst_feedback_returns_rows_not_word_summary():
    df = pd.DataFrame({
        'customerID': ['C1', 'C2', 'C3'],
        'churn': [0, 1, 0],
        'customerfeedback': [
            'I am satisfied and happy with the reliable service',
            'This is the worst service. I will cancel because the internet is slow and expensive.',
            'The company is okay but support is slow.'
        ],
    })
    response = answer_question('who gave worst customerfeedback?', df, dataset_name='telco_prep.csv', target_column='churn', positive_label=1)
    assert 'worst' in response.answer.lower() or 'negative' in response.answer.lower()
    assert response.table is not None
    assert 'customerfeedback' in response.table.columns
    assert response.table.iloc[0]['customerID'] == 'C2'
    assert 'word' not in response.table.columns


def test_copilot_followup_reuses_previous_text_column():
    df = pd.DataFrame({
        'customerID': ['C1', 'C2'],
        'customerfeedback': ['happy excellent service', 'worst bad problem cancel'],
    })
    response = answer_question('worst?', df, dataset_name='demo.csv', previous_context={'text_col': 'customerfeedback'})
    assert response.table is not None
    assert response.table.iloc[0]['customerID'] == 'C2'


def test_explicit_customerfeedback_not_overridden_by_previous_paymentmethod_context():
    df = pd.DataFrame({
        'customerID': ['C1', 'C2', 'C3'],
        'paymentmethod': ['bank transfer', 'credit card', 'bank transfer'],
        'customerfeedback': [
            'happy and satisfied with the service',
            'worst service, slow internet, expensive, cancel now',
            'good reliable service',
        ],
        'churn': [0, 1, 0],
    })
    response = answer_question(
        'which customerfeedback is worst?',
        df,
        dataset_name='telco_prep.csv',
        target_column='churn',
        positive_label=1,
        previous_context={'group_col': 'paymentmethod'},
    )
    assert response.table is not None
    assert 'customerfeedback' in response.table.columns
    assert 'paymentmethod' not in response.interpreted_question
    assert response.table.iloc[0]['customerID'] == 'C2'
    assert 'customerfeedback' in response.answer.lower()


def test_bank_short_column_y_not_selected_inside_employee_word():
    df = pd.DataFrame({
        'job': ['unemployed', 'admin.', 'unemployed', 'technician'],
        'marital': ['married', 'single', 'single', 'married'],
        'y': ['no', 'yes', 'no', 'no'],
    })
    response = answer_question('unemploye marital status?', df, dataset_name='bank-full.csv', target_column='y', positive_label='yes')
    assert response.table is not None
    assert 'marital' in response.answer.lower()
    assert 'unemployed' in response.answer.lower()
    assert 'y' not in response.table.columns
    assert set(response.table['marital'].astype(str)) == {'married', 'single'}


def test_one_letter_target_y_only_when_explicitly_asked():
    df = pd.DataFrame({
        'job': ['unemployed', 'admin.'],
        'marital': ['married', 'single'],
        'y': ['no', 'yes'],
    })
    response = answer_question('show y distribution', df, dataset_name='bank-full.csv', target_column='y', positive_label='yes')
    assert response.table is not None
    assert 'y' in response.table.columns


def test_feedback_themes_returns_word_summary_not_ranking():
    df = pd.DataFrame({
        'customerID': ['C1', 'C2'],
        'customerfeedback': ['happy satisfied service', 'worst bad problem cancel'],
    })
    response = answer_question('show feedback themes', df, dataset_name='telco.csv')
    assert response.table is not None
    assert 'word' in response.table.columns
    assert 'rank' not in response.table.columns


def test_count_categorical_value_question():
    df = pd.DataFrame({'job': ['unemployed', 'admin', 'unemployed'], 'marital': ['single', 'married', 'single']})
    response = answer_question('how many unemployed?', df, dataset_name='bank.csv')
    assert response.table is not None
    assert response.table.iloc[0]['count'] == 2
    assert 'unemployed' in response.answer.lower()


def test_unknown_question_does_not_guess_y_target():
    df = pd.DataFrame({'job': ['admin'], 'y': ['yes']})
    response = answer_question('what is the best strategy for tomorrow?', df, dataset_name='bank.csv', target_column='y', positive_label='yes')
    assert 'could not safely map' in response.answer.lower() or 'need one dataset column' in response.answer.lower()


def test_feedback_improvement_returns_business_actions():
    import pandas as pd
    from src.copilot import answer_question

    df = pd.DataFrame({
        "customerid": ["c1", "c2", "c3"],
        "churn": [1, 0, 1],
        "customerfeedback": [
            "The monthly charges are expensive and I may cancel the internet service.",
            "I am happy with the support and the price is reasonable.",
            "The internet connection is slow and unreliable, I want to switch providers.",
        ],
    })
    response = answer_question(
        "how to improve this customerfeedback?",
        df,
        dataset_name="telco_prep.csv",
        target_column="churn",
        positive_label=1,
    )
    assert "business-action themes" in response.answer
    assert response.table is not None
    assert "recommended_action" in response.table.columns
    assert any("Pricing" in str(v) or "Internet" in str(v) for v in response.table["theme"].tolist())
    assert response.context.get("topic") == "feedback_business_actions"


def test_sales_business_suggestions_from_uploaded_data():
    df = pd.DataFrame({
        "InvoiceDate": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-02-01", "2024-02-03"]),
        "Description": ["Product A", "Product B", "Product A", "Product B"],
        "Quantity": [10, 100, 40, 50],
        "UnitPrice": [5.0, 2.0, 5.0, 2.0],
    })
    response = answer_question("which products should we purchase more?", df, dataset_name="sales.csv")
    assert response.table is not None
    assert "business_suggestion" in response.table.columns
    assert any(response.table["insight_type"].astype(str).str.contains("Strong seller|Recent growth"))
    assert "sales" in response.answer.lower() or "retail" in response.answer.lower()


def test_employee_business_suggestions_safe_language():
    df = pd.DataFrame({
        "EmployeeID": [1, 2, 3, 4],
        "MonthlyIncome": [9000, 2500, 3000, 8700],
        "PerformanceRating": [2, 5, 5, 2],
        "OverTime": ["Yes", "No", "Yes", "No"],
        "JobSatisfaction": [1, 4, 2, 3],
        "Department": ["Sales", "IT", "IT", "Sales"],
        "Attrition": ["Yes", "No", "Yes", "No"],
    })
    response = answer_question("employee support and promotion suggestions", df, dataset_name="hr.csv", target_column="Attrition", positive_label="Yes")
    assert response.table is not None
    assert "business_suggestion" in response.table.columns
    text = (response.answer + " " + response.safety_warning).lower()
    assert "human" in text or "hr" in text
    assert "fire" not in text


def test_marketing_business_suggestions():
    df = pd.DataFrame({
        "job": ["admin", "admin", "student", "student", "retired", "retired"],
        "campaign": [1, 2, 1, 2, 1, 1],
        "duration": [100, 200, 300, 250, 400, 450],
        "y": ["no", "yes", "yes", "yes", "yes", "no"],
    })
    response = answer_question("marketing campaign suggestions", df, dataset_name="bank.csv", target_column="y", positive_label="yes")
    assert response.table is not None
    assert "business_suggestion" in response.table.columns
    assert "campaign" in response.answer.lower() or "marketing" in response.answer.lower()
