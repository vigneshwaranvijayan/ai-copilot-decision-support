PROJECT_TITLE = "Proof-of-Concept Explainable AI Copilot for Operational Business Decision Support"

PRIMARY_CASE_STUDY = {
    "name": "Customer Churn Decision Support",
    "description": (
        "Primary dissertation case study using customer churn data. "
        "The app also supports flexible CSV exploration for usability, but prediction/explainability needs a selected binary target."
    ),
    "target_column": "Churn",
    "drop_columns": ["customerID", "CustomerID", "customer_id", "id", "ID"],
    "positive_label": "Yes",
}

RISK_THRESHOLDS = {
    "high": 0.70,
    "medium": 0.40,
}

SUPPORTED_COPILOT_SCOPE = [
    "dataset summary",
    "data quality",
    "column analysis",
    "distribution charts",
    "highest and average numeric values",
    "binary target rate when target is selected",
    "model explanation when binary target model is trained",
    "recommendation generation",
    "safety and governance warning",
]
