from __future__ import annotations

import streamlit as st
from jinja2 import Template


REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AI Copilot Decision-Support Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 32px; line-height: 1.5; color: #111827; }
        h1, h2 { color: #0f172a; }
        .card { border: 1px solid #d1d5db; border-radius: 10px; padding: 16px; margin: 12px 0; background: #f9fafb; }
        .warning { border-left: 5px solid #f59e0b; padding: 12px; background: #fffbeb; }
        table { border-collapse: collapse; width: 100%; margin-top: 8px; }
        th, td { border: 1px solid #d1d5db; padding: 8px; text-align: left; font-size: 13px; }
        th { background: #e5e7eb; }
    </style>
</head>
<body>
    <h1>AI Copilot Decision-Support Report</h1>

    <div class="card">
        <h2>1. Prototype Scope</h2>
        <p><strong>Proof-of-concept explainable AI Copilot for operational business decision support.</strong></p>
        <p>Primary dissertation case study: customer churn analytics. Flexible CSV analysis is supported for robustness.</p>
    </div>

    <div class="card">
        <h2>2. Dataset Summary</h2>
        <p>Total records: {{ rows }}</p>
        <p>Total columns: {{ columns }}</p>
        <p>Selected target column: {{ target_col }}</p>
    </div>

    <div class="card">
        <h2>3. Model Evaluation</h2>
        {{ metrics_html }}
    </div>

    <div class="card">
        <h2>4. Selected Prediction</h2>
        <p>{{ prediction_text }}</p>
    </div>

    <div class="card">
        <h2>5. Copilot Explanation</h2>
        <p>{{ explanation_text }}</p>
    </div>

    <div class="card">
        <h2>6. Recommendations</h2>
        <ul>
        {% for rec in recommendations %}
            <li>{{ rec }}</li>
        {% endfor %}
        </ul>
    </div>

    <div class="warning">
        <h2>Decision-Support Warning</h2>
        <p>
            This system is a proof-of-concept decision-support prototype only. It should explain data patterns and model outputs, but it must not automatically make decisions.
        </p>
    </div>
</body>
</html>
"""


def build_html_report(
    df,
    target_col=None,
    trained_result=None,
    prediction_pack=None,
    explanation_pack=None,
):
    if trained_result is not None:
        metrics_html = trained_result["metrics_table"].to_html(index=False)
    else:
        metrics_html = "<p>Model has not been trained yet, or no binary target was selected.</p>"

    if prediction_pack:
        prediction_text = (
            f"Prediction: {prediction_pack['risk_label']}; "
            f"positive-class probability: {prediction_pack['positive_probability'] * 100:.2f}%."
        )
    else:
        prediction_text = "No selected prediction has been generated yet."

    if explanation_pack:
        explanation_text = explanation_pack.get("explanation_text", "No explanation available.")
        recommendations = explanation_pack.get("recommendations", [])
    else:
        explanation_text = "No explanation has been generated yet."
        recommendations = ["Use the dashboard and Copilot chat for dataset-level decision support."]

    template = Template(REPORT_TEMPLATE)
    return template.render(
        rows=df.shape[0],
        columns=df.shape[1],
        target_col=target_col or "None selected",
        metrics_html=metrics_html,
        prediction_text=prediction_text,
        explanation_text=explanation_text,
        recommendations=recommendations,
    )


def create_download_button(html_report: str, filename: str = "ai_copilot_decision_support_report.html"):
    st.download_button(
        label="Download HTML decision-support report",
        data=html_report,
        file_name=filename,
        mime="text/html",
    )
