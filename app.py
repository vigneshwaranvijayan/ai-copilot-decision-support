import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.config import PRIMARY_CASE_STUDY
from src.data_utils import (
    SUPPORTED_FILE_TYPES,
    create_dashboard_metrics,
    data_quality_report,
    detect_target_column,
    is_binary_target,
    load_from_api_url,
    load_tabular_file,
    prepare_features,
)
from src.explain_utils import get_shap_explanation_safe
from src.model_utils import predict_single_record, train_and_evaluate_models
from src.recommendation_utils import (
    answer_copilot_question,
    generate_plain_english_explanation,
    generate_recommendations,
    supported_question_examples,
)
from src.report_utils import build_html_report, create_download_button


st.set_page_config(
    page_title="Explainable AI Copilot POC",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
<style>
    .main-header {
        padding: 1.1rem 1.2rem;
        border-radius: 18px;
        background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 55%, #0369a1 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .main-header h1 { color: white; margin-bottom: 0.2rem; }
    .scope-chip {
        display: inline-block;
        padding: 0.35rem 0.6rem;
        border-radius: 999px;
        background: #e0f2fe;
        color: #075985;
        font-size: 0.85rem;
        margin: 0.15rem;
        font-weight: 600;
    }
    .warning-box {
        padding: 0.9rem;
        border-radius: 12px;
        border-left: 5px solid #f59e0b;
        background: #fffbeb;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-header">
    <h1>🤖 Proof-of-Concept Explainable AI Copilot</h1>
    <p>Operational business decision-support prototype with flexible CSV analysis and focused churn case-study support.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<span class="scope-chip">CSV / Excel / JSON / API</span>
<span class="scope-chip">Customer churn POC</span>
<span class="scope-chip">SHAP when binary target exists</span>
<span class="scope-chip">Controlled Copilot</span>
<span class="scope-chip">Evaluation evidence</span>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Data Source")
    data_source_type = st.radio(
        "Choose input method",
        ["Upload file", "Public API / URL"],
        help="The main dissertation case study remains customer churn. Extra formats are a robustness feature."
    )

    uploaded_file = None
    api_url = ""

    if data_source_type == "Upload file":
        uploaded_file = st.file_uploader(
            "Upload CSV, Excel, or JSON dataset",
            type=SUPPORTED_FILE_TYPES,
        )
    else:
        api_url = st.text_input(
            "Paste public CSV/JSON API link",
            placeholder="https://example.com/data.csv",
            help="Use public no-auth CSV or JSON URLs only."
        )


if data_source_type == "Upload file" and uploaded_file is None:
    st.warning("Upload a CSV, Excel, or JSON dataset to start.")
    st.markdown(
        """
### Supported inputs

You can upload:

- **CSV** files
- **Excel** files: `.xlsx` or `.xls`
- **JSON** files containing records

Or choose **Public API / URL** in the sidebar for a public CSV/JSON link.

### Important dissertation scope note

These input options are **data ingestion robustness features**.  
The main evaluated dissertation case study remains **customer churn decision support**.

For **prediction + SHAP explanation**, select a **binary target column** such as `Churn` with Yes/No values.
"""
    )
    st.stop()

if data_source_type == "Public API / URL" and not api_url.strip():
    st.warning("Paste a public CSV or JSON API link to start.")
    st.markdown(
        """
Example supported URL types:

- public `.csv` link
- public `.json` link
- API response containing a list of records or a `data`, `records`, `items`, or `results` list

Authentication/API keys are not included in this proof-of-concept.
"""
    )
    st.stop()

try:
    if data_source_type == "Upload file":
        df = load_tabular_file(uploaded_file)
        data_name = uploaded_file.name
    else:
        df = load_from_api_url(api_url)
        data_name = api_url
except Exception as exc:
    st.error(f"Could not load this data source. Error: {exc}")
    st.stop()

# Target selection after file upload.
default_target = detect_target_column(df, PRIMARY_CASE_STUDY["target_column"])
target_options = ["No target - EDA/Copilot only"] + list(df.columns)
default_index = target_options.index(default_target) if default_target in target_options else 0

with st.sidebar:
    st.header("Target / prediction")
    selected_target_option = st.selectbox(
        "Choose target column for ML prediction",
        target_options,
        index=default_index,
        help="Choose Churn or another binary Yes/No / 0/1 target. Choose no target for general dataset analysis only.",
    )
    target_col = None if selected_target_option == "No target - EDA/Copilot only" else selected_target_option

    if target_col:
        if is_binary_target(df[target_col]):
            st.success(f"Binary target selected: {target_col}")
        else:
            st.warning(f"{target_col} is not binary. ML prediction is disabled, but EDA/Copilot still works.")
    else:
        st.info("No target selected. EDA and Copilot will still work.")

    st.divider()
    page = st.radio(
        "Navigation",
        [
            "1. Data Upload & Quality",
            "2. Dashboard Summary",
            "3. Train ML Models",
            "4. Prediction & SHAP Explanation",
            "5. Controlled Copilot Chat",
            "6. Evaluation & Report Evidence",
        ],
    )

# Reset model if file/target changes.
current_signature = (data_source_type, data_name, df.shape, target_col)
if "data_signature" not in st.session_state or st.session_state.data_signature != current_signature:
    st.session_state.data_signature = current_signature
    st.session_state.trained_result = None
    st.session_state.selected_prediction = None
    st.session_state.selected_explanation = None
    st.session_state.copilot_messages = None

if "trained_result" not in st.session_state:
    st.session_state.trained_result = None
if "selected_prediction" not in st.session_state:
    st.session_state.selected_prediction = None
if "selected_explanation" not in st.session_state:
    st.session_state.selected_explanation = None
if "copilot_draft_question" not in st.session_state:
    st.session_state.copilot_draft_question = ""


if page == "1. Data Upload & Quality":
    st.header("1. Data Upload & Quality")

    quality = data_quality_report(df, target_col)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Target", target_col or "None")
    col4.metric("Duplicates", quality["duplicate_rows"])

    st.subheader("Dataset Preview")
    st.dataframe(df.head(30), use_container_width=True)

    st.subheader("Data Quality Report")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Missing cells", quality["missing_cells"])
    q2.metric("Duplicate rows", quality["duplicate_rows"])
    q3.metric("Numeric columns", quality["numeric_columns"])
    q4.metric("Categorical columns", quality["categorical_columns"])

    tab1, tab2, tab3, tab4 = st.tabs(["Missing values", "Data types", "Target distribution", "ID-like columns"])
    with tab1:
        st.dataframe(quality["missing_by_column"], use_container_width=True)
    with tab2:
        st.dataframe(quality["dtypes"], use_container_width=True)
    with tab3:
        if target_col and not quality["target_distribution"].empty:
            st.dataframe(quality["target_distribution"], use_container_width=True)
        else:
            st.info("No target selected.")
    with tab4:
        st.write(quality["id_like_columns"] or "No strong ID-like columns detected.")

    st.success("Data quality check completed.")


elif page == "2. Dashboard Summary":
    st.header("2. Dashboard Summary")

    metrics, charts = create_dashboard_metrics(df, target_col)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total records", metrics["total_records"])
    col2.metric("Total columns", metrics["total_columns"])
    col3.metric("Numeric columns", metrics["numeric_columns"])
    col4.metric("Categorical columns", metrics["categorical_columns"])

    if metrics["positive_rate"] is not None:
        st.metric(f"{target_col} positive rate", f"{metrics['positive_rate']:.2f}%")

    st.subheader("Charts")
    for name, fig in charts.items():
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Automatic Summary")
    for item in metrics["summary_points"]:
        st.write(f"✅ {item}")


elif page == "3. Train ML Models":
    st.header("3. Train ML Models")

    if target_col is None:
        st.warning("Please select a binary target column in the sidebar before training a model.")
        st.stop()

    if not is_binary_target(df[target_col]):
        st.error(
            f"The selected target column **{target_col}** is not binary. "
            "This proof-of-concept supports binary classification only. Use EDA/Copilot pages for this dataset, or choose a Yes/No or 0/1 target column."
        )
        st.stop()

    st.markdown(
        f"""
The model will predict the selected binary target: **{target_col}**.

The proof-of-concept trains:

- **Logistic Regression** as a transparent baseline
- **Random Forest Classifier** as a stronger non-linear model
"""
    )

    if st.button("Train / retrain models", type="primary"):
        with st.spinner("Preparing data and training models..."):
            try:
                prepared = prepare_features(df, target_col, PRIMARY_CASE_STUDY["drop_columns"])
                trained_result = train_and_evaluate_models(prepared)
                st.session_state.trained_result = trained_result
                st.success("Model training completed.")
            except Exception as exc:
                st.error(f"Model training failed: {exc}")

    if st.session_state.trained_result is not None:
        trained_result = st.session_state.trained_result
        st.subheader("Model Performance Comparison")
        st.dataframe(trained_result["metrics_table"], use_container_width=True)

        st.info(
            f"Best model: **{trained_result['best_model_name']}**\\n\\n"
            "Reason: selected by highest F1-score, recall and ROC-AUC."
        )

        st.subheader("Confusion Matrix")
        st.dataframe(trained_result["confusion_matrix"], use_container_width=True)
    else:
        st.info("Click the button above to train the baseline and final model.")


elif page == "4. Prediction & SHAP Explanation":
    st.header("4. Prediction & SHAP Explanation")

    if st.session_state.trained_result is None:
        st.warning("Please train the ML models first on the Train ML Models page.")
        st.stop()

    trained_result = st.session_state.trained_result

    st.subheader("Select a record")
    selected_index = st.selectbox(
        "Choose row index",
        list(df.index[: min(len(df), 500)]),
    )

    selected_raw = df.loc[selected_index]
    st.dataframe(pd.DataFrame([selected_raw]), use_container_width=True)

    if st.button("Predict and Explain", type="primary"):
        try:
            prediction_result = predict_single_record(
                trained_result=trained_result,
                raw_df=df,
                row_index=selected_index,
            )

            shap_result = get_shap_explanation_safe(
                trained_result=trained_result,
                row_index=selected_index,
            )

            explanation_text = generate_plain_english_explanation(
                prediction_result=prediction_result,
                shap_result=shap_result,
            )

            recommendations = generate_recommendations(
                shap_result=shap_result,
                selected_raw=selected_raw,
            )

            st.session_state.selected_prediction = prediction_result
            st.session_state.selected_explanation = {
                "shap_result": shap_result,
                "explanation_text": explanation_text,
                "recommendations": recommendations,
            }
        except Exception as exc:
            st.error(f"Prediction/explanation failed: {exc}")

    if st.session_state.selected_prediction:
        prediction_result = st.session_state.selected_prediction
        explanation_pack = st.session_state.selected_explanation

        probability = prediction_result["positive_probability"] * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", prediction_result["risk_label"])
        c2.metric("Positive-class probability", f"{probability:.2f}%")
        c3.metric("Model", trained_result["best_model_name"])

        st.markdown(
            '<div class="warning-box">Decision-support warning: this output should support a human user, not automatically make decisions.</div>',
            unsafe_allow_html=True,
        )

        st.subheader("SHAP / Feature Explanation")
        shap_result = explanation_pack["shap_result"]

        if shap_result["available"]:
            st.dataframe(shap_result["top_features"], use_container_width=True)
        else:
            st.info(shap_result["message"])

        st.subheader("Plain-English Copilot Explanation")
        st.write(explanation_pack["explanation_text"])

        st.subheader("Recommended Actions")
        for rec in explanation_pack["recommendations"]:
            st.write(f"- {rec}")


elif page == "5. Controlled Copilot Chat":
    st.header("5. Controlled Copilot Chat")

    st.markdown(
        """
This Copilot answers from the **uploaded CSV**. It can work even when no prediction target is selected.  
For prediction explanations, choose a binary target and train the model first.
"""
    )

    examples = supported_question_examples()

    st.subheader("Quick question buttons")
    cols = st.columns(3)
    for i, example in enumerate(examples[:6]):
        with cols[i % 3]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.copilot_draft_question = example

    if not st.session_state.get("copilot_messages"):
        st.session_state.copilot_messages = [
            {
                "role": "assistant",
                "text": (
                    "Hello. Upload any CSV and ask about columns, missing values, duplicates, distributions, averages, highest values, or target rates. "
                    "I can also correct simple spelling mistakes."
                ),
                "chart": None,
                "table": None,
                "recommendations": None,
                "suggestions": None,
            }
        ]

    if st.button("Clear chat"):
        st.session_state.copilot_messages = [
            {
                "role": "assistant",
                "text": "Chat cleared. Ask a new data-grounded question.",
                "chart": None,
                "table": None,
                "recommendations": None,
                "suggestions": None,
            }
        ]
        st.rerun()

    for msg in st.session_state.copilot_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])
            if msg.get("chart") is not None:
                st.plotly_chart(msg["chart"], use_container_width=True)
            if msg.get("table") is not None:
                st.dataframe(msg["table"], use_container_width=True)
            if msg.get("recommendations"):
                st.write("### Recommendations")
                for rec in msg["recommendations"]:
                    st.write(f"- {rec}")
            if msg.get("suggestions"):
                st.write("### Try asking")
                for suggestion in msg["suggestions"][:5]:
                    st.code(suggestion)

    typed_question = st.chat_input("Ask a question about this uploaded dataset...")

    user_prompt = typed_question or st.session_state.copilot_draft_question
    if user_prompt:
        st.session_state.copilot_draft_question = ""

        st.session_state.copilot_messages.append(
            {"role": "user", "text": user_prompt, "chart": None, "table": None, "recommendations": None, "suggestions": None}
        )

        answer = answer_copilot_question(
            question=user_prompt,
            df=df,
            target_col=target_col,
            prediction_pack=st.session_state.selected_prediction,
            explanation_pack=st.session_state.selected_explanation,
        )

        st.session_state.copilot_messages.append(
            {
                "role": "assistant",
                "text": answer["text"],
                "chart": answer.get("chart"),
                "table": answer.get("table"),
                "recommendations": answer.get("recommendations"),
                "suggestions": answer.get("suggestions"),
            }
        )
        st.rerun()


elif page == "6. Evaluation & Report Evidence":
    st.header("6. Evaluation & Report Evidence")

    st.subheader("A/B Evaluation Design")
    st.markdown(
        """
| Condition | User sees |
|---|---|
| A: Prediction only | Risk score/category only |
| B: Copilot supported | Prediction + SHAP drivers + plain-English explanation + recommendation + safety warning |

For the dissertation, this evaluation should use the main customer churn dataset. Flexible uploads are included for robustness and demonstration.
"""
    )

    html_report = build_html_report(
        df=df,
        target_col=target_col,
        trained_result=st.session_state.trained_result,
        prediction_pack=st.session_state.selected_prediction,
        explanation_pack=st.session_state.selected_explanation,
    )

    st.subheader("Optional HTML Decision-Support Report")
    components.html(html_report, height=600, scrolling=True)
    create_download_button(html_report, filename="ai_copilot_decision_support_report.html")
