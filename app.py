from __future__ import annotations

import traceback
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.cleaning import clean_dataframe, detect_column_roles
from src.copilot import answer_question
from src.business_insights import detect_business_domain
from src.data_loader import (
    LoadedDataset,
    dataset_profile,
    find_common_columns,
    load_public_url,
    load_uploaded_files,
    merge_datasets,
)
from src.eda import (
    categorical_summary,
    figure_categorical_counts,
    figure_correlation,
    figure_date_trend,
    figure_missing,
    figure_numeric_distribution,
    figure_target_by_category,
    figure_target_by_numeric,
    missing_value_table,
    numeric_summary,
    overview_metrics,
    target_distribution,
    text_column_summary,
    simple_sentiment_summary,
)
from src.explainability import explain_model, top_driver_sentences
from src.modeling import detect_binary_targets, infer_positive_label, train_models, XGBOOST_AVAILABLE
from src.report import make_markdown_report

st.set_page_config(
    page_title="Explainable AI Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_plotly_chart(fig, key: str) -> None:
    """Render Plotly figures with a unique Streamlit key to avoid duplicate element errors."""
    st.plotly_chart(fig, use_container_width=True, key=key)

CUSTOM_CSS = """
<style>
:root { --card-bg: rgba(255,255,255,0.72); --border: rgba(49,51,63,0.12); }
.block-container { padding-top: 1.5rem; }
.hero {
  padding: 1.4rem 1.6rem; border-radius: 1.25rem;
  background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 52%, #0891b2 100%);
  color: white; box-shadow: 0 16px 40px rgba(15,23,42,0.20); margin-bottom: 1rem;
}
.hero h1 { margin: 0; font-size: 2rem; }
.hero p { margin: 0.35rem 0 0 0; opacity: 0.92; }
.card {
  padding: 1.05rem; border: 1px solid var(--border); border-radius: 1rem;
  background: var(--card-bg); box-shadow: 0 10px 28px rgba(15,23,42,0.06);
}
.metric-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: .04em; }
.metric-value { font-size: 1.55rem; font-weight: 750; color: #0f172a; }
.good { color: #047857; font-weight: 700; }
.warn { color: #b45309; font-weight: 700; }
.bad { color: #b91c1c; font-weight: 700; }
.small-muted { color: #64748b; font-size: 0.9rem; }
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
.stTabs [data-baseweb="tab"] { border-radius: 999px; padding-left: 1rem; padding-right: 1rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_state():
    defaults = {
        "datasets": {},
        "cleaned": {},
        "cleaning_reports": {},
        "active_dataset": None,
        "model_outputs": {},
        "explanations": {},
        "chat_history": [],
        "chat_contexts": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def add_dataset(ds: LoadedDataset):
    if ds.dataframe is None or ds.dataframe.empty:
        st.warning(f"Could not load `{ds.name}`. {ds.notes}")
        return
    name = ds.name
    base = name
    idx = 2
    while name in st.session_state.datasets:
        name = f"{base}_{idx}"
        idx += 1
    st.session_state.datasets[name] = ds.dataframe
    cleaned, report = clean_dataframe(ds.dataframe)
    st.session_state.cleaned[name] = cleaned
    st.session_state.cleaning_reports[name] = report
    if st.session_state.active_dataset is None:
        st.session_state.active_dataset = name


def active_df() -> pd.DataFrame | None:
    name = st.session_state.active_dataset
    if not name:
        return None
    return st.session_state.cleaned.get(name)


def render_metric_card(label: str, value: object):
    st.markdown(f"<div class='card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)


def sample_for_display(df: pd.DataFrame, max_rows: int = 30000) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=42)


init_state()

st.markdown(
    """
<div class='hero'>
  <h1>Explainable AI Copilot for Business Decision Support</h1>
  <p>Upload structured business data, clean it, explore it, train controlled models, explain predictions and ask data-grounded Copilot questions.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dataset Hub")
    uploads = st.file_uploader(
        "Upload CSV, TSV, Excel, JSON or ZIP",
        type=["csv", "tsv", "txt", "xlsx", "xls", "json", "zip"],
        accept_multiple_files=True,
        help="Multiple files are supported. ZIP files are scanned for CSV/Excel/JSON files.",
    )
    if st.button("Load uploaded files", use_container_width=True):
        try:
            loaded = load_uploaded_files(uploads)
            for ds in loaded:
                add_dataset(ds)
            st.success(f"Loaded {len(loaded)} dataset(s).")
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

    st.divider()
    url = st.text_input("Public CSV/JSON/Excel/API URL")
    if st.button("Load URL/API data", use_container_width=True):
        try:
            ds = load_public_url(url)
            add_dataset(ds)
            st.success(f"Loaded `{ds.name}` from URL/API.")
        except Exception as exc:
            st.error(f"URL/API load failed: {exc}")

    st.divider()
    if st.session_state.cleaned:
        selected = st.selectbox("Active dataset", list(st.session_state.cleaned.keys()), index=list(st.session_state.cleaned.keys()).index(st.session_state.active_dataset) if st.session_state.active_dataset in st.session_state.cleaned else 0)
        st.session_state.active_dataset = selected
        prof = dataset_profile(st.session_state.cleaned[selected])
        st.caption(f"Rows: {prof['rows']:,} | Columns: {prof['columns']:,} | Memory: {prof['memory_mb']} MB")
    else:
        st.info("Upload at least one dataset to begin.")

    with st.expander("Optional merge by shared key"):
        names = list(st.session_state.cleaned.keys())
        if len(names) >= 2:
            left_name = st.selectbox("Left/main dataset", names, key="merge_left")
            right_name = st.selectbox("Right/support dataset", [n for n in names if n != left_name], key="merge_right")
            left_df, right_df = st.session_state.cleaned[left_name], st.session_state.cleaned[right_name]
            common = find_common_columns(left_df, right_df)
            if common:
                left_key = st.selectbox("Left key", common if common else list(left_df.columns), key="left_key")
                right_key = st.selectbox("Right key", common if common else list(right_df.columns), key="right_key")
            else:
                left_key = st.selectbox("Left key", list(left_df.columns), key="left_key_all")
                right_key = st.selectbox("Right key", list(right_df.columns), key="right_key_all")
            how = st.selectbox("Merge type", ["left", "inner", "outer"], index=0)
            if st.button("Create merged dataset", use_container_width=True):
                try:
                    merged = merge_datasets(left_df, right_df, left_key, right_key, how=how)
                    name = f"merged_{left_name}_plus_{right_name}"
                    st.session_state.datasets[name] = merged
                    cleaned, report = clean_dataframe(merged)
                    st.session_state.cleaned[name] = cleaned
                    st.session_state.cleaning_reports[name] = report
                    st.session_state.active_dataset = name
                    st.success(f"Created `{name}` with {cleaned.shape[0]:,} rows and {cleaned.shape[1]:,} columns.")
                except Exception as exc:
                    st.error(f"Merge failed: {exc}")
        else:
            st.caption("Load at least two datasets to enable merging.")

    st.divider()
    st.caption("Scope guard: the main dissertation evaluation should remain customer churn. Extra datasets are robustness tests.")

if not st.session_state.cleaned:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='card'><b>1. Upload data</b><br><span class='small-muted'>CSV, Excel, JSON, ZIP or public URL/API.</span></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='card'><b>2. Analyse and train</b><br><span class='small-muted'>Automatic cleaning, EDA and model comparison if a binary target exists.</span></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='card'><b>3. Ask Copilot</b><br><span class='small-muted'>Answers are restricted to the active uploaded dataset.</span></div>""", unsafe_allow_html=True)
    st.stop()

name = st.session_state.active_dataset
df = active_df()
assert df is not None
report = st.session_state.cleaning_reports.get(name)
model_output = st.session_state.model_outputs.get(name)
explanation_output = st.session_state.explanations.get(name)

st.subheader(f"Active dataset: `{name}`")
metrics = overview_metrics(df)
cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    with col:
        render_metric_card(label, f"{value:,}" if isinstance(value, int) else value)

main_tabs = st.tabs(["📁 Data", "🧹 Cleaning", "📊 EDA", "🤖 Models", "🔎 Explain", "💡 Business", "💬 Copilot", "📄 Export"])

with main_tabs[0]:
    st.markdown("### Dataset preview")
    st.dataframe(df.head(200), use_container_width=True, height=350)
    roles = detect_column_roles(df)
    r1, r2, r3, r4 = st.columns(4)
    r1.info(f"Numeric: {len(roles['numeric'])}")
    r2.info(f"Categorical: {len(roles['categorical'])}")
    r3.info(f"Date/time: {len(roles['datetime'])}")
    r4.info(f"Long text: {len(roles['text'])}")
    with st.expander("Detected column roles"):
        st.json(roles)

with main_tabs[1]:
    st.markdown("### Automatic cleaning report")
    if report:
        st.dataframe(report.to_dataframe(), use_container_width=True)
        if report.renamed_columns:
            with st.expander("Column rename map"):
                st.dataframe(pd.DataFrame(list(report.renamed_columns.items()), columns=["Original", "Cleaned"]), use_container_width=True)
    st.markdown("### Missing-value profile")
    miss_table = missing_value_table(df)
    st.dataframe(miss_table, use_container_width=True)
    fig = figure_missing(df)
    if fig:
        render_plotly_chart(fig, key=f"missing_chart_{name}")

with main_tabs[2]:
    st.markdown("### Exploratory data analysis")
    eda_tabs = st.tabs(["Summary", "Numeric", "Categorical", "Dates/Text", "Target", "Correlation"])
    sample_df = sample_for_display(df)

    with eda_tabs[0]:
        st.dataframe(numeric_summary(df), use_container_width=True)
        st.dataframe(categorical_summary(df), use_container_width=True)

    with eda_tabs[1]:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            num_col = st.selectbox("Numeric column", numeric_cols, key="eda_num")
            render_plotly_chart(figure_numeric_distribution(sample_df, num_col), key=f"eda_numeric_{name}_{num_col}")
        else:
            st.info("No numeric columns found.")

    with eda_tabs[2]:
        cat_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
        if cat_cols:
            cat_col = st.selectbox("Categorical/text column", cat_cols, key="eda_cat")
            render_plotly_chart(figure_categorical_counts(sample_df, cat_col), key=f"eda_categorical_{name}_{cat_col}")
        else:
            st.info("No categorical columns found.")

    with eda_tabs[3]:
        date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if date_cols:
            date_col = st.selectbox("Date column", date_cols, key="eda_date")
            value_col = st.selectbox("Optional numeric value", [None] + numeric_cols, key="eda_date_value")
            fig = figure_date_trend(sample_df, date_col, value_col)
            if fig:
                render_plotly_chart(fig, key=f"eda_date_{name}_{date_col}_{value_col}")
        else:
            st.caption("No date columns detected.")
        text_cols = [c for c in df.columns if c not in numeric_cols and c not in date_cols]
        if text_cols:
            text_col = st.selectbox("Text/theme column", text_cols, key="eda_text")
            top_words = text_column_summary(df, text_col)
            if not top_words.empty:
                st.dataframe(top_words, use_container_width=True)
                render_plotly_chart(px.bar(top_words, x="word", y="count", title=f"Top words in {text_col}"), key=f"eda_text_{name}_{text_col}")
                st.json(simple_sentiment_summary(df, text_col))

    with eda_tabs[4]:
        target_candidates = detect_binary_targets(df)
        if target_candidates:
            target_col = st.selectbox("Binary target column", target_candidates + [c for c in df.columns if c not in target_candidates], key="eda_target")
            pos = infer_positive_label(df[target_col]) if df[target_col].nunique(dropna=True) == 2 else None
            st.dataframe(target_distribution(df, target_col), use_container_width=True)
            if pos is not None:
                st.caption(f"Detected positive class: `{pos}`")
            cat_cols = [c for c in df.select_dtypes(include=["object", "category", "string", "bool"]).columns if c != target_col]
            numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target_col]
            if cat_cols and pos is not None:
                cat = st.selectbox("Target by category", cat_cols, key="target_cat")
                fig = figure_target_by_category(sample_df, target_col, cat, pos)
                if fig:
                    render_plotly_chart(fig, key=f"target_category_{name}_{target_col}_{cat}")
            if numeric_cols:
                num = st.selectbox("Target by numeric", numeric_cols, key="target_num")
                fig = figure_target_by_numeric(sample_df, target_col, num, pos)
                if fig:
                    render_plotly_chart(fig, key=f"target_numeric_{name}_{target_col}_{num}")
        else:
            st.info("No binary target detected. EDA and Copilot still work, but model training needs a Yes/No or 0/1 style target.")

    with eda_tabs[5]:
        fig = figure_correlation(sample_df)
        if fig:
            render_plotly_chart(fig, key=f"correlation_{name}")
        else:
            st.info("Need at least two numeric columns for correlation heatmap.")

with main_tabs[3]:
    st.markdown("### Controlled model comparison")
    target_candidates = detect_binary_targets(df)
    if target_candidates:
        left, right = st.columns([2, 1])
        with left:
            target_col = st.selectbox("Select binary target", target_candidates + [c for c in df.columns if c not in target_candidates], key="model_target")
        with right:
            possible_values = list(df[target_col].dropna().unique()) if target_col in df.columns else []
            default_pos = infer_positive_label(df[target_col]) if target_col in df.columns and df[target_col].nunique(dropna=True) == 2 else (possible_values[0] if possible_values else None)
            positive_value = st.selectbox("Positive class", possible_values, index=possible_values.index(default_pos) if default_pos in possible_values else 0, key="model_pos") if possible_values else None
        inc_xgb = st.checkbox("Try optional XGBoost if installed", value=XGBOOST_AVAILABLE, help="The app continues normally if XGBoost is not installed or fails.")
        max_rows = st.slider("Maximum rows for model training", min_value=5000, max_value=200000, value=min(max(len(df), 5000), 120000), step=5000, help="Large datasets may be sampled for model training to keep Streamlit responsive.")
        if st.button("Train and compare all models", type="primary", use_container_width=True):
            try:
                with st.spinner("Training models and selecting the best model..."):
                    output = train_models(df, target_col, positive_value, include_xgboost=inc_xgb, max_training_rows=max_rows)
                    st.session_state.model_outputs[name] = output
                    st.session_state.explanations.pop(name, None)
                st.success(f"Training complete. Best model: {output.best_result.model_name}")
                st.rerun()
            except Exception as exc:
                st.error(f"Model training failed: {exc}")
                with st.expander("Error detail"):
                    st.code(traceback.format_exc())
    else:
        st.warning("No binary target detected. Model training is disabled for this active dataset.")

    if model_output and model_output.best_result:
        st.markdown("#### Model leaderboard")
        st.dataframe(model_output.leaderboard, use_container_width=True)
        best = model_output.best_result
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best model", best.model_name)
        c2.metric("F1", f"{best.metrics['f1']:.3f}")
        c3.metric("Recall", f"{best.metrics['recall']:.3f}")
        c4.metric("Precision", f"{best.metrics['precision']:.3f}")
        c5.metric("ROC-AUC", f"{best.metrics['roc_auc']:.3f}" if not np.isnan(best.metrics['roc_auc']) else "n/a")
        st.markdown("#### Confusion matrix")
        conf = pd.DataFrame(best.confusion, index=["Actual negative", "Actual positive"], columns=["Predicted negative", "Predicted positive"])
        st.dataframe(conf, use_container_width=True)
        if model_output.skipped_models:
            with st.expander("Skipped/failed models"):
                st.json(model_output.skipped_models)
        if best.metrics["f1"] < 0.50:
            st.warning("Model performance is weak. Use explanations carefully and report this limitation honestly.")

with main_tabs[4]:
    st.markdown("### Prediction explanation")
    if not (model_output and model_output.best_result):
        st.info("Train a model first to generate prediction explanations.")
    else:
        best = model_output.best_result
        st.caption(f"Using best model: `{best.model_name}` trained on `{name}`.")
        row_idx = st.number_input("Select row index to explain", min_value=0, max_value=max(len(df) - 1, 0), value=0, step=1)
        row_full = df.iloc[[int(row_idx)]].copy()
        row_features = row_full[[c for c in best.feature_columns if c in row_full.columns]].copy()
        if st.button("Generate prediction and explanation", type="primary", use_container_width=True):
            try:
                pred = best.pipeline.predict(row_features)[0]
                proba = best.pipeline.predict_proba(row_features)[0, 1] if hasattr(best.pipeline, "predict_proba") else None
                with st.spinner("Computing SHAP/fallback explanation..."):
                    explanation = explain_model(best, row_features)
                    st.session_state.explanations[name] = explanation
                st.success("Explanation generated.")
                if proba is not None:
                    risk = "High" if proba >= 0.67 else "Medium" if proba >= 0.34 else "Low"
                    st.metric("Predicted risk/probability", f"{proba:.2%}", risk)
                st.write(f"Predicted class: `{pred}`")
                st.rerun()
            except Exception as exc:
                st.error(f"Explanation failed: {exc}")
                st.code(traceback.format_exc())

        explanation_output = st.session_state.explanations.get(name)
        if explanation_output:
            if explanation_output.warning:
                st.warning(explanation_output.warning)
            st.markdown(f"Explanation method: **{explanation_output.method}**")
            st.markdown("#### Global drivers")
            st.dataframe(explanation_output.global_importance.head(20), use_container_width=True)
            if not explanation_output.global_importance.empty and "importance" in explanation_output.global_importance.columns:
                render_plotly_chart(px.bar(explanation_output.global_importance.head(15), x="importance", y="feature", orientation="h", title="Top global drivers"), key=f"explain_global_{name}")
            st.markdown("#### Local drivers for selected record")
            st.dataframe(explanation_output.local_importance.head(15), use_container_width=True)
            for sent in top_driver_sentences(explanation_output.local_importance, 5):
                st.write("• " + sent)
            st.info("Safety warning: outputs support, not replace, human judgement.")

with main_tabs[5]:
    st.markdown("### Business Insight Engine")
    st.caption("Domain-aware business suggestions generated only from the active uploaded dataset. Use this page for sales, employee/HR, customer feedback, marketing, and operations improvement ideas.")
    domain = detect_business_domain(df)
    st.info(f"Detected business domain: `{domain}`. This can be overridden by asking a more specific question in Copilot, for example sales, employee, feedback, marketing or operations.")
    business_questions = ["give business suggestions", "what should we improve", "what actions should we take"]
    if domain == "sales_retail":
        business_questions = ["which products should we purchase more", "which products are declining", "sales improvement ideas"]
    elif domain == "employee_hr":
        business_questions = ["employee support and retention suggestions", "who may need promotion review", "which groups have attrition risk"]
    elif domain in {"customer_feedback", "customer_churn"}:
        business_questions = ["how to improve customerfeedback", "customer retention suggestions", "who gave worst customerfeedback"]
    elif domain == "bank_marketing":
        business_questions = ["marketing campaign suggestions", "which customer segments should we target", "how to improve campaign response"]
    elif domain == "operations_service":
        business_questions = ["operations improvement suggestions", "which service area needs attention", "which complaints are increasing"]
    bcols = st.columns(min(3, len(business_questions)))
    for i, qbtn in enumerate(business_questions):
        if bcols[i % len(bcols)].button(qbtn, key=f"business_btn_{name}_{i}_{qbtn}", use_container_width=True):
            from src.copilot import answer_question as _answer_business_question
            binary_targets = detect_binary_targets(df)
            target_col = model_output.target_column if model_output else (binary_targets[0] if binary_targets else None)
            positive = model_output.positive_label if model_output else (infer_positive_label(df[target_col]) if target_col else None)
            response = _answer_business_question(qbtn, df, dataset_name=name, target_column=target_col, positive_label=positive, model_output=model_output, explanation_output=st.session_state.explanations.get(name), previous_context=st.session_state.chat_contexts.get(name))
            st.session_state.chat_history.append({"dataset": name, "question": qbtn, "response": response})
            if response.context:
                st.session_state.chat_contexts[name] = response.context
            st.rerun()
    st.markdown("Business insight answers will appear in the Copilot tab so they remain part of the evidence chat history.")
    st.warning("Safety guard: the system must not make final business, financial or HR decisions automatically. It provides data-grounded suggestions for human review.")

with main_tabs[6]:
    st.markdown("### Controlled data-grounded Copilot")
    st.caption("Answers are restricted to the active uploaded dataset, trained model and explanation artefacts. Short follow-up questions can reuse the previous topic/column from this dataset.")

    tool_cols = st.columns([1, 1, 2, 2])
    latest_first = tool_cols[0].toggle("Latest first", value=True, help="Shows the newest answer at the top so you do not need to scroll down after every question.")
    show_context = tool_cols[1].toggle("Show context", value=False, help="Shows the last column/topic remembered for follow-up questions.")
    if tool_cols[2].button("Clear chat for this dataset", use_container_width=True):
        st.session_state.chat_history = [item for item in st.session_state.chat_history if item.get("dataset") != name] if st.session_state.chat_history and isinstance(st.session_state.chat_history[0], dict) else []
        st.session_state.chat_contexts.pop(name, None)
        st.rerun()

    # Dataset-aware suggested questions reduce typing mistakes and keep the
    # controlled Copilot inside supported, data-grounded operations.
    numeric_cols_for_chat = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_for_chat = [c for c in df.columns if c not in numeric_cols_for_chat]
    feedback_cols_for_chat = [c for c in categorical_cols_for_chat if any(k in str(c).lower().replace("_", "") for k in ["feedback", "comment", "review"])]
    binary_targets_for_chat = detect_binary_targets(df)
    chat_target = model_output.target_column if model_output else (binary_targets_for_chat[0] if binary_targets_for_chat else None)
    examples = ["show dataset overview", "show missing values", "show model performance"]
    detected_domain_for_chat = detect_business_domain(df)
    if detected_domain_for_chat == "sales_retail":
        examples += ["which products should we purchase more", "which products are declining"]
    elif detected_domain_for_chat == "employee_hr":
        examples += ["employee support and retention suggestions", "who may need promotion review"]
    elif detected_domain_for_chat == "bank_marketing":
        examples += ["marketing campaign suggestions", "which customer segments should we target"]
    elif detected_domain_for_chat == "operations_service":
        examples += ["operations improvement suggestions", "which service area needs attention"]
    else:
        examples += ["give business suggestions"]
    if numeric_cols_for_chat:
        examples.append(f"who has highest {numeric_cols_for_chat[0]}")
    if categorical_cols_for_chat:
        examples.append(f"show {categorical_cols_for_chat[0]} distribution")
    if chat_target and categorical_cols_for_chat:
        group = next((c for c in categorical_cols_for_chat if c != chat_target), categorical_cols_for_chat[0])
        examples.append(f"show {chat_target} by {group}")
    if feedback_cols_for_chat:
        examples.append(f"who gave worst {feedback_cols_for_chat[0]}")
        examples.append(f"how to improve {feedback_cols_for_chat[0]}")
        examples.append(f"show {feedback_cols_for_chat[0]} themes")
    if "job" in df.columns and "marital" in df.columns:
        examples.append("unemployed marital status")
    examples.append("show top drivers")
    st.caption("Suggested questions for this active dataset:")
    example_cols = st.columns(min(4, len(examples)))
    for idx, ex in enumerate(examples[:8]):
        col = example_cols[idx % len(example_cols)]
        if col.button(ex, use_container_width=True, key=f"suggested_{name}_{idx}_{ex}"):
            st.session_state.pending_question = ex

    if show_context:
        st.info(f"Current dataset: `{name}`. Last remembered context: `{st.session_state.chat_contexts.get(name, {})}`")

    question = st.chat_input("Ask about the active dataset. Examples: who gave worst feedback, show churn by contract, highest balance, same again, missing values...")
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
    if question:
        binary_targets = detect_binary_targets(df)
        target_col = model_output.target_column if model_output else (binary_targets[0] if binary_targets else None)
        positive = model_output.positive_label if model_output else (infer_positive_label(df[target_col]) if target_col else None)
        response = answer_question(
            question,
            df,
            dataset_name=name,
            target_column=target_col,
            positive_label=positive,
            model_output=model_output,
            explanation_output=st.session_state.explanations.get(name),
            previous_context=st.session_state.chat_contexts.get(name),
        )
        if response.context:
            st.session_state.chat_contexts[name] = response.context
        st.session_state.chat_history.append({"dataset": name, "question": question, "response": response})

    # Keep chat scoped to the active dataset so answers from another uploaded file do not mix in.
    if st.session_state.chat_history and isinstance(st.session_state.chat_history[0], dict):
        active_history = [item for item in st.session_state.chat_history if item.get("dataset") == name]
    else:
        active_history = [{"dataset": name, "question": q, "response": r} for q, r in st.session_state.chat_history]

    items = active_history[-8:]
    if latest_first:
        items = list(reversed(items))

    for display_idx, item in enumerate(items):
        user_q = item["question"]
        response = item["response"]
        hist_idx = len(active_history) - display_idx if latest_first else display_idx
        with st.chat_message("user"):
            st.write(user_q)
        with st.chat_message("assistant"):
            if response.corrections:
                st.caption("Interpreted corrections: " + ", ".join(response.corrections))
            if response.interpreted_question:
                st.caption(f"Interpreted question: `{response.interpreted_question}`")
            st.markdown(response.answer)
            if response.table is not None and not response.table.empty:
                st.dataframe(response.table, use_container_width=True)
            if response.chart is not None:
                render_plotly_chart(response.chart, key=f"chat_chart_{name}_{hist_idx}_{display_idx}")
            st.warning(response.safety_warning)

with main_tabs[7]:
    st.markdown("### Export evidence")
    markdown = make_markdown_report(name, df, report, model_output, st.session_state.explanations.get(name))
    st.download_button("Download Markdown report", markdown, file_name=f"{name}_copilot_report.md", mime="text/markdown", use_container_width=True)
    st.download_button("Download cleaned active dataset CSV", df.to_csv(index=False), file_name=f"{name}_cleaned.csv", mime="text/csv", use_container_width=True)
    if report:
        st.download_button("Download cleaning report CSV", report.to_dataframe().to_csv(index=False), file_name=f"{name}_cleaning_report.csv", mime="text/csv", use_container_width=True)

