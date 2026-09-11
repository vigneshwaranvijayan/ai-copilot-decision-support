from __future__ import annotations

import traceback
import time
from pathlib import Path
from typing import Dict, List

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
    load_google_drive_url,
    load_postgres_table,
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
from src.modeling import (
    detect_binary_targets,
    detect_regression_targets,
    infer_positive_label,
    train_models,
    train_regression_models,
    XGBOOST_AVAILABLE,
    infer_clear_binary_target,
)
from src.report import make_markdown_report, build_results_export_package
from src.visual_analytics import build_visual_dashboard
from src.validation import build_readiness_report, status_badge_html
from src.persistence import record_audit_event, read_audit_events
from src.research_contribution import contribution_summary, contribution_table
from src.readiness_gates import readiness_criteria_table
from src.evaluation_framework import evaluation_metrics_table
from src.evidence_package import build_evidence_package, evidence_package_to_text
from src.memory_manager import update_short_term_memory, short_term_memory_table, memory_architecture_table
from src.vector_memory import add_memory_document, query_memory, memory_backend_status
from src.document_retrieval import (
    extract_text_from_bytes,
    chunk_text,
    chunks_to_dataframe,
    hybrid_search,
    evaluate_retrieval,
    retrieval_pipeline_description,
)
from src.performance_monitor import (
    performance_start,
    process_rss_mb,
    process_peak_rss_mb,
    utc_timestamp,
    build_runtime_summary,
    build_memory_allocation_evidence,
    build_copilot_evaluation_summary,
)
from src.question_batch import split_reviewer_questions

st.set_page_config(
    page_title="Dataset-Grounded AI Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_plotly_chart(fig, key: str) -> None:
    """Render Plotly figures with a consistent modern presentation and unique Streamlit key."""
    if fig is not None:
        fig.update_layout(
            template="plotly_white",
            hovermode="closest",
            margin=dict(l=35, r=25, t=60, b=35),
            title=dict(x=0.02, xanchor="left"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        # Some Plotly traces, such as Indicator/Gauge, do not support
        # hovertemplate. Updating all traces at once can crash the app on
        # those chart types, so only update traces that expose the property.
        for trace in fig.data:
            if hasattr(trace, "hovertemplate"):
                trace.hovertemplate = None
    st.plotly_chart(fig, use_container_width=True, key=key)

CUSTOM_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/* Keep Streamlit header visible so the sidebar collapse/expand control works. */
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
.answer-card { padding: 1.0rem 1.1rem; border-radius: 1rem; border: 1px solid rgba(15,23,42,0.10); background: linear-gradient(180deg,#ffffff 0%,#f8fafc 100%); box-shadow: 0 10px 24px rgba(15,23,42,.06); }
.pipeline-step { padding: .7rem; border: 1px solid rgba(100,116,139,.25); border-radius: .75rem; background: white; text-align:center; min-height: 92px; }
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
.stTabs [data-baseweb="tab"] { border-radius: 999px; padding-left: 1rem; padding-right: 1rem; }

/* Compact app chrome and dashboard navigation */
.block-container { padding-top: 0.8rem; max-width: 1500px; }
.compact-topbar {
  padding: .85rem 1.05rem; border-radius: 1rem;
  background: linear-gradient(135deg,#0f172a,#1d4ed8,#0891b2);
  color: white; box-shadow: 0 12px 30px rgba(15,23,42,.16); margin-bottom: .85rem;
}
.compact-topbar h2 { margin: 0; font-size: 1.25rem; line-height: 1.25; }
.compact-topbar p { margin: .25rem 0 0 0; opacity: .9; font-size: .92rem; }
.page-topline { padding: .65rem .9rem; border:1px solid var(--border); border-radius:.85rem; background:#fff; margin-bottom:.75rem; }
.nav-card {
  border: 1px solid rgba(15,23,42,.10); border-radius: 1rem; padding: .85rem;
  background: linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);
  box-shadow: 0 10px 24px rgba(15,23,42,.06); min-height: 92px;
}
.nav-card .nav-title { font-weight: 800; color:#0f172a; font-size:.98rem; }
.nav-card .nav-text { color:#64748b; font-size:.82rem; margin-top:.2rem; }
button[kind="secondary"], div.stButton > button {
  border-radius: .85rem !important; min-height: 2.45rem; font-weight: 650;
}
.dashboard-section-title { font-size:1.05rem; font-weight:800; margin: 1rem 0 .45rem 0; color:#0f172a; }
.chat-fixed-note { color:#64748b; font-size:.86rem; margin-bottom:.35rem; }
div[data-testid="stChatMessage"] {
  border: 1px solid rgba(15,23,42,.08); border-radius: 1rem; padding: .3rem .6rem;
  background: #ffffff; box-shadow: 0 8px 20px rgba(15,23,42,.04); margin-bottom: .45rem;
}


.ai-topbar {
  display:flex; align-items:center; justify-content:space-between;
  padding:.85rem 1.1rem; border-radius:1rem;
  background: linear-gradient(135deg,#0f172a 0%,#1d4ed8 55%,#0891b2 100%);
  color:white; box-shadow:0 12px 30px rgba(15,23,42,.16); margin-bottom:.55rem;
}
.ai-brand { font-size:1.18rem; font-weight:900; letter-spacing:.02em; }
.ai-subtitle { font-size:.82rem; opacity:.9; margin-top:.12rem; }
.ai-user { font-size:.86rem; opacity:.95; background:rgba(255,255,255,.14); padding:.35rem .65rem; border-radius:999px; }
.side-brand-title { font-size:1.18rem; font-weight:900; color:#0f172a; margin:.15rem 0 .2rem 0; }
.side-nav {
  padding:.55rem .65rem; border:1px solid rgba(15,23,42,.10);
  border-radius:.8rem; margin:.28rem 0 .18rem 0;
  background:linear-gradient(180deg,#ffffff,#f8fafc);
  box-shadow:0 4px 12px rgba(15,23,42,.04);
}
.side-nav.active { border-color:#2563eb; background:linear-gradient(180deg,#eff6ff,#ffffff); }
.side-nav span { font-size:.74rem; color:#64748b; }
.upload-hero {
  padding:1.15rem 1.25rem; border-radius:1.1rem;
  background:linear-gradient(135deg,#eff6ff,#ffffff);
  border:1px solid rgba(37,99,235,.18); margin-bottom:.8rem;
}
.upload-hero h2 { margin:0; color:#0f172a; font-size:1.45rem; }
.upload-hero p { margin:.25rem 0 0 0; color:#475569; }
.model-use-card { padding:.9rem; border-radius:.9rem; background:#f8fafc; border:1px solid rgba(15,23,42,.10); }


.reviewer-mode-box {
  padding:.95rem; border-radius:1rem; border:1px solid rgba(37,99,235,.18);
  background:linear-gradient(135deg,#eff6ff,#ffffff); margin:.7rem 0;
}
.reviewer-mode-box h4 { margin:.1rem 0 .25rem 0; color:#0f172a; }
.review-tip { font-size:.84rem; color:#475569; }
.compact-answer {
  padding:.75rem .9rem; border-radius:.9rem; background:#ffffff;
  border:1px solid rgba(15,23,42,.08); box-shadow:0 4px 14px rgba(15,23,42,.04);
}

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)




def render_ai_topbar(active_name: str | None = None, df: pd.DataFrame | None = None) -> None:
    """Compact top navbar: requested AI Solution branding, dummy account, theme and export."""
    if "ui_dark_mode" not in st.session_state:
        st.session_state.ui_dark_mode = False

    if st.session_state.ui_dark_mode:
        st.markdown(
            """
<style>
.stApp { background:#0b1220; color:#e5e7eb; }
.card, .nav-card, .page-topline, div[data-testid="stChatMessage"], .model-use-card { background:#111827 !important; color:#e5e7eb !important; border-color:rgba(148,163,184,.25) !important; }
.metric-value, .dashboard-section-title, .nav-card .nav-title, .side-brand-title { color:#f8fafc !important; }
.small-muted, .nav-card .nav-text, .side-nav span { color:#cbd5e1 !important; }
.side-nav { background:#111827 !important; }
.side-nav.active { background:#1e293b !important; }
</style>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div class="ai-topbar">
  <div>
    <div class="ai-brand">AI Solution</div>
    <div class="ai-subtitle">Dataset-grounded explainable Copilot for operational decision support</div>
  </div>
  <div class="ai-user">Account: Research User</div>
</div>
""",
        unsafe_allow_html=True,
    )
    top_left, top_theme, top_export = st.columns([7, 1.3, 1.5])
    with top_left:
        if active_name and df is not None:
            st.caption(f"Active dataset: `{active_name}` • {len(df):,} rows • {df.shape[1]:,} columns")
        else:
            st.caption("Upload a dataset or connect a data source to begin.")
    with top_theme:
        st.session_state.ui_dark_mode = st.toggle("Dark", value=st.session_state.get("ui_dark_mode", False), key="global_dark_toggle")
    with top_export:
        if active_name and df is not None:
            if st.button(
                "Export",
                key="topbar_go_export",
                use_container_width=True,
            ):
                st.session_state.current_page = "Export Data"
                st.rerun()
        else:
            st.button("Export", disabled=True, use_container_width=True)


def render_upload_page() -> None:
    """Main upload/connect page so upload is not hidden only inside the sidebar."""
    st.markdown(
        """
<div class="upload-hero">
  <h2>Upload or connect data</h2>
  <p>Choose local files, public URL/API, Google Drive, or PostgreSQL. After loading, the app cleans the data, runs readiness checks, stores evidence and auto-trains once when a valid target is detected.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    up_tab, url_tab, drive_tab, pg_tab, doc_tab = st.tabs(["Local files", "URL / API", "Google Drive", "PostgreSQL", "Document memory"])
    with up_tab:
        st.markdown("#### Local upload")
        uploads_main = st.file_uploader(
            "Upload CSV, TSV, Excel, JSON or ZIP",
            type=["csv", "tsv", "txt", "xlsx", "xls", "json", "zip"],
            accept_multiple_files=True,
            key="upload_page_files",
        )
        if st.button("Load local files", key="upload_page_load", use_container_width=True):
            try:
                with st.status("Loading uploaded data...", expanded=True) as status:
                    loaded = load_uploaded_files(uploads_main)
                    for ds in loaded:
                        st.write(f"Processing `{ds.name}`")
                        add_dataset(ds)
                    status.update(label="Data loaded", state="complete")
                st.success(f"Loaded {len(loaded)} dataset(s).")
                st.session_state.current_page = "Dashboard"
                st.rerun()
            except Exception as exc:
                st.error(f"Upload failed: {exc}")
    with url_tab:
        st.markdown("#### Public URL / API")
        url_main = st.text_input("Public CSV / JSON / Excel / API URL", key="upload_page_url")
        if st.button("Load URL/API", key="upload_page_load_url", use_container_width=True):
            try:
                with st.status("Loading URL/API data...", expanded=True) as status:
                    ds = load_public_url(url_main)
                    add_dataset(ds)
                    status.update(label="URL/API data loaded", state="complete")
                st.success(f"Loaded `{ds.name}`.")
                st.session_state.current_page = "Dashboard"
                st.rerun()
            except Exception as exc:
                st.error(f"URL/API load failed: {exc}")
    with drive_tab:
        st.markdown("#### Google Drive / Google Sheets")
        st.caption("Use a public/shared Drive file or Google Sheets link. Private Drive folder OAuth is documented as a cloud extension.")
        gd_main = st.text_input("Shared Google Drive / Google Sheets link", key="upload_page_gdrive")
        if st.button("Load Google Drive", key="upload_page_load_gdrive", use_container_width=True):
            try:
                with st.status("Loading Google Drive data...", expanded=True) as status:
                    ds = load_google_drive_url(gd_main)
                    add_dataset(ds)
                    status.update(label="Google Drive data loaded", state="complete")
                st.success(f"Loaded `{ds.name}`.")
                st.session_state.current_page = "Dashboard"
                st.rerun()
            except Exception as exc:
                st.error(f"Google Drive load failed: {exc}")
    with pg_tab:
        st.markdown("#### PostgreSQL structured data")
        st.caption("PostgreSQL is for structured/live business tables. Keep credentials in environment variables or secrets; do not commit them.")
        pg_uri_main = st.text_input("PostgreSQL URI", type="password", key="upload_page_pg_uri", placeholder="postgresql+psycopg2://user:password@host:5432/db")
        pg_query_main = st.text_input("Table name or SELECT query", key="upload_page_pg_query", placeholder="public.customers or SELECT * FROM public.customers")
        pg_limit_main = st.number_input("Maximum rows", min_value=1000, max_value=500000, value=100000, step=1000, key="upload_page_pg_limit")
        if st.button("Load PostgreSQL", key="upload_page_load_pg", use_container_width=True):
            try:
                with st.status("Connecting to PostgreSQL...", expanded=True) as status:
                    ds = load_postgres_table(pg_uri_main, pg_query_main, limit=int(pg_limit_main))
                    add_dataset(ds)
                    status.update(label="PostgreSQL data loaded", state="complete")
                st.success(f"Loaded `{ds.name}`.")
                st.session_state.current_page = "Dashboard"
                st.rerun()
            except Exception as exc:
                st.error(f"PostgreSQL load failed: {exc}")
    with doc_tab:
        st.markdown("#### Future/optional document retrieval demonstration")
        st.caption("Future architecture demonstration only. It is separate from the main assessed churn workflow and does not write long-term semantic memory unless the experimental semantic-memory option is enabled.")
        docs = st.file_uploader(
            "Upload documents for retrieval memory",
            type=["txt", "md", "pdf", "docx", "csv", "json"],
            accept_multiple_files=True,
            key="document_memory_files",
        )
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            doc_company = st.text_input("Company / organisation metadata", value="demo", key="doc_company_meta")
        with dcol2:
            doc_department = st.text_input("Department metadata", value="general", key="doc_department_meta")
        with dcol3:
            doc_topic = st.text_input("Topic metadata", value="decision-support", key="doc_topic_meta")
        c1, c2 = st.columns(2)
        with c1:
            chunk_size = st.number_input("Chunk size (words)", min_value=200, max_value=1200, value=700, step=50, key="doc_chunk_size")
        with c2:
            overlap_size = st.number_input("Overlap (words)", min_value=20, max_value=300, value=120, step=10, key="doc_overlap_size")
        if st.button("Add documents to retrieval memory", key="add_docs_memory", use_container_width=True):
            added_chunks = []
            for doc in docs or []:
                try:
                    raw = doc.getvalue()
                    text = extract_text_from_bytes(doc.name, raw)
                    chunks = chunk_text(
                        text,
                        document_name=doc.name,
                        chunk_words=int(chunk_size),
                        overlap_words=int(overlap_size),
                        base_metadata={
                            "company": doc_company,
                            "department": doc_department,
                            "topic": doc_topic,
                            "document_type": Path(doc.name).suffix.lower().replace(".", "") or "text",
                        },
                    )
                    st.session_state.document_chunks.extend(chunks)
                    if st.session_state.get("semantic_memory_enabled", False):
                        for ch in chunks:
                            try:
                                mem_id = add_memory_document(ch.text, metadata={**ch.metadata, "type": "document_chunk"})
                                st.session_state.semantic_memory_events.append({"dataset": st.session_state.active_dataset, "type": "document_chunk", "memory_id": mem_id, "document": doc.name, "chunk_no": ch.metadata.get("chunk_no")})
                            except Exception:
                                pass
                    added_chunks.extend(chunks)
                except Exception as exc:
                    st.error(f"Could not process {getattr(doc, 'name', 'document')}: {exc}")
            if added_chunks:
                st.success(f"Added {len(added_chunks)} overlapping document chunks to session memory.")
                st.dataframe(chunks_to_dataframe(added_chunks).drop(columns=["text"], errors="ignore"), use_container_width=True, hide_index=True)
        if st.session_state.get("document_chunks"):
            st.markdown("##### Current document chunks")
            st.dataframe(chunks_to_dataframe(st.session_state.document_chunks).drop(columns=["text"], errors="ignore"), use_container_width=True, hide_index=True)

    if st.session_state.cleaned:
        st.markdown("### Loaded datasets")
        rows = []
        for ds_name, ds_df in st.session_state.cleaned.items():
            meta = st.session_state.source_registry.get(ds_name, {})
            rows.append({
                "dataset": ds_name,
                "source": meta.get("source_type", "unknown"),
                "rows": len(ds_df),
                "columns": ds_df.shape[1],
                "readiness": st.session_state.readiness_reports.get(ds_name, {}).get("overall_status", "WARNING"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def init_state():
    defaults = {
        "datasets": {},
        "cleaned": {},
        "cleaning_reports": {},
        "active_dataset": None,
        "model_outputs": {},
        "regression_outputs": {},
        "explanations": {},
        "chat_history": [],
        "chat_contexts": {},
        "business_last_response": None,
        "evaluation_responses": [],
        "scale_evidence": [],
        "source_registry": {},
        "readiness_reports": {},
        "auto_model_done": {},
        "audit_events_session": [],
        "session_memory": {},
        "semantic_memory_events": [],
        "semantic_memory_enabled": False,
        "document_chunks": [],
        "auto_model_enabled": True,
        "readiness_target_columns": {},
        "current_page": "Upload Data",
        "performance_events": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def record_performance_event(step: str, start_time, *, status: str = "PASS", details: dict | None = None) -> None:
    """Store runtime/process evidence for Chapter 5 and export."""
    if isinstance(start_time, dict):
        start_perf = float(start_time.get("start_perf", time.perf_counter()))
        start_rss = start_time.get("start_rss_mb")
    else:
        start_perf = float(start_time)
        start_rss = None
    elapsed = round(time.perf_counter() - start_perf, 3)
    end_rss = process_rss_mb()
    try:
        delta_rss = round(float(end_rss) - float(start_rss), 3) if start_rss is not None else None
    except Exception:
        delta_rss = None
    st.session_state.performance_events.append({
        "dataset": st.session_state.get("active_dataset"),
        "step": step,
        "status": status,
        "elapsed_seconds": elapsed,
        "recorded_at_utc": utc_timestamp(),
        "process_rss_start_mb": start_rss,
        "process_rss_end_mb": end_rss,
        "process_rss_delta_mb": delta_rss,
        "process_peak_rss_mb": process_peak_rss_mb(),
        "details": details or {},
    })


def add_dataset(ds: LoadedDataset):
    perf_start = performance_start()
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
    st.session_state.source_registry[name] = {
        "source_type": getattr(ds, "source_type", "unknown"),
        "notes": getattr(ds, "notes", ""),
        "original_name": getattr(ds, "name", name),
    }
    detected_readiness_target = infer_readiness_target(cleaned)
    if detected_readiness_target:
        st.session_state.readiness_target_columns[name] = detected_readiness_target
    st.session_state.readiness_reports[name] = build_readiness_report(
        cleaned, dataset_name=name, target_column=detected_readiness_target
    ).to_dict()
    record_performance_event("load_clean_validate_dataset", perf_start, details={"rows": int(cleaned.shape[0]), "columns": int(cleaned.shape[1]), "source": getattr(ds, "source_type", "unknown")})
    try:
        record_audit_event("dataset_loaded", dataset_name=name, status="PASS", details=st.session_state.source_registry[name])
    except Exception:
        pass
    # Long-term semantic memory: store dataset summary/readiness evidence.
    try:
        if st.session_state.get("semantic_memory_enabled", False):
            package = build_evidence_package(
                dataset_name=name,
                df=cleaned,
                question="dataset loaded",
                readiness=st.session_state.readiness_reports[name],
            )
            mem_id = add_memory_document(
                evidence_package_to_text(package),
                metadata={"dataset": name, "type": "dataset_summary", "source_type": getattr(ds, "source_type", "unknown")},
            )
            st.session_state.semantic_memory_events.append({"dataset": name, "type": "dataset_summary", "memory_id": mem_id})
    except Exception:
        pass
    if st.session_state.active_dataset is None:
        st.session_state.active_dataset = name

    # Dissertation demo default: prepare the four-model comparison + explanation
    # once at upload time so later Copilot answers are immediate. The auto-model
    # routine is guarded and only runs for a clearly detected target such as churn.
    if st.session_state.get("auto_model_enabled", True):
        maybe_auto_train_classification(name, cleaned)


def active_df() -> pd.DataFrame | None:
    name = st.session_state.active_dataset
    if not name:
        return None
    return st.session_state.cleaned.get(name)


def render_metric_card(label: str, value: object):
    st.markdown(f"<div class='card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)




def compact_copilot_answer(full_answer: str) -> str:
    """Return a screenshot-friendly answer while preserving core evidence fields."""
    if not full_answer:
        return ""
    important_prefixes = [
        "**Question understood as:**",
        "**Dataset used:**",
        "**Data readiness:**",
        "**Answer grounding:**",
        "**Confidence:**",
        "**Columns/artefacts used:**",
        "**Model / prediction evidence:**",
        "**Evidence from the integrated data:**",
        "**Business / decision-support use:**",
        "**Limitations:**",
        "**Evidence format:**",
    ]
    lines = [ln.strip() for ln in full_answer.splitlines() if ln.strip()]
    picked = []
    for ln in lines:
        if ln.startswith("### Copilot answer"):
            picked.append(ln)
        elif any(ln.startswith(prefix) for prefix in important_prefixes):
            # Keep long evidence/model lines readable in screenshots.
            if len(ln) > 650:
                ln = ln[:650].rstrip() + "..."
            picked.append(ln)
    if not picked:
        return full_answer[:1200] + ("..." if len(full_answer) > 1200 else "")
    return "\n\n".join(picked)

def sample_for_display(df: pd.DataFrame, max_rows: int = 30000) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=42)


def infer_readiness_target(df: pd.DataFrame) -> str | None:
    """Auto-detect a clear target for readiness without training a model."""
    return infer_clear_binary_target(df)


def readiness_dict_for(name: str, df: pd.DataFrame, target_column: str | None = None) -> dict:
    previous_target = st.session_state.get("readiness_target_columns", {}).get(name)
    effective_target = target_column or previous_target or infer_readiness_target(df)
    report_dict = st.session_state.readiness_reports.get(name)

    # Rebuild only when no report exists or the effective target changes.
    if not report_dict or effective_target != previous_target:
        report_dict = build_readiness_report(df, dataset_name=name, target_column=effective_target).to_dict()
        st.session_state.readiness_reports[name] = report_dict

    if effective_target:
        st.session_state.readiness_target_columns[name] = effective_target
    return report_dict


def render_readiness_summary(name: str, df: pd.DataFrame, target_column: str | None = None) -> None:
    rep = readiness_dict_for(name, df, target_column=target_column)
    status_html = status_badge_html(str(rep.get("overall_status", "WARNING")))
    st.markdown(
        f"<div class='card'><b>Dataset readiness:</b> {status_html} "
        f"<span class='small-muted'>Quality score {float(rep.get('quality_score', 0)):.1f}/100 • "
        f"{int(rep.get('rows', 0)):,} rows • {int(rep.get('columns', 0)):,} columns • "
        f"{float(rep.get('memory_mb', 0)):.2f} MB</span></div>",
        unsafe_allow_html=True,
    )


def maybe_auto_train_classification(name: str, df: pd.DataFrame) -> None:
    if not st.session_state.get("auto_model_enabled", False):
        return
    if name in st.session_state.model_outputs and st.session_state.model_outputs.get(name) is not None:
        return
    if st.session_state.auto_model_done.get(name):
        return
    targets = detect_binary_targets(df)
    if not targets:
        st.session_state.auto_model_done[name] = "no_binary_target"
        return
    target_col = st.session_state.get("readiness_target_columns", {}).get(name) or infer_readiness_target(df)
    if not target_col:
        st.session_state.auto_model_done[name] = "target_confirmation_required"
        return
    st.session_state.readiness_target_columns[name] = target_col
    positive = infer_positive_label(df[target_col])
    # Avoid surprising long blocking runs for very large datasets. Manual mode can still train.
    max_rows = min(max(len(df), 5000), 120000)
    try:
        auto_timer = performance_start()
        with st.status("Automatic modelling started", expanded=False) as status:
            st.write(f"Detected target `{target_col}` and positive class `{positive}`")
            st.write("Training classification model set once and caching the result")
            output = train_models(df, target_col, positive, include_xgboost=False, max_training_rows=max_rows)
            st.session_state.model_outputs[name] = output
            # v15.4: automatically generate SHAP/fallback explanation once after training.
            # This makes reviewer questions such as "top model drivers" work without a manual extra step.
            try:
                if output.best_result is not None:
                    auto_row = output.best_result.X_test.head(1) if len(output.best_result.X_test) else df[output.best_result.feature_columns].head(1)
                    st.session_state.explanations[name] = explain_model(output.best_result, auto_row)
                    st.write(f"Auto explanation ready: {st.session_state.explanations[name].method}")
            except Exception as explain_exc:
                st.write(f"Auto explanation could not be generated yet: {explain_exc}")
            st.session_state.auto_model_done[name] = "trained + explained" if name in st.session_state.explanations else "trained"
            st.session_state.readiness_reports[name] = build_readiness_report(df, dataset_name=name, target_column=target_col).to_dict()
            record_performance_event("auto_train_and_explain", auto_timer, details={"target": target_col, "best_model": output.best_result.model_name if output.best_result else None, "model_training_seconds": getattr(output, "total_training_seconds", None)})
            try:
                record_audit_event("auto_model_trained", dataset_name=name, status="PASS", details={"target": target_col, "best_model": output.best_result.model_name if output.best_result else None})
            except Exception:
                pass
            status.update(label=f"Auto modelling/explanation complete: {output.best_result.model_name if output.best_result else 'no best model'}", state="complete")
    except Exception as exc:
        try:
            record_performance_event("auto_train_and_explain", auto_timer, status="FAIL", details={"error": str(exc)})
        except Exception:
            pass
        st.session_state.auto_model_done[name] = f"failed: {exc}"
        st.warning(f"Automatic modelling could not run: {exc}. Use the manual model page to adjust settings.")


init_state()

# Header is now compact and rendered contextually after data is loaded.

with st.sidebar:
    st.markdown("<div class='side-brand-title'>AI Solution</div>", unsafe_allow_html=True)
    st.caption("Navigation")
    sidebar_nav = [
        ("Upload Data", "⬆️ Upload", "connect/load data"),
        ("Dashboard", "🏠 Dashboard", "overview and readiness"),
        ("Cleaning & EDA", "🧹 Cleaning & EDA", "quality and exploration"),
        ("Modeling & Prediction Explanation", "🤖 Model", "prediction and explainability"),
        ("Visualization", "📈 Graphs", "auto/manual visuals"),
        ("Business Improvements", "💡 Business", "recommendations"),
        ("Chat", "💬 Chat", "Copilot Q&A"),
        ("Evaluation", "🧪 Evaluation", "proof and user study"),
        ("Export Data", "📄 Export", "reports and evidence"),
        ("Research & Memory", "🧠 Research", "contribution and memory"),
    ]
    for page_value, label, desc in sidebar_nav:
        active = " active" if st.session_state.get("current_page") == page_value else ""
        st.markdown(f"<div class='side-nav{active}'><b>{label}</b><br><span>{desc}</span></div>", unsafe_allow_html=True)
        if st.button(label, key=f"sidebar_nav_{page_value}", use_container_width=True):
            st.session_state.current_page = page_value
            st.rerun()
    st.divider()
    st.caption("Data source controls")

    st.header("Dataset Hub")
    uploads = st.file_uploader(
        "Upload CSV, TSV, Excel, JSON or ZIP",
        type=["csv", "tsv", "txt", "xlsx", "xls", "json", "zip"],
        accept_multiple_files=True,
        help="Multiple files are supported. ZIP files are scanned for CSV/Excel/JSON files.",
    )
    if st.button("Load uploaded files", use_container_width=True):
        try:
            with st.status("Loading uploaded data...", expanded=True) as status:
                st.write("Reading files and supported ZIP contents")
                loaded = load_uploaded_files(uploads)
                for ds in loaded:
                    st.write(f"Processing `{ds.name}`")
                    add_dataset(ds)
                status.update(label="Uploaded data loaded", state="complete")
            st.success(f"Loaded {len(loaded)} dataset(s).")
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

    st.divider()
    url = st.text_input("Public CSV/JSON/Excel/API URL")
    if st.button("Load URL/API data", use_container_width=True):
        try:
            with st.status("Loading public URL/API data...", expanded=True) as status:
                st.write("Reading source")
                ds = load_public_url(url)
                st.write("Cleaning and validating dataset")
                add_dataset(ds)
                status.update(label="URL/API data loaded", state="complete")
            st.success(f"Loaded `{ds.name}` from URL/API.")
        except Exception as exc:
            st.error(f"URL/API load failed: {exc}")

    st.divider()
    gdrive_url = st.text_input("Google Drive / Google Sheets shared link")
    if st.button("Load Google Drive data", use_container_width=True):
        try:
            with st.status("Loading Google Drive data...", expanded=True) as status:
                st.write("Converting shared link to a readable source")
                ds = load_google_drive_url(gdrive_url)
                st.write("Cleaning and validating dataset")
                add_dataset(ds)
                status.update(label="Google Drive data loaded", state="complete")
            st.success(f"Loaded `{ds.name}` from Google Drive/Sheets.")
        except Exception as exc:
            st.error(f"Google Drive load failed: {exc}")
            st.caption("Use a public/shared file link or Google Sheets link. Private Drive folders require OAuth and are documented as the cloud extension.")

    with st.expander("Optional / future connectors: PostgreSQL + ChromaDB"):
        st.caption("These connectors demonstrate future/optional enterprise extensions. The assessed dissertation workflow uses uploaded structured data and short-term session memory; semantic long-term memory is disabled by default.")
        store_tab, memory_tab = st.tabs(["PostgreSQL (future/optional)", "ChromaDB (future/optional)"])
        with store_tab:
            pg_uri = st.text_input("PostgreSQL URI", type="password", placeholder="postgresql+psycopg2://user:password@host:5432/db")
            pg_query = st.text_input("Table name or SELECT query", placeholder="public.customers or SELECT * FROM public.customers")
            pg_limit = st.number_input("Maximum rows", min_value=1000, max_value=500000, value=100000, step=1000, key="pg_limit")
            if st.button("Load PostgreSQL data", use_container_width=True):
                try:
                    with st.status("Connecting to PostgreSQL...", expanded=True) as status:
                        ds = load_postgres_table(pg_uri, pg_query, limit=int(pg_limit))
                        add_dataset(ds)
                        status.update(label="PostgreSQL data loaded", state="complete")
                    st.success("Loaded PostgreSQL dataset.")
                except Exception as exc:
                    st.error(f"PostgreSQL load failed: {exc}")
        with memory_tab:
            st.session_state.semantic_memory_enabled = st.checkbox(
                "Enable experimental semantic-memory logging (not part of the assessed core workflow)",
                value=st.session_state.get("semantic_memory_enabled", False),
                help="Optional future-architecture demonstration. When enabled, it stores evidence summaries in ChromaDB or a local JSONL fallback. It is OFF by default so the implemented dissertation memory remains short-term session memory.",
            )
            st.json(memory_backend_status())
            memory_query = st.text_input("Search semantic memory", placeholder="e.g. churn model evidence, feedback improvement, readiness fail")
            if st.button("Search memory", use_container_width=True):
                results = query_memory(memory_query or "dataset evidence", dataset_name=st.session_state.active_dataset, n_results=5)
                if results:
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
                else:
                    st.info("No semantic memory records found yet for this dataset.")

    st.divider()
    st.session_state.auto_model_enabled = st.checkbox(
        "Auto-model once after data load (recommended)",
        value=st.session_state.get("auto_model_enabled", True),
        help="ON by default for the dissertation demo so model, prediction and SHAP evidence are ready after upload. It runs only once per dataset and the result is cached for fast Copilot answers.",
    )

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
    st.caption("Scope guard: customer churn remains the main evaluation; extra datasets provide robustness, gap and scalability evidence.")

if not st.session_state.cleaned:
    st.session_state.current_page = "Upload Data"
    render_ai_topbar()
    render_upload_page()
    st.stop()

name = st.session_state.active_dataset
df = active_df()
assert df is not None
report = st.session_state.cleaning_reports.get(name)
maybe_auto_train_classification(name, df)
model_output = st.session_state.model_outputs.get(name)
regression_output = st.session_state.regression_outputs.get(name)
explanation_output = st.session_state.explanations.get(name)

PAGE_LABELS = [
    "Upload Data",
    "Dashboard",
    "Cleaning & EDA",
    "Modeling & Prediction Explanation",
    "Visualization",
    "Business Improvements",
    "Chat",
    "Evaluation",
    "Export Data",
    "Research & Memory",
]
if st.session_state.current_page not in PAGE_LABELS:
    st.session_state.current_page = "Dashboard"

if st.session_state.current_page == "Upload Data":
    render_ai_topbar(name, df)
    render_upload_page()
    st.stop()

current_page = st.session_state.current_page
render_ai_topbar(name, df)

if current_page != "Dashboard":
    st.markdown(
        f"<div class='page-topline'><b>{current_page}</b> "
        f"<span class='small-muted'>• Active dataset: <code>{name}</code> • {len(df):,} rows • {df.shape[1]:,} columns</span></div>",
        unsafe_allow_html=True,
    )


if current_page == "Dashboard":
    st.markdown("<div class='dashboard-section-title'>Dataset readiness and key details</div>", unsafe_allow_html=True)
    render_readiness_summary(name, df, target_column=model_output.target_column if model_output else st.session_state.get("readiness_target_columns", {}).get(name))
    metrics = overview_metrics(df)
    cols = st.columns(6)
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            render_metric_card(label, f"{value:,}" if isinstance(value, int) else value)

    st.markdown("<div class='dashboard-section-title'>Quick model status</div>", unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    auto_status = st.session_state.auto_model_done.get(name, "queued" if st.session_state.get("auto_model_enabled", True) else "manual mode")
    q1.metric("Auto-model", str(auto_status))
    if model_output and model_output.best_result:
        q2.metric("Preferred model (F1-first)", model_output.best_result.model_name)
        q3.metric("F1", f"{model_output.best_result.metrics.get('f1', 0):.3f}")
        roc = model_output.best_result.metrics.get("roc_auc", np.nan)
        q4.metric("ROC-AUC", f"{roc:.3f}" if not np.isnan(roc) else "n/a")
    else:
        q2.metric("Preferred model (F1-first)", "not trained")
        q3.metric("F1", "n/a")
        q4.metric("ROC-AUC", "n/a")

    st.markdown("<div class='dashboard-section-title'>How modelling is used in the Copilot</div>", unsafe_allow_html=True)
    st.markdown(
        """
<div class='model-use-card'>
<b>Model is not only shown as a value.</b> The trained model is used for prediction confidence, model suitability verdicts, top-driver explanations, retention-risk reasoning, and evidence-grounded Copilot answers. EDA answers use dataset statistics; prediction/explanation questions use the cached model output and SHAP/fallback explanation; business recommendations combine dataset evidence, model drivers and safety warnings when available.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='dashboard-section-title'>Research contribution snapshot</div>", unsafe_allow_html=True)
    st.info(contribution_summary())
    contrib_preview = contribution_table().head(3)
    st.dataframe(contrib_preview[["id", "contribution", "evidence"]], use_container_width=True, hide_index=True)

    st.markdown("<div class='dashboard-section-title'>Automatic visual preview</div>", unsafe_allow_html=True)
    try:
        preview_target = model_output.target_column if model_output else (detect_binary_targets(df)[0] if detect_binary_targets(df) else None)
        if preview_target:
            dist = target_distribution(df, preview_target)
            st.dataframe(dist, use_container_width=True, hide_index=True)
            fig_preview = px.bar(dist, x=preview_target, y="count", title=f"Distribution of {preview_target}")
            render_plotly_chart(fig_preview, key=f"dashboard_target_preview_{name}_{preview_target}")
        else:
            numeric_cols_preview = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols_preview:
                fig_preview = figure_numeric_distribution(sample_for_display(df), numeric_cols_preview[0])
                render_plotly_chart(fig_preview, key=f"dashboard_numeric_preview_{name}_{numeric_cols_preview[0]}")
            else:
                st.info("No target or numeric column detected for automatic graph preview.")
    except Exception as exc:
        st.info(f"Automatic preview chart could not be created safely: {exc}")


if current_page == "Dashboard":
    st.markdown("### Dashboard data details and integration readiness")
    src_meta = st.session_state.source_registry.get(name, {})
    source_cols = st.columns(4)
    source_cols[0].metric("Source type", src_meta.get("source_type", "unknown"))
    source_cols[1].metric("Rows", f"{len(df):,}")
    source_cols[2].metric("Columns", f"{df.shape[1]:,}")
    source_cols[3].metric("Domain", detect_business_domain(df))
    if src_meta.get("notes"):
        st.caption(src_meta.get("notes"))
    st.markdown("#### Data integration pipeline")
    pcols = st.columns(5)
    pipe = [
        ("1. Source", src_meta.get("source_type", "uploaded/local")),
        ("2. Ingestion", "parsed to DataFrame"),
        ("3. Cleaning", "standardised columns/types"),
        ("4. Readiness", readiness_dict_for(name, df).get("overall_status", "WARNING")),
        ("5. Evidence", "EDA/model/Copilot ready"),
    ]
    for pc, (title, desc) in zip(pcols, pipe):
        pc.markdown(f"<div class='pipeline-step'><b>{title}</b><br><span class='small-muted'>{desc}</span></div>", unsafe_allow_html=True)
    rep_dict = readiness_dict_for(name, df)
    gates = pd.DataFrame(rep_dict.get("gates", []))
    if not gates.empty:
        st.markdown("#### PASS / WARNING / FAIL readiness gates")
        st.dataframe(gates, use_container_width=True, hide_index=True)
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

if current_page == "Cleaning & EDA":
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

if current_page == "Cleaning & EDA":
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

if current_page == "Modeling & Prediction Explanation":
    st.markdown("### Predictive model centre")
    st.caption(
        "Classification is the main evaluated workflow for customer churn. "
        "Regression is included as an optional robustness mode for numeric business outcomes such as sales, revenue or profit."
    )
    auto_status = st.session_state.auto_model_done.get(name, "queued" if st.session_state.get("auto_model_enabled", True) else "manual mode")
    st.markdown(f"<div class='card'><b>Auto-modelling status:</b> <span class='small-muted'>{auto_status}</span><br><span class='small-muted'>Automatic training runs once when Auto-model is enabled (ON by default for the dissertation demo). Results are cached; manual retraining remains available below.</span></div>", unsafe_allow_html=True)
    model_tabs = st.tabs(["Classification: Yes/No targets", "Regression: numeric targets", "Model evidence and guidance"])

    with model_tabs[0]:
        st.markdown("#### Controlled classification model comparison")
        target_candidates = detect_binary_targets(df)
        if target_candidates:
            left, right = st.columns([2, 1])
            with left:
                target_options = target_candidates + [c for c in df.columns if c not in target_candidates]
                preferred_target = st.session_state.get("readiness_target_columns", {}).get(name)
                target_index = target_options.index(preferred_target) if preferred_target in target_options else 0
                target_col = st.selectbox(
                    "Select binary target",
                    target_options,
                    index=target_index,
                    key=f"model_target::{name}",
                )
                st.session_state.readiness_target_columns[name] = target_col
            with right:
                possible_values = list(df[target_col].dropna().unique()) if target_col in df.columns else []
                default_pos = infer_positive_label(df[target_col]) if target_col in df.columns and df[target_col].nunique(dropna=True) == 2 else (possible_values[0] if possible_values else None)
                positive_index = possible_values.index(default_pos) if default_pos in possible_values else 0
                positive_value = st.selectbox(
                    "Positive class",
                    possible_values,
                    index=positive_index,
                    format_func=lambda value: str(value),
                    key=f"model_pos::{name}::{target_col}",
                ) if possible_values else None
                if positive_value is not None:
                    st.caption(f"Positive/risk class: `{positive_value}`")
            st.caption("Assessed workflow: Logistic Regression, Random Forest, Gradient Boosting and MLP Neural Network Baseline.")
            inc_xgb = False
            max_rows = st.slider("Maximum rows for classification training", min_value=5000, max_value=200000, value=min(max(len(df), 5000), 120000), step=5000, help="Large datasets may be sampled for model training to keep the interface responsive.", key="cls_rows")
            if st.button("Train and compare classification models", type="primary", use_container_width=True):
                try:
                    manual_timer = performance_start()
                    with st.spinner("Training four classification models and selecting the preferred model using the F1-first rule..."):
                        output = train_models(df, target_col, positive_value, include_xgboost=False, max_training_rows=max_rows)
                        st.session_state.model_outputs[name] = output
                        st.session_state.explanations.pop(name, None)
                        st.session_state.readiness_target_columns[name] = target_col
                        st.session_state.readiness_reports[name] = build_readiness_report(
                            df, dataset_name=name, target_column=target_col
                        ).to_dict()
                    record_performance_event("manual_classification_training", manual_timer, details={"target": target_col, "best_model": output.best_result.model_name if output.best_result else None, "model_training_seconds": getattr(output, "total_training_seconds", None)})
                    st.success(f"Training complete. Preferred F1-selected model: {output.best_result.model_name}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Classification training failed: {exc}")
                    with st.expander("Error detail"):
                        st.code(traceback.format_exc())
        else:
            st.warning("No binary target detected. Classification training is disabled for this active dataset.")

        if model_output and model_output.best_result:
            st.markdown("#### Classification leaderboard")
            st.dataframe(model_output.leaderboard, use_container_width=True)
            if "training_time_sec" in model_output.leaderboard.columns:
                st.caption(f"Total model training time recorded by the modelling module: {getattr(model_output, 'total_training_seconds', 0):.3f} seconds. Per-model timings are shown in the leaderboard.")
            best = model_output.best_result
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Preferred model (F1-first)", best.model_name)
            c2.metric("F1", f"{best.metrics['f1']:.3f}")
            c3.metric("Recall", f"{best.metrics['recall']:.3f}")
            c4.metric("Precision", f"{best.metrics['precision']:.3f}")
            c5.metric("ROC-AUC", f"{best.metrics['roc_auc']:.3f}" if not np.isnan(best.metrics['roc_auc']) else "n/a")
            try:
                metric_long = model_output.leaderboard.melt(id_vars=["model"], value_vars=["f1", "recall", "precision", "roc_auc"], var_name="metric", value_name="score")
                fig_metrics = px.bar(metric_long, x="model", y="score", color="metric", barmode="group", title="Classification metric comparison")
                render_plotly_chart(fig_metrics, key=f"classification_metrics_{name}")
            except Exception:
                pass
            st.markdown("#### Confusion matrix")
            conf = pd.DataFrame(best.confusion, index=["Actual negative", "Actual positive"], columns=["Predicted negative", "Predicted positive"])
            st.dataframe(conf, use_container_width=True)
            if model_output.skipped_models:
                with st.expander("Skipped/failed models"):
                    st.json(model_output.skipped_models)
            if best.metrics["f1"] < 0.50:
                st.warning("Model performance is weak. Use explanations carefully and report this limitation honestly.")
            else:
                st.success("This classification model is suitable for decision-support demonstration, but not automatic decision-making.")

    with model_tabs[1]:
        st.markdown("#### Optional regression model comparison")
        regression_candidates = detect_regression_targets(df)
        if regression_candidates:
            reg_target = st.selectbox("Select numeric target", regression_candidates + [c for c in df.select_dtypes(include=np.number).columns if c not in regression_candidates], key="reg_target")
            st.caption("Optional regression mode uses scikit-learn baseline models only and is not part of the main assessed churn workflow.")
            inc_xgb_reg = False
            max_rows_reg = st.slider("Maximum rows for regression training", min_value=5000, max_value=200000, value=min(max(len(df), 5000), 120000), step=5000, key="reg_rows")
            if st.button("Train and compare regression models", use_container_width=True):
                try:
                    with st.spinner("Training regression models..."):
                        reg_output = train_regression_models(df, reg_target, include_xgboost=False, max_training_rows=max_rows_reg)
                        st.session_state.regression_outputs[name] = reg_output
                    st.success(f"Regression training complete. Best model: {reg_output.best_result.model_name}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Regression training failed: {exc}")
                    with st.expander("Error detail"):
                        st.code(traceback.format_exc())
        else:
            st.info("No strong continuous numeric target detected. Regression mode is optional and is only useful for sales, revenue, profit, quantity, score or time targets.")

        regression_output = st.session_state.regression_outputs.get(name)
        if regression_output and regression_output.best_result:
            st.markdown("#### Regression leaderboard")
            st.dataframe(regression_output.leaderboard, use_container_width=True)
            best_reg = regression_output.best_result
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Best regressor", best_reg.model_name)
            r2.metric("MAE", f"{best_reg.metrics['mae']:.3f}")
            r3.metric("RMSE", f"{best_reg.metrics['rmse']:.3f}")
            r4.metric("R²", f"{best_reg.metrics['r2']:.3f}" if not np.isnan(best_reg.metrics['r2']) else "n/a")
            st.dataframe(regression_output.target_summary, use_container_width=True)
            pred_df = pd.DataFrame({"actual": best_reg.y_test.reset_index(drop=True), "predicted": best_reg.y_pred}).head(1000)
            fig_scatter = px.scatter(pred_df, x="actual", y="predicted", title=f"Actual vs predicted {regression_output.target_column}")
            render_plotly_chart(fig_scatter, key=f"regression_scatter_{name}")
            if regression_output.skipped_models:
                with st.expander("Skipped/failed regression models"):
                    st.json(regression_output.skipped_models)
            st.info("Regression outputs are optional robustness evidence. They should not replace the main churn classification evaluation.")

    with model_tabs[2]:
        st.markdown("#### Model mode decision logic")
        st.markdown(
            """
| Uploaded target type | System mode | Example | Dissertation status |
|---|---|---|---|
| Yes/No, 0/1, True/False | Classification | Churn, Attrition, Exited, y | Main implemented and evaluated workflow |
| Continuous numeric | Regression | Sales, revenue, profit, delivery time | Optional robustness mode |
| No target column | EDA + Business Copilot | Retail transactions, feedback, operations logs | Implemented insight workflow |
| Action/reward history | RL / bandit extension | Retention action feedback | Future work only |
"""
        )
        st.warning("The system supports multiple dataset types, but customer churn remains the primary evaluated case study. Other modes strengthen robustness and discussion, not scope creep.")


if current_page == "Modeling & Prediction Explanation":
    st.markdown("### Prediction explanation")
    if not (model_output and model_output.best_result):
        st.info("Train a model first to generate prediction explanations.")
    else:
        best = model_output.best_result
        st.caption(f"Using preferred F1-selected model: `{best.model_name}` trained on `{name}`. F1 is the primary selection metric; ROC-AUC, recall and precision are tie-breakers.")
        row_idx = st.number_input("Select row index to explain", min_value=0, max_value=max(len(df) - 1, 0), value=0, step=1)
        row_full = df.iloc[[int(row_idx)]].copy()
        row_features = row_full[[c for c in best.feature_columns if c in row_full.columns]].copy()
        if st.button("Generate prediction and explanation", type="primary", use_container_width=True):
            try:
                pred = best.pipeline.predict(row_features)[0]
                proba = best.pipeline.predict_proba(row_features)[0, 1] if hasattr(best.pipeline, "predict_proba") else None
                explain_timer = performance_start()
                with st.spinner("Computing SHAP/fallback explanation..."):
                    explanation = explain_model(best, row_features)
                    st.session_state.explanations[name] = explanation
                record_performance_event("manual_shap_or_fallback_explanation", explain_timer, details={"method": getattr(explanation, "method", "unknown"), "model": best.model_name})
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

if current_page == "Business Improvements":
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
            st.session_state.business_last_response = {"dataset": name, "question": qbtn, "response": response}
            if response.context:
                st.session_state.chat_contexts[name] = response.context
            st.rerun()

    last_business = st.session_state.get("business_last_response")
    if last_business and last_business.get("dataset") == name:
        st.markdown("#### Latest business recommendation")
        st.caption(f"Question: {last_business['question']}")
        response = last_business["response"]
        st.markdown(response.answer)
        if response.table is not None and not response.table.empty:
            st.dataframe(response.table, use_container_width=True)
        if response.chart is not None:
            render_plotly_chart(response.chart, key=f"business_inline_chart_{name}")
        st.warning(response.safety_warning)
    else:
        st.info("Click a business question above. The answer will appear here and also be saved in the AI Copilot chat history as evaluation evidence.")
    st.warning("Safety guard: the system must not make final business, financial or HR decisions automatically. It provides data-grounded suggestions for human review.")

if current_page == "Chat":
    st.markdown("### Dataset Copilot Chat")
    st.markdown("<div class='chat-fixed-note'>Answers use only the active dataset, model metrics and explanation artefacts. The chat input stays fixed at the bottom of the page and new answers appear latest-first.</div>", unsafe_allow_html=True)

    tool_cols = st.columns([1, 1, 1.2, 1.8])
    latest_first = tool_cols[0].toggle("Latest first", value=True, help="Shows the newest answer at the top so you do not need to scroll down after every question.")
    show_context = tool_cols[1].toggle("Show context", value=False, help="Shows the last column/topic remembered for follow-up questions.")
    screenshot_mode = tool_cols[2].toggle("Screenshot mode", value=True, help="Shows shorter, presentation-ready answers while exports still keep full evidence.")
    if tool_cols[3].button("Clear chat for this dataset", use_container_width=True):
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
    st.markdown("""
<div class='reviewer-mode-box'>
  <h4>Reviewer Mode: safe high-standard test questions</h4>
  <div class='review-tip'>Use these buttons for final screenshots. They reduce spelling/intent mistakes and test readiness, modelling, explanation, business action, memory, audit and safe refusal.</div>
</div>
""", unsafe_allow_html=True)
    reviewer_questions = [
        ("Readiness", "Is this dataset ready for analysis and modelling?"),
        ("Model", "Is this model good enough for decision support?"),
        ("Drivers", "What are the top model drivers?"),
        ("Prediction evidence", "What prediction evidence supports this recommendation?"),
        ("Churn segment", "Which contract type has the highest churn?"),
        ("Tenure", "How does tenure relate to churn?"),
        ("Retention", "Which customers should be prioritised for retention?"),
        ("Feedback issues", "What technical or service issues did customers report?"),
        ("Improve", "Which area should the company improve first?"),
        ("Memory", "What does short-term memory store in this session?"),
        ("Audit", "What does the audit trail prove?"),
        ("Legal refusal", "What happens if I ask for legal advice?"),
    ]
    rq_cols = st.columns(4)
    for idx, (label, q_text) in enumerate(reviewer_questions):
        with rq_cols[idx % 4]:
            if st.button(label, key=f"reviewer_mode_{name}_{idx}", use_container_width=True):
                st.session_state.pending_question = q_text
                st.rerun()

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
        questions_to_run = split_reviewer_questions(question)
        if len(questions_to_run) > 1:
            st.info(f"Detected {len(questions_to_run)} numbered questions. I will answer them as separate evidence records so the chat does not mix different intents.")
        for single_question in questions_to_run:
            chat_timer = performance_start()
            response = answer_question(
                single_question,
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
            st.session_state.session_memory = update_short_term_memory(
                st.session_state.session_memory,
                dataset=name,
                question=single_question,
                context=response.context,
                answer_summary=response.answer,
            )
            try:
                if st.session_state.get("semantic_memory_enabled", False):
                    package = build_evidence_package(
                        dataset_name=name,
                        df=df,
                        question=single_question,
                        readiness=readiness_dict_for(name, df),
                        model_output=model_output,
                        explanation_output=st.session_state.explanations.get(name),
                        response_context=response.context,
                    )
                    memory_text = evidence_package_to_text(package) + " | Answer: " + response.answer[:1200]
                    mem_id = add_memory_document(memory_text, metadata={"dataset": name, "type": "copilot_answer", "question": single_question})
                    st.session_state.semantic_memory_events.append({"dataset": name, "type": "copilot_answer", "memory_id": mem_id, "question": single_question})
            except Exception:
                pass
            st.session_state.chat_history.append({"dataset": name, "question": single_question, "response": response})
            record_performance_event("copilot_answer_generation", chat_timer, status=getattr(response, "grounding_status", "PASS") or "PASS", details={"question": single_question[:120], "topic": (response.context or {}).get("topic"), "grounding_status": getattr(response, "grounding_status", ""), "confidence": getattr(response, "confidence", ""), "evidence_rows": len(response.table) if response.table is not None else 0})
            st.session_state.audit_events_session.append({
                "dataset": name,
                "event": "copilot_answer",
                "question": single_question,
                "interpreted_question": getattr(response, "interpreted_question", ""),
                "intent": getattr(response, "intent", "") or (response.context or {}).get("topic"),
                "grounding_status": getattr(response, "grounding_status", ""),
                "confidence": getattr(response, "confidence", ""),
                "limitation_or_refusal": getattr(response, "limitation_or_refusal", False),
                "context": response.context,
                "evidence_rows": len(response.table) if response.table is not None else 0,
            })
            try:
                record_audit_event(
                    "copilot_answer",
                    dataset_name=name,
                    status=getattr(response, "grounding_status", "WARNING") or "WARNING",
                    question=single_question,
                    details={"context": response.context, "grounding_status": getattr(response, "grounding_status", ""), "confidence": getattr(response, "confidence", ""), "evidence_rows": len(response.table) if response.table is not None else 0},
                )
            except Exception:
                pass

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
            st.markdown(compact_copilot_answer(response.answer) if screenshot_mode else response.answer)
            if response.table is not None and not response.table.empty:
                st.dataframe(response.table, use_container_width=True)
            if response.chart is not None:
                render_plotly_chart(response.chart, key=f"chat_chart_{name}_{hist_idx}_{display_idx}")
            st.warning(response.safety_warning)

if current_page == "Evaluation":
    st.markdown("### Evaluation workspace — why this page is used")
    st.info("Evaluation is used to prove that the Copilot works as a dissertation artefact. It compares Condition A: prediction-only against Condition B: prediction + explanation + recommendation + safety warning. This gives evidence for understanding, trust, usefulness and decision confidence.")
    st.caption("Use this page to collect evidence for the dissertation evaluation: prediction-only output versus explanation-supported Copilot output.")
    eval_tabs = st.tabs(["Participant task", "Questionnaire", "Collected responses"])

    with eval_tabs[0]:
        if model_output and model_output.best_result:
            best = model_output.best_result
            st.markdown("#### Condition A: prediction-only")
            sample_idx = st.number_input("Evaluation row index", min_value=0, max_value=max(len(df) - 1, 0), value=0, step=1, key=f"eval_row_{name}")
            row_eval = df.iloc[[int(sample_idx)]].copy()
            row_features_eval = row_eval[[c for c in best.feature_columns if c in row_eval.columns]].copy()
            try:
                pred_eval = best.pipeline.predict(row_features_eval)[0]
                proba_eval = best.pipeline.predict_proba(row_features_eval)[0, 1] if hasattr(best.pipeline, "predict_proba") else None
                a_cols = st.columns(3)
                a_cols[0].metric("Predicted class", str(pred_eval))
                if proba_eval is not None:
                    a_cols[1].metric("Predicted probability", f"{proba_eval:.2%}")
                    a_cols[2].metric("Risk band", "High" if proba_eval >= 0.67 else "Medium" if proba_eval >= 0.34 else "Low")
                st.markdown("Participants first make a decision using only this prediction output.")

                st.markdown("#### Condition B: explanation-supported Copilot output")
                if st.button("Generate evaluation explanation and recommendation", use_container_width=True, key=f"eval_explain_{name}"):
                    with st.spinner("Generating explanation and Copilot recommendation..."):
                        explanation = explain_model(best, row_features_eval)
                        st.session_state.explanations[name] = explanation
                    st.rerun()
                explanation_for_eval = st.session_state.explanations.get(name)
                if explanation_for_eval:
                    st.dataframe(explanation_for_eval.local_importance.head(10), use_container_width=True)
                    for sent in top_driver_sentences(explanation_for_eval.local_importance, 5):
                        st.write("• " + sent)
                    target_for_eval = model_output.target_column
                    positive_for_eval = model_output.positive_label
                    eval_q = "recommend action for this prediction"
                    resp = answer_question(eval_q, df, dataset_name=name, target_column=target_for_eval, positive_label=positive_for_eval, model_output=model_output, explanation_output=explanation_for_eval, previous_context=st.session_state.chat_contexts.get(name))
                    st.markdown(resp.answer)
                    if resp.table is not None and not resp.table.empty:
                        st.dataframe(resp.table, use_container_width=True)
                    if resp.chart is not None:
                        render_plotly_chart(resp.chart, key=f"eval_resp_chart_{name}")
                    st.warning(resp.safety_warning)
                else:
                    st.info("Generate the explanation to show Condition B.")
            except Exception as exc:
                st.error(f"Could not create evaluation task from this row: {exc}")
        else:
            st.info("Train a classification model first. The evaluation task is based on prediction-only versus explanation-supported output.")

    with eval_tabs[1]:
        st.markdown("#### Participant questionnaire")
        st.caption("Likert scale: 1 = strongly disagree, 5 = strongly agree.")
        participant_id = st.text_input("Participant code", value=f"P{len(st.session_state.evaluation_responses)+1:02d}")
        understanding = st.slider("I understood why the prediction was made.", 1, 5, 3)
        trust = st.slider("I trusted the output appropriately.", 1, 5, 3)
        usefulness = st.slider("The recommendation was useful for decision support.", 1, 5, 3)
        usability = st.slider("The system was easy to use.", 1, 5, 3)
        confidence = st.slider("I felt more confident choosing an action.", 1, 5, 3)
        safety_awareness = st.slider("The system made clear that human review is required.", 1, 5, 3)
        comments = st.text_area("Optional comments")
        if st.button("Save evaluation response", use_container_width=True):
            st.session_state.evaluation_responses.append({
                "dataset": name,
                "participant_id": participant_id,
                "understanding": understanding,
                "trust": trust,
                "usefulness": usefulness,
                "usability": usability,
                "decision_confidence": confidence,
                "safety_awareness": safety_awareness,
                "comments": comments,
            })
            st.success("Evaluation response saved in this session. Download responses from the next tab before closing the app.")

    with eval_tabs[2]:
        responses = pd.DataFrame(st.session_state.evaluation_responses)
        if responses.empty:
            st.info("No evaluation responses saved yet.")
        else:
            st.dataframe(responses, use_container_width=True)
            metrics_cols = ["understanding", "trust", "usefulness", "usability", "decision_confidence", "safety_awareness"]
            if all(c in responses.columns for c in metrics_cols):
                means = responses[metrics_cols].mean().reset_index()
                means.columns = ["measure", "average_score"]
                fig_eval = px.bar(means, x="measure", y="average_score", title="Average evaluation scores")
                render_plotly_chart(fig_eval, key=f"eval_scores_{name}")
            st.download_button("Download evaluation responses CSV", responses.to_csv(index=False), file_name="evaluation_responses.csv", mime="text/csv", use_container_width=True)


if current_page == "Research & Memory":
    st.markdown("### Research contribution, readiness criteria and memory architecture")
    st.caption("Use this page to explain the research contribution, dissertation-aligned PASS/WARNING/FAIL criteria and implemented short-term session memory. Long-term ChromaDB/retrieval is shown only as a future/optional extension.")

    prof = dataset_profile(df)
    rows = int(prof.get("rows", len(df)))
    cols_count = int(prof.get("columns", len(df.columns)))
    memory_mb = float(prof.get("memory_mb", 0) or 0)
    if rows < 100_000:
        scale_label = "Small / medium proof-of-concept dataset"
        scale_msg = "Suitable for the full local workflow: cleaning, EDA, model training, explanation, Copilot and evaluation."
    elif rows < 1_000_000:
        scale_label = "Large dataset"
        scale_msg = "Suitable for cleaning, aggregated EDA and business insight; model training should use sampling or backend services."
    else:
        scale_label = "Very large dataset"
        scale_msg = "Use aggregated analysis and sampling locally; production deployment should use database, batch jobs or distributed processing."

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Rows", f"{rows:,}")
    s2.metric("Columns", f"{cols_count:,}")
    s3.metric("Memory", f"{memory_mb:.2f} MB")
    s4.metric("Scale category", scale_label)
    st.info(scale_msg)

    st.markdown("#### Research contribution")
    st.write(contribution_summary())
    st.dataframe(contribution_table(), use_container_width=True, hide_index=True)

    st.markdown("#### Measurable PASS / WARNING / FAIL readiness criteria")
    st.dataframe(readiness_criteria_table(), use_container_width=True, hide_index=True)

    st.markdown("#### Memory architecture: implemented core vs future extension")
    st.dataframe(memory_architecture_table(), use_container_width=True, hide_index=True)
    mem_status = memory_backend_status()
    st.json(mem_status)
    stm = short_term_memory_table(st.session_state.get("session_memory", {}), dataset=name)
    if not stm.empty:
        st.markdown("#### Current short-term session memory")
        st.dataframe(stm, use_container_width=True)
    semantic_events = pd.DataFrame(st.session_state.get("semantic_memory_events", []))
    if not semantic_events.empty:
        st.markdown("#### Optional experimental semantic-memory events recorded this session")
        st.dataframe(semantic_events, use_container_width=True, hide_index=True)

    st.markdown("#### Advanced document retrieval pipeline")
    st.caption("This section demonstrates the future long-term memory design: overlapping chunking, metadata filtering, hybrid TF-IDF/BM25 search, reranking and retrieval metrics.")
    st.dataframe(retrieval_pipeline_description(), use_container_width=True, hide_index=True)
    if st.session_state.get("document_chunks"):
        r1, r2, r3 = st.columns(3)
        with r1:
            q_doc = st.text_input("Document retrieval query", placeholder="e.g. model evaluation metrics", key="research_doc_query")
        with r2:
            department_filter = st.text_input("Filter: department", key="research_doc_department_filter")
        with r3:
            topic_filter = st.text_input("Filter: topic", key="research_doc_topic_filter")
        c1, c2 = st.columns(2)
        with c1:
            candidate_limit = st.slider("Candidate chunks before reranking", min_value=10, max_value=100, value=50, step=10, key="research_candidate_limit")
        with c2:
            top_k = st.slider("Top chunks after reranking", min_value=3, max_value=10, value=5, step=1, key="research_top_k")
        if st.button("Run hybrid retrieval + reranking", use_container_width=True):
            results = hybrid_search(
                q_doc or "decision support",
                st.session_state.document_chunks,
                metadata_filters={"department": department_filter, "topic": topic_filter},
                candidate_limit=int(candidate_limit),
                top_k=int(top_k),
            )
            if results.empty:
                st.info("No matching document chunks were found after metadata filtering.")
            else:
                st.dataframe(results, use_container_width=True, hide_index=True)
                st.download_button("Download retrieval results CSV", results.to_csv(index=False), file_name="retrieval_results.csv", mime="text/csv", use_container_width=True)
                retrieved_ids = results["chunk_id"].astype(str).tolist()
                relevant_text = st.text_input("Optional evaluation: relevant chunk IDs separated by comma", key="relevant_chunk_ids")
                if relevant_text.strip():
                    relevant_ids = [x.strip() for x in relevant_text.split(",") if x.strip()]
                    metrics = evaluate_retrieval(retrieved_ids, relevant_ids, k=min(int(top_k), len(retrieved_ids)))
                    st.json(metrics)
    else:
        st.info("Upload documents from the Upload Data → Document memory tab to test hybrid retrieval.")

    st.markdown("#### Evaluation framework metrics")
    st.dataframe(evaluation_metrics_table(), use_container_width=True, hide_index=True)

    st.markdown("#### Two-level evaluation strategy")
    strategy = pd.DataFrame([
        {"Level": "Small-scale implementation", "Dataset example": "IBM/Telco Customer Churn", "What is demonstrated": "Full end-to-end workflow: cleaning, EDA, classification, SHAP, Copilot recommendation and user evaluation", "Evidence to collect": "Screenshots, model metrics, explanation output, questionnaire responses"},
        {"Level": "Large-scale robustness", "Dataset example": "UCI Online Retail / Online Retail II / operations data", "What is demonstrated": "Large upload, cleaning, aggregation, EDA, business insight and sampled modelling where relevant", "Evidence to collect": "Rows processed, memory, sampling size, runtime notes, charts and exported report"},
        {"Level": "Enterprise-scale design", "Dataset example": "Production data warehouse or data lake", "What is demonstrated": "Scalable architecture rather than full implementation", "Evidence to collect": "Architecture diagram, RBAC/audit/monitoring plan, limitations and future work"},
    ])
    st.dataframe(strategy, use_container_width=True, hide_index=True)

    st.markdown("#### Local scaling controls")
    local_controls = pd.DataFrame([
        {"Challenge": "Large upload", "Current control": "Dataset profile and memory display", "Production extension": "Object storage / data lake ingestion"},
        {"Challenge": "Slow charts", "Current control": "Sampling for display and aggregated EDA", "Production extension": "Pre-aggregations, materialised views, query engine"},
        {"Challenge": "Slow model training", "Current control": "Configurable max training rows", "Production extension": "Scheduled training jobs or scalable ML service"},
        {"Challenge": "Explanation cost", "Current control": "SHAP where possible, fallback importance", "Production extension": "Cached explanations and explanation service"},
        {"Challenge": "Governance", "Current control": "Safety warnings and dataset-grounded answers", "Production extension": "RBAC, audit logs, model registry and monitoring"},
    ])
    st.dataframe(local_controls, use_container_width=True, hide_index=True)

    if st.button("Record scale evidence for this dataset", use_container_width=True):
        st.session_state.scale_evidence.append({
            "dataset": name,
            "rows": rows,
            "columns": cols_count,
            "memory_mb": memory_mb,
            "scale_category": scale_label,
            "domain": detect_business_domain(df),
            "classification_model_trained": bool(model_output and model_output.best_result),
            "regression_model_trained": bool(regression_output and regression_output.best_result),
        })
        st.success("Scale evidence recorded for this session.")

    evidence_df = pd.DataFrame(st.session_state.scale_evidence)
    if not evidence_df.empty:
        st.markdown("#### Recorded scale evidence")
        st.dataframe(evidence_df, use_container_width=True, hide_index=True)
        st.download_button("Download scale evidence CSV", evidence_df.to_csv(index=False), file_name="scale_evidence.csv", mime="text/csv", use_container_width=True)

    scale_doc = Path(__file__).parent / "docs" / "scalability" / "small_large_scale_strategy.md"
    if scale_doc.exists():
        st.download_button("Download small/large-scale strategy", scale_doc.read_text(encoding="utf-8"), file_name="small_large_scale_strategy.md", mime="text/markdown", use_container_width=True)

    st.markdown("#### Enterprise extension architecture")
    st.code("""
Frontend UI
  -> API gateway / FastAPI backend
  -> Data ingestion and readiness validation service
  -> PostgreSQL structured store / data warehouse
  -> ChromaDB semantic memory layer
  -> Model training and model registry service
  -> Explanation service (SHAP/fallback)
  -> Evidence-package and Copilot answer service
  -> Audit log, RBAC, monitoring and human approval workflow
""", language="text")

if current_page == "Export Data":
    st.markdown("### Export data, reports, chat answers and audit evidence")
    readiness_current = readiness_dict_for(name, df, target_column=model_output.target_column if model_output else None)
    export_timer = performance_start()
    full_package = build_results_export_package(
        dataset_name=name,
        df=df,
        cleaning_report=report,
        readiness=readiness_current,
        model_output=model_output,
        explanation_output=st.session_state.explanations.get(name),
        chat_history=st.session_state.chat_history,
        audit_events=st.session_state.audit_events_session,
        performance_events=st.session_state.performance_events,
    )
    record_performance_event("build_full_results_export_package", export_timer, details={"package_bytes": len(full_package)})
    # Rebuild once so the ZIP itself contains the export-build timing event that
    # was just recorded. The second packaging pass is not separately timed to
    # avoid recursive/self-inflating timing records.
    full_package = build_results_export_package(
        dataset_name=name,
        df=df,
        cleaning_report=report,
        readiness=readiness_current,
        model_output=model_output,
        explanation_output=st.session_state.explanations.get(name),
        chat_history=st.session_state.chat_history,
        audit_events=st.session_state.audit_events_session,
        performance_events=st.session_state.performance_events,
    )
    st.download_button(
        "Download FULL results package (HTML report + graphs + tables + chat + audit)",
        full_package,
        file_name=f"{name}_full_results_package.zip",
        mime="application/zip",
        use_container_width=True,
        type="primary",
    )
    st.caption("Open `full_results_report.html` inside the ZIP for the complete dissertation evidence report with graphs.")
    markdown = make_markdown_report(
        name,
        df,
        report,
        model_output,
        st.session_state.explanations.get(name),
        readiness=readiness_current,
        chat_history=st.session_state.chat_history,
        performance_events=st.session_state.performance_events,
        audit_events=st.session_state.audit_events_session,
    )
    st.download_button("Download Markdown report", markdown, file_name=f"{name}_copilot_report.md", mime="text/markdown", use_container_width=True)
    st.download_button("Download cleaned active dataset CSV", df.to_csv(index=False), file_name=f"{name}_cleaned.csv", mime="text/csv", use_container_width=True)
    if report:
        st.download_button("Download cleaning report CSV", report.to_dataframe().to_csv(index=False), file_name=f"{name}_cleaning_report.csv", mime="text/csv", use_container_width=True)
    readiness_export = pd.DataFrame(readiness_current.get("gates", []))
    if not readiness_export.empty:
        st.download_button("Download readiness gates CSV", readiness_export.to_csv(index=False), file_name=f"{name}_readiness_gates.csv", mime="text/csv", use_container_width=True)
    explanation_export = st.session_state.explanations.get(name)
    if explanation_export is not None:
        global_exp = getattr(explanation_export, "global_importance", pd.DataFrame())
        local_exp = getattr(explanation_export, "local_importance", pd.DataFrame())
        if global_exp is not None and not global_exp.empty:
            st.download_button("Download global explanation drivers CSV", global_exp.to_csv(index=False), file_name=f"{name}_top_explanation_drivers.csv", mime="text/csv", use_container_width=True)
        if local_exp is not None and not local_exp.empty:
            st.download_button("Download local prediction explanation drivers CSV", local_exp.to_csv(index=False), file_name=f"{name}_local_prediction_explanation_drivers.csv", mime="text/csv", use_container_width=True)
    audit_session = pd.DataFrame(st.session_state.audit_events_session)
    if not audit_session.empty:
        st.download_button("Download session audit log CSV", audit_session.to_csv(index=False), file_name=f"{name}_session_audit_log.csv", mime="text/csv", use_container_width=True)
    perf_export = pd.DataFrame(st.session_state.get("performance_events", []))
    if not perf_export.empty:
        st.markdown("#### Runtime/process evidence")
        st.dataframe(perf_export, use_container_width=True, hide_index=True)
        st.download_button("Download runtime/process evidence CSV", perf_export.to_csv(index=False), file_name=f"{name}_runtime_process_evidence.csv", mime="text/csv", use_container_width=True)
        runtime_summary = build_runtime_summary(perf_export, model_output)
        st.markdown("#### Whole computational process-time summary")
        st.caption("This sums recorded computational stages and excludes time while a user is reading or typing questions.")
        st.dataframe(runtime_summary, use_container_width=True, hide_index=True)
        st.download_button("Download runtime summary CSV", runtime_summary.to_csv(index=False), file_name=f"{name}_runtime_summary.csv", mime="text/csv", use_container_width=True)
    # Export Copilot chat history with outputs for dissertation evidence.
    active_chat_rows = []
    if st.session_state.chat_history and isinstance(st.session_state.chat_history[0], dict):
        for item in st.session_state.chat_history:
            if item.get("dataset") == name:
                resp = item.get("response")
                active_chat_rows.append({
                    "dataset": name,
                    "question": item.get("question", ""),
                    "interpreted_question": getattr(resp, "interpreted_question", ""),
                    "intent": getattr(resp, "intent", "") or str((getattr(resp, "context", {}) or {}).get("topic", "")),
                    "grounding_status": getattr(resp, "grounding_status", ""),
                    "grounding_reason": getattr(resp, "grounding_reason", ""),
                    "confidence": getattr(resp, "confidence", ""),
                    "response_type": getattr(resp, "response_type", "GROUNDED_ANSWER"),
                    "limitation_or_refusal": getattr(resp, "limitation_or_refusal", False),
                    "answer": getattr(resp, "answer", ""),
                    "safety_warning": getattr(resp, "safety_warning", ""),
                    "evidence_rows": len(getattr(resp, "table", pd.DataFrame())) if getattr(resp, "table", None) is not None else 0,
                    "has_chart": bool(getattr(resp, "chart", None) is not None),
                })
    chat_export_df = pd.DataFrame(active_chat_rows)
    if not chat_export_df.empty:
        st.download_button("Download Copilot chat answers CSV", chat_export_df.to_csv(index=False), file_name=f"{name}_copilot_chat_answers.csv", mime="text/csv", use_container_width=True)
        evaluation_summary = build_copilot_evaluation_summary(chat_export_df)
        st.markdown("#### Copilot evaluation/test summary")
        st.dataframe(evaluation_summary, use_container_width=True, hide_index=True)
        st.download_button("Download Copilot evaluation summary CSV", evaluation_summary.to_csv(index=False), file_name=f"{name}_copilot_evaluation_summary.csv", mime="text/csv", use_container_width=True)
        chat_md = "\n\n".join([
            f"## Question\n{row['question']}\n\n## Answer\n{row['answer']}\n\n## Safety warning\n{row['safety_warning']}"
            for _, row in chat_export_df.iterrows()
        ])
        st.download_button("Download Copilot chat answers Markdown", chat_md, file_name=f"{name}_copilot_chat_answers.md", mime="text/markdown", use_container_width=True)
    else:
        st.info("No Copilot chat answers are available yet for export. Ask questions in the Chat page first.")

    memory_export = build_memory_allocation_evidence(
        df=df,
        readiness=readiness_current,
        model_output=model_output,
        explanation_output=explanation_export,
        chat_df=chat_export_df,
        audit_df=audit_session,
        perf_df=perf_export,
    )
    st.markdown("#### Memory allocation evidence")
    st.caption("Shows tracked short-term session artefacts plus current/peak Python-process RSS. Long-term ChromaDB/PostgreSQL memory remains future/optional and is disabled by default.")
    st.dataframe(memory_export, use_container_width=True, hide_index=True)
    st.download_button("Download memory allocation evidence CSV", memory_export.to_csv(index=False), file_name=f"{name}_memory_allocation_evidence.csv", mime="text/csv", use_container_width=True)

    semantic_events_df = pd.DataFrame(st.session_state.get("semantic_memory_events", []))
    if not semantic_events_df.empty:
        st.download_button("Download semantic memory events CSV", semantic_events_df.to_csv(index=False), file_name=f"{name}_semantic_memory_events.csv", mime="text/csv", use_container_width=True)
    research_bundle = "# Research contribution\n\n" + contribution_summary() + "\n\n" + contribution_table().to_markdown(index=False) + "\n\n# Readiness criteria\n\n" + readiness_criteria_table().to_markdown(index=False) + "\n\n# Memory architecture\n\n" + memory_architecture_table().to_markdown(index=False) + "\n\n# Evaluation metrics\n\n" + evaluation_metrics_table().to_markdown(index=False)
    st.download_button("Download research framework Markdown", research_bundle, file_name="research_memory_readiness_framework.md", mime="text/markdown", use_container_width=True)

    try:
        sqlite_audit = read_audit_events()
        if not sqlite_audit.empty:
            st.download_button("Download SQLite audit events CSV", sqlite_audit.to_csv(index=False), file_name="sqlite_audit_events.csv", mime="text/csv", use_container_width=True)
    except Exception:
        pass



if current_page == "Visualization":
    st.markdown("### Visualisation: automatic dashboard and manual visual builder")
    st.caption(
        "Use Auto view for generated visuals or scroll to Manual builder to create a graph using selected columns."
    )
    view_mode = st.radio("Visual mode", ["Auto dashboard", "Manual visual builder"], horizontal=True, key=f"visual_mode_{name}")
    if view_mode == "Manual visual builder":
        st.info("Manual graph controls are shown below under Custom visual builder. Auto charts remain available by switching back to Auto dashboard.")

    binary_targets_visual = detect_binary_targets(df)
    visual_target = None
    visual_positive = None
    if model_output:
        visual_target = model_output.target_column
        visual_positive = model_output.positive_label
    elif binary_targets_visual:
        visual_target = binary_targets_visual[0]
        try:
            visual_positive = infer_positive_label(df[visual_target])
        except Exception:
            visual_positive = None

    dashboard = build_visual_dashboard(
        df,
        dataset_name=name,
        target_col=visual_target,
        positive_label=visual_positive,
        max_figures=10,
    )

    st.info(dashboard["large_dataset_note"])
    st.markdown("#### Executive visual summary")
    kpi_cols = st.columns(4)
    for idx, card in enumerate(dashboard["cards"][:8]):
        with kpi_cols[idx % 4]:
            st.markdown(
                f"<div class='card'><div class='metric-label'>{card.label}</div>"
                f"<div class='metric-value'>{card.value}</div>"
                f"<div class='small-muted'>{card.note}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### Recommended charts for this dataset")
    st.caption(
        "The system selects charts based on detected dataset type, target availability, date/product/customer fields, "
        "data quality and scale. Large datasets are summarised using aggregation or sampling."
    )
    figures = dashboard.get("figures", [])
    if not figures:
        st.warning("No suitable visual charts could be created for this dataset. Check whether columns have usable numeric, categorical, date or target values.")
    else:
        for idx, visual in enumerate(figures):
            st.markdown(f"#### {idx + 1}. {visual.title}")
            st.caption(visual.description)
            render_plotly_chart(visual.fig, key=f"visual_dashboard_{name}_{idx}_{visual.title}")
            if visual.evidence is not None and not visual.evidence.empty:
                with st.expander("Show evidence table for this chart"):
                    st.dataframe(visual.evidence.head(30), use_container_width=True, hide_index=True)



    st.markdown("#### Custom visual builder")
    st.caption(
        "Use this field-picker to create your own business chart from two or more columns. "
        "This is more reliable than drag-and-drop alone and is easier to evaluate for the dissertation."
    )
    all_columns = list(df.columns)
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [c for c in all_columns if c not in numeric_columns]
    date_like_columns = []
    for c in all_columns:
        if str(c).lower() in {"date", "time", "month", "year"} or any(k in str(c).lower() for k in ["date", "time", "month", "year", "invoice"]):
            converted = pd.to_datetime(df[c], errors="coerce")
            if converted.notna().mean() > 0.40:
                date_like_columns.append(c)

    builder_left, builder_mid, builder_right = st.columns(3)
    with builder_left:
        chart_type = st.selectbox(
            "Chart type",
            ["Bar", "Line", "Scatter", "Box", "Pie"],
            key=f"custom_chart_type_{name}",
            help="Bar for comparisons, line for trends, scatter for numeric relationships, box for distribution by group, pie for simple share views.",
        )
        x_col = st.selectbox("X / category / date field", all_columns, key=f"custom_x_{name}")
    with builder_mid:
        y_options = ["Record count"] + numeric_columns
        y_col = st.selectbox("Y / numeric value", y_options, key=f"custom_y_{name}")
        agg = st.selectbox("Aggregation", ["sum", "mean", "median", "count"], key=f"custom_agg_{name}")
    with builder_right:
        colour_options = ["None"] + categorical_columns
        colour_col = st.selectbox("Optional group/colour field", colour_options, key=f"custom_colour_{name}")
        top_n = st.slider("Top categories", min_value=5, max_value=30, value=12, step=1, key=f"custom_topn_{name}")

    if st.button("Generate custom visual", use_container_width=True, key=f"generate_custom_visual_{name}"):
        try:
            colour = None if colour_col == "None" else colour_col
            builder_sample = df.copy()
            if len(builder_sample) > 100000 and chart_type in {"Scatter", "Box"}:
                builder_sample = builder_sample.sample(100000, random_state=42)
                st.info("Large dataset detected: the custom scatter/box chart uses a 100,000-row sample for responsiveness.")

            evidence = pd.DataFrame()
            fig = None
            value_name = "record_count" if y_col == "Record count" else f"{agg}_{y_col}"

            if chart_type == "Scatter":
                if x_col not in numeric_columns or y_col == "Record count" or y_col not in numeric_columns:
                    st.warning("Scatter charts need numeric X and numeric Y fields.")
                else:
                    fig = px.scatter(
                        builder_sample,
                        x=x_col,
                        y=y_col,
                        color=colour if colour in builder_sample.columns else None,
                        opacity=0.65,
                        title=f"{x_col} vs {y_col}",
                        hover_data=[c for c in [colour] if c],
                    )
                    evidence = builder_sample[[c for c in [x_col, y_col, colour] if c and c in builder_sample.columns]].head(100)

            elif chart_type == "Box":
                if y_col == "Record count" or y_col not in numeric_columns:
                    st.warning("Box charts need a numeric Y field.")
                else:
                    box_df = builder_sample.copy()
                    if box_df[x_col].nunique(dropna=True) > top_n:
                        top_values = box_df[x_col].astype("string").value_counts().head(top_n).index
                        box_df = box_df[box_df[x_col].astype("string").isin(top_values)]
                    fig = px.box(
                        box_df,
                        x=x_col,
                        y=y_col,
                        color=colour if colour in box_df.columns else None,
                        points=False,
                        title=f"Distribution of {y_col} by {x_col}",
                    )
                    evidence = box_df[[c for c in [x_col, y_col, colour] if c and c in box_df.columns]].head(100)

            elif chart_type == "Pie":
                if y_col == "Record count" or agg == "count":
                    grouped = df.groupby(x_col, dropna=False).size().reset_index(name="record_count")
                    value_name = "record_count"
                else:
                    grouped = getattr(df.groupby(x_col, dropna=False)[y_col], agg)().reset_index(name=value_name)
                grouped[x_col] = grouped[x_col].astype("string").fillna("Missing")
                evidence = grouped.sort_values(value_name, ascending=False).head(top_n)
                fig = px.pie(evidence, names=x_col, values=value_name, hole=0.45, title=f"Share of {value_name} by {x_col}")

            elif chart_type == "Line":
                temp = df.copy()
                is_date = x_col in date_like_columns
                if is_date:
                    temp[x_col] = pd.to_datetime(temp[x_col], errors="coerce")
                    temp = temp.dropna(subset=[x_col])
                    temp["period"] = temp[x_col].dt.to_period("M").dt.to_timestamp()
                    group_keys = ["period"] + ([colour] if colour else [])
                    x_plot = "period"
                else:
                    group_keys = [x_col] + ([colour] if colour else [])
                    x_plot = x_col
                if y_col == "Record count" or agg == "count":
                    grouped = temp.groupby(group_keys, dropna=False).size().reset_index(name="record_count")
                    value_name = "record_count"
                else:
                    grouped = getattr(temp.groupby(group_keys, dropna=False)[y_col], agg)().reset_index(name=value_name)
                if not is_date:
                    grouped = grouped.sort_values(value_name, ascending=False).head(top_n)
                evidence = grouped.head(100)
                fig = px.line(grouped, x=x_plot, y=value_name, color=colour if colour else None, markers=True, title=f"Trend of {value_name} by {x_col}")

            else:  # Bar
                group_keys = [x_col] + ([colour] if colour else [])
                if y_col == "Record count" or agg == "count":
                    grouped = df.groupby(group_keys, dropna=False).size().reset_index(name="record_count")
                    value_name = "record_count"
                else:
                    grouped = getattr(df.groupby(group_keys, dropna=False)[y_col], agg)().reset_index(name=value_name)
                grouped[x_col] = grouped[x_col].astype("string").fillna("Missing")
                if colour:
                    top_values = grouped.groupby(x_col)[value_name].sum().sort_values(ascending=False).head(top_n).index
                    grouped = grouped[grouped[x_col].isin(top_values)]
                    evidence = grouped.sort_values(value_name, ascending=False).head(100)
                    fig = px.bar(grouped, x=x_col, y=value_name, color=colour, title=f"{value_name} by {x_col} and {colour}")
                else:
                    evidence = grouped.sort_values(value_name, ascending=False).head(top_n)
                    fig = px.bar(evidence.sort_values(value_name), x=value_name, y=x_col, orientation="h", title=f"Top {x_col} by {value_name}")

            if fig is not None:
                render_plotly_chart(fig, key=f"custom_visual_{name}_{chart_type}_{x_col}_{y_col}_{colour_col}_{agg}_{top_n}")
                st.markdown("**Evidence used for this custom chart**")
                st.dataframe(evidence, use_container_width=True, hide_index=True)
                st.caption("Custom visuals are generated only from the active uploaded dataset. For large datasets, charts use aggregation or sampling where needed.")
        except Exception as exc:
            st.error(f"Custom visual could not be generated safely: {exc}")

    st.markdown("#### How these visuals improve the research contribution")
    visual_contribution = pd.DataFrame([
        {
            "Improvement": "BI-style evidence layer",
            "Why it matters": "Power BI and Tableau are strong at visual analytics, so the prototype must also show clear interactive evidence.",
            "How implemented": "Auto-selected Plotly charts, KPI cards, data quality gauge, target views, domain dashboards and evidence tables.",
        },
        {
            "Improvement": "Explainable visual decision support",
            "Why it matters": "Graphs alone do not answer what action should be considered.",
            "How implemented": "Charts are placed beside Copilot explanations, model evidence, business insight and human-review warnings.",
        },
        {
            "Improvement": "Small and large scale readiness",
            "Why it matters": "Large datasets cannot be visualised row-by-row inside a local web prototype.",
            "How implemented": "Aggregation, top-N ranking, sampling for scatter/distribution charts and explicit large-data notes.",
        },
    ])
    st.dataframe(visual_contribution, use_container_width=True, hide_index=True)

    st.warning(
        "Do not claim this visual dashboard is better than commercial BI tools. "
        "The research contribution is the integration of visual evidence with explainable modelling, dataset-grounded Copilot answers, business recommendations and safety warnings."
    )
