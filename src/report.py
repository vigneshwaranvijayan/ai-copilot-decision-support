"""Report and evidence export utilities for the Streamlit prototype."""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import json
import html as html_lib
import zipfile
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .performance_monitor import (
    build_memory_allocation_evidence,
    build_runtime_summary,
    build_copilot_evaluation_summary,
    process_rss_mb,
    process_peak_rss_mb,
)


def dataframe_to_markdown_safe(df: pd.DataFrame, index: bool = False) -> str:
    """Return a markdown table without requiring optional formatting packages."""
    if df is None or df.empty:
        return "_No rows available._"
    table = df.copy()
    if index:
        table = table.reset_index()
    table = table.astype(object).where(pd.notna(table), "")

    def clean_cell(value: object) -> str:
        return str(value).replace("\n", " ").replace("|", "\\|")

    headers = [clean_cell(c) for c in table.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(clean_cell(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def _df_to_html_table(df: Optional[pd.DataFrame], max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "<p><em>No rows available.</em></p>"
    view = df.head(max_rows).copy()
    return view.to_html(index=False, escape=True, border=0, classes="evidence-table")


def _safe_to_csv(df: Optional[pd.DataFrame]) -> bytes:
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")


def _plotly_html(fig: Any) -> str:
    if fig is None:
        return ""
    try:
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception:
        return ""


def _readiness_to_df(readiness: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if not readiness:
        return pd.DataFrame()
    return pd.DataFrame(readiness.get("gates", []))


def _chat_rows(chat_history: Optional[Iterable[Dict[str, Any]]], dataset_name: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in chat_history or []:
        if item.get("dataset") != dataset_name:
            continue
        resp = item.get("response")
        rows.append({
            "dataset": dataset_name,
            "question": item.get("question", ""),
            "interpreted_question": getattr(resp, "interpreted_question", ""),
            "intent": getattr(resp, "intent", "") or str((getattr(resp, "context", {}) or {}).get("topic", "")),
            "grounding_status": getattr(resp, "grounding_status", ""),
            "grounding_reason": getattr(resp, "grounding_reason", ""),
            "confidence": getattr(resp, "confidence", ""),
            "response_type": getattr(resp, "response_type", "GROUNDED_ANSWER"),
            "limitation_or_refusal": bool(getattr(resp, "limitation_or_refusal", False)),
            "answer": getattr(resp, "answer", ""),
            "safety_warning": getattr(resp, "safety_warning", ""),
            "evidence_rows": len(getattr(resp, "table", pd.DataFrame())) if getattr(resp, "table", None) is not None else 0,
            "has_chart": bool(getattr(resp, "chart", None) is not None),
        })
    return pd.DataFrame(rows)



def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_packaged_software_test_results() -> str:
    """Return the verified software-test report packaged with this code version."""
    path = _project_root() / "TEST_RESULTS.txt"
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except Exception:
        return ""


def _load_questionnaire_template_bytes() -> bytes:
    path = _project_root() / "evaluation" / "questionnaire_template.csv"
    try:
        return path.read_bytes() if path.exists() else b""
    except Exception:
        return b""


def _normalise_eval_question(value: object) -> str:
    import re
    text = "" if value is None else str(value).strip().lower()
    text = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_reviewer_100_questions() -> List[str]:
    path = _project_root() / "evaluation" / "reviewer_100_questions.txt"
    try:
        lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    except Exception:
        return []
    out: List[str] = []
    import re
    for line in lines:
        out.append(re.sub(r"^\s*\d+\s*[\.\)]\s*", "", line).strip())
    return out


def build_reviewer_100_results(chat_df: pd.DataFrame) -> pd.DataFrame:
    """Return one matched row per official dissertation reviewer question.

    Manual exploratory questions are excluded so the 100-question regression
    result is not diluted by unrelated chat history.
    """
    expected = _load_reviewer_100_questions()
    if not expected:
        return pd.DataFrame()
    work = chat_df.copy() if chat_df is not None else pd.DataFrame()
    if work.empty or "question" not in work.columns:
        return pd.DataFrame([{
            "question_number": i,
            "question": q,
            "matched": False,
            "grounding_status": "NOT_RUN",
            "response_type": "NOT_RUN",
            "intent": "",
        } for i, q in enumerate(expected, 1)])
    work["__norm_q"] = work["question"].map(_normalise_eval_question)
    rows = []
    for i, q in enumerate(expected, 1):
        norm = _normalise_eval_question(q)
        matches = work[work["__norm_q"] == norm]
        if matches.empty:
            rows.append({
                "question_number": i,
                "question": q,
                "matched": False,
                "grounding_status": "NOT_RUN",
                "response_type": "NOT_RUN",
                "intent": "",
            })
        else:
            row = matches.iloc[-1]
            rows.append({
                "question_number": i,
                "question": q,
                "matched": True,
                "grounding_status": row.get("grounding_status", ""),
                "response_type": row.get("response_type", ""),
                "intent": row.get("intent", ""),
                "interpreted_question": row.get("interpreted_question", ""),
                "answer": row.get("answer", ""),
                "safety_warning": row.get("safety_warning", ""),
            })
    return pd.DataFrame(rows)


def build_reviewer_100_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df is None or results_df.empty:
        return pd.DataFrame([{
            "metric": "expected_questions",
            "value": 100,
            "notes": "Official reviewer question set not available.",
        }])
    matched = results_df["matched"].fillna(False).astype(bool)
    statuses = results_df["grounding_status"].astype(str)
    types = results_df["response_type"].astype(str)
    return pd.DataFrame([
        {"metric": "expected_questions", "value": int(len(results_df)), "notes": "Official reviewer regression question count."},
        {"metric": "matched_questions", "value": int(matched.sum()), "notes": "Official questions found as individual saved chat records."},
        {"metric": "grounding_PASS", "value": int(((statuses == "PASS") & matched).sum()), "notes": "PASS among matched official questions."},
        {"metric": "grounding_WARNING", "value": int(((statuses == "WARNING") & matched).sum()), "notes": "WARNING among matched official questions."},
        {"metric": "grounding_FAIL", "value": int(((statuses == "FAIL") & matched).sum()), "notes": "FAIL among matched official questions."},
        {"metric": "safe_refusal_count", "value": int(((types == "SAFE_REFUSAL") & matched).sum()), "notes": "Correct safety refusals within the official question set."},
        {"metric": "routing_failure_count", "value": int(((types == "ROUTING_FAILURE") & matched).sum()), "notes": "Routing failures within the official question set."},
        {"metric": "matched_pass_rate_percent", "value": round(float((((statuses == "PASS") & matched).sum()) / max(int(matched.sum()), 1) * 100), 2), "notes": "PASS / matched official questions."},
    ])


def make_markdown_report(
    dataset_name: str,
    df: pd.DataFrame,
    cleaning_report: Optional[object] = None,
    model_output: Optional[object] = None,
    explanation_output: Optional[object] = None,
    readiness: Optional[Dict[str, Any]] = None,
    chat_history: Optional[Iterable[Dict[str, Any]]] = None,
    performance_events: Optional[Iterable[Dict[str, Any]]] = None,
    audit_events: Optional[Iterable[Dict[str, Any]]] = None,
) -> str:
    lines = [
        "# Explainable AI Copilot Dataset Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Active dataset: `{dataset_name}`",
        "",
        "## Dataset overview",
        "",
        f"- Rows: {df.shape[0]}",
        f"- Columns: {df.shape[1]}",
        f"- Missing cells: {int(df.isna().sum().sum())}",
        f"- Duplicate rows: {int(df.duplicated().sum())}",
        "",
    ]
    if readiness:
        lines.extend([
            "## Data-readiness result",
            "",
            f"- Overall readiness: {readiness.get('overall_status', 'unknown')}",
            f"- Quality score: {readiness.get('quality_score', 'unknown')}/100",
            "",
            dataframe_to_markdown_safe(_readiness_to_df(readiness), index=False),
            "",
        ])
    if cleaning_report is not None:
        lines.extend(["## Cleaning summary", ""])
        try:
            for _, row in cleaning_report.to_dataframe().iterrows():
                lines.append(f"- {row['Step']}: {row['Result']}")
            lines.append("")
        except Exception:
            pass
    if model_output is not None and getattr(model_output, "best_result", None) is not None:
        best = model_output.best_result
        lines.extend([
            "## Model summary",
            "",
            f"- Target column: `{model_output.target_column}`",
            f"- Positive class: `{model_output.positive_label}`",
            "- Selection rule: highest F1-score; ROC-AUC, recall and precision are tie-breakers.",
            f"- Preferred model under F1-first selection rule: `{best.model_name}`",
            f"- F1: {best.metrics.get('f1', float('nan')):.3f}",
            f"- Recall: {best.metrics.get('recall', float('nan')):.3f}",
            f"- Precision: {best.metrics.get('precision', float('nan')):.3f}",
            f"- ROC-AUC: {best.metrics.get('roc_auc', float('nan')):.3f}",
            f"- Total training time: {getattr(model_output, 'total_training_seconds', 0):.3f} seconds",
            "",
            "Model comparison:",
            "",
            dataframe_to_markdown_safe(model_output.leaderboard, index=False),
            "",
        ])
    if explanation_output is not None and getattr(explanation_output, "global_importance", None) is not None:
        lines.extend(["## Explanation summary", ""])
        lines.append(f"Explanation method: {getattr(explanation_output, 'method', 'Unknown')}")
        lines.append("")
        lines.append("### Global explanation drivers")
        lines.append("")
        lines.append(dataframe_to_markdown_safe(explanation_output.global_importance.head(10), index=False))
        lines.append("")
        local = getattr(explanation_output, "local_importance", pd.DataFrame())
        if local is not None and not local.empty:
            lines.append("### Local explanation drivers for the currently explained record")
            lines.append("")
            lines.append(dataframe_to_markdown_safe(local.head(10), index=False))
            lines.append("")
    chat_df = _chat_rows(chat_history, dataset_name)
    if not chat_df.empty:
        lines.extend(["## Copilot chat evidence", ""])
        for _, row in chat_df.iterrows():
            lines.append(f"### Question: {row['question']}")
            lines.append("")
            lines.append(str(row["answer"]))
            lines.append("")
            lines.append(f"Safety warning: {row['safety_warning']}")
            lines.append("")
    perf_df = pd.DataFrame(performance_events or [])
    audit_df = pd.DataFrame(audit_events or [])
    if not perf_df.empty:
        lines.extend(["## Runtime and process evidence", "", dataframe_to_markdown_safe(perf_df, index=False), ""])
        runtime_summary = build_runtime_summary(perf_df, model_output)
        lines.extend(["### Runtime summary", "", dataframe_to_markdown_safe(runtime_summary, index=False), ""])
    memory_df = build_memory_allocation_evidence(
        df=df,
        readiness=readiness,
        model_output=model_output,
        explanation_output=explanation_output,
        chat_df=chat_df,
        audit_df=audit_df,
        perf_df=perf_df,
    )
    lines.extend(["## Memory allocation evidence", "", dataframe_to_markdown_safe(memory_df, index=False), ""])
    eval_summary = build_copilot_evaluation_summary(chat_df)
    lines.extend(["## Copilot evaluation summary", "", dataframe_to_markdown_safe(eval_summary, index=False), ""])
    official_results = build_reviewer_100_results(chat_df)
    official_summary = build_reviewer_100_summary(official_results)
    lines.extend(["## Official 100-question regression summary", "", dataframe_to_markdown_safe(official_summary, index=False), ""])
    if not audit_df.empty:
        lines.extend(["## Audit/session events", "", dataframe_to_markdown_safe(audit_df, index=False), ""])
    software_tests = _load_packaged_software_test_results()
    if software_tests:
        lines.extend(["## Packaged software test evidence", "", "```text", software_tests.strip(), "```", ""])
    lines.extend([
        "## User-centred evaluation status",
        "",
        "The questionnaire template is included in the ZIP, but participant results are not auto-generated. Real participant feedback must be collected and reported separately; the system does not fabricate human-evaluation evidence.",
        "",
    ])
    lines.extend([
        "## Safety note",
        "",
        "The outputs are decision support only. A human reviewer should verify the context, data quality and business implications before taking action.",
    ])
    return "\n".join(lines)


def build_results_export_package(
    *,
    dataset_name: str,
    df: pd.DataFrame,
    cleaning_report: Optional[object] = None,
    readiness: Optional[Dict[str, Any]] = None,
    model_output: Optional[object] = None,
    explanation_output: Optional[object] = None,
    chat_history: Optional[Iterable[Dict[str, Any]]] = None,
    audit_events: Optional[Iterable[Dict[str, Any]]] = None,
    performance_events: Optional[Iterable[Dict[str, Any]]] = None,
) -> bytes:
    """Build one ZIP evidence package for dissertation results.

    The ZIP contains a readable HTML report, markdown report, CSV evidence tables
    and JSON metadata. Plotly charts are embedded in the HTML report so the user
    can open the report locally and see graphs without manually exporting each
    chart.
    """
    readiness_df = _readiness_to_df(readiness)
    chat_df = _chat_rows(chat_history, dataset_name)
    audit_df = pd.DataFrame(audit_events or [])
    perf_df = pd.DataFrame(performance_events or [])
    runtime_summary_df = build_runtime_summary(perf_df, model_output)
    evaluation_summary_df = build_copilot_evaluation_summary(chat_df)
    reviewer_100_results_df = build_reviewer_100_results(chat_df)
    reviewer_100_summary_df = build_reviewer_100_summary(reviewer_100_results_df)
    software_test_results = _load_packaged_software_test_results()
    questionnaire_template = _load_questionnaire_template_bytes()
    cleaning_df = pd.DataFrame()
    if cleaning_report is not None:
        try:
            cleaning_df = cleaning_report.to_dataframe()
        except Exception:
            cleaning_df = pd.DataFrame()

    leaderboard = getattr(model_output, "leaderboard", pd.DataFrame()) if model_output is not None else pd.DataFrame()
    confusion_df = pd.DataFrame()
    if model_output is not None and getattr(model_output, "best_result", None) is not None:
        conf = getattr(model_output.best_result, "confusion", None)
        if conf is not None:
            confusion_df = pd.DataFrame(conf, index=["Actual negative", "Actual positive"], columns=["Predicted negative", "Predicted positive"]).reset_index(names="actual")
    global_drivers = getattr(explanation_output, "global_importance", pd.DataFrame()) if explanation_output is not None else pd.DataFrame()
    local_drivers = getattr(explanation_output, "local_importance", pd.DataFrame()) if explanation_output is not None else pd.DataFrame()
    memory_df = build_memory_allocation_evidence(
        df=df,
        readiness=readiness,
        model_output=model_output,
        explanation_output=explanation_output,
        chat_df=chat_df,
        audit_df=audit_df,
        perf_df=perf_df,
    )

    # Build a small set of report charts as HTML.
    chart_sections: List[str] = []
    try:
        import plotly.express as px
        missing = df.isna().sum().rename_axis("column").reset_index(name="missing_count")
        missing["missing_percent"] = (missing["missing_count"] / max(len(df), 1) * 100).round(3)
        missing = missing[missing["missing_count"] > 0].sort_values("missing_percent", ascending=False).head(15)
        if not missing.empty:
            chart_sections.append("<h3>Missing-value profile</h3>" + _plotly_html(px.bar(missing, x="column", y="missing_percent", title="Missing values by column (%)")))
        target = getattr(model_output, "target_column", None) if model_output is not None else None
        if target and target in df.columns:
            target_counts = df[target].astype(str).value_counts().rename_axis(target).reset_index(name="count")
            target_counts["percent"] = (target_counts["count"] / target_counts["count"].sum() * 100).round(2)
            chart_sections.append("<h3>Target distribution</h3>" + _plotly_html(px.bar(target_counts, x=target, y="count", hover_data=["percent"], title=f"Distribution of {target}")))
        if leaderboard is not None and not leaderboard.empty:
            metric_cols = [c for c in ["f1", "recall", "precision", "roc_auc", "accuracy"] if c in leaderboard.columns]
            if metric_cols:
                long_lb = leaderboard.melt(id_vars="model", value_vars=metric_cols, var_name="metric", value_name="score")
                chart_sections.append("<h3>Model metric comparison</h3>" + _plotly_html(px.bar(long_lb, x="model", y="score", color="metric", barmode="group", title="Classification metric comparison")))
        if global_drivers is not None and not global_drivers.empty and {"feature", "importance"}.issubset(global_drivers.columns):
            chart_sections.append("<h3>Top global drivers</h3>" + _plotly_html(px.bar(global_drivers.head(15), x="importance", y="feature", orientation="h", title="Top global prediction drivers")))
    except Exception as exc:
        chart_sections.append(f"<p><em>Charts could not be embedded: {exc}</em></p>")

    best_summary = "No model trained."
    if model_output is not None and getattr(model_output, "best_result", None) is not None:
        best = model_output.best_result
        metrics = getattr(best, "metrics", {}) or {}
        best_summary = (
            f"Preferred model under the F1-first selection rule: {best.model_name}. F1={metrics.get('f1', float('nan')):.3f}, "
            f"Recall={metrics.get('recall', float('nan')):.3f}, Precision={metrics.get('precision', float('nan')):.3f}, "
            f"ROC-AUC={metrics.get('roc_auc', float('nan')):.3f}."
        )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Explainable AI Copilot Results Package</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #0f172a; line-height: 1.45; }}
h1, h2, h3 {{ color: #0f172a; }}
.summary {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px; padding: 14px; }}
.warning {{ background: #fef9c3; border: 1px solid #fde68a; border-radius: 10px; padding: 14px; }}
.evidence-table {{ border-collapse: collapse; width: 100%; margin: 12px 0 26px 0; font-size: 14px; }}
.evidence-table th, .evidence-table td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: left; vertical-align: top; }}
.evidence-table th {{ background: #f8fafc; }}
.question {{ border-left: 4px solid #2563eb; padding-left: 12px; margin: 18px 0; }}
pre {{ white-space: pre-wrap; background: #f8fafc; padding: 10px; border-radius: 8px; }}
</style>
</head>
<body>
<h1>Explainable AI Copilot Results Package</h1>
<div class="summary">
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><strong>Dataset:</strong> {dataset_name}</p>
<p><strong>Rows:</strong> {len(df):,} &nbsp; <strong>Columns:</strong> {df.shape[1]:,} &nbsp; <strong>Missing cells:</strong> {int(df.isna().sum().sum()):,} &nbsp; <strong>Duplicate rows:</strong> {int(df.duplicated().sum()):,}</p>
<p><strong>Readiness:</strong> {(readiness or {}).get('overall_status', 'not available')} &nbsp; <strong>Quality score:</strong> {(readiness or {}).get('quality_score', 'not available')}</p>
<p><strong>Model summary:</strong> {best_summary}</p>
</div>
<h2>Data-readiness gates</h2>
{_df_to_html_table(readiness_df)}
<h2>Cleaning summary</h2>
{_df_to_html_table(cleaning_df)}
<h2>Model leaderboard</h2>
{_df_to_html_table(leaderboard)}
<h2>Confusion matrix for preferred F1-selected model</h2>
{_df_to_html_table(confusion_df)}
<h2>Top global explanation drivers</h2>
{_df_to_html_table(global_drivers.head(20) if global_drivers is not None else pd.DataFrame())}
<h2>Local explanation drivers for the currently explained record</h2>
{_df_to_html_table(local_drivers.head(20) if local_drivers is not None else pd.DataFrame())}
<h2>Graphs</h2>
{''.join(chart_sections)}
<h2>Copilot chat evidence</h2>
{''.join('<div class="question"><h3>Question</h3><p>' + html_lib.escape(str(row.question)) + '</p><h3>Answer</h3><pre>' + html_lib.escape(str(row.answer)) + '</pre><p><strong>Safety warning:</strong> ' + html_lib.escape(str(row.safety_warning)) + '</p></div>' for row in chat_df.itertuples()) if not chat_df.empty else '<p><em>No chat answers saved yet.</em></p>'}
<h2>Runtime and process evidence</h2>
{_df_to_html_table(perf_df)}
<h3>Runtime summary</h3>
{_df_to_html_table(runtime_summary_df)}
<h2>Memory allocation evidence</h2>
{_df_to_html_table(memory_df)}
<h2>Copilot evaluation summary (all saved chat)</h2>
{_df_to_html_table(evaluation_summary_df)}
<h2>Official 100-question regression summary</h2>
{_df_to_html_table(reviewer_100_summary_df)}
<h2>Audit/session events</h2>
{_df_to_html_table(audit_df)}
<h2>Packaged software test evidence</h2>
<pre>{html_lib.escape(software_test_results.strip()) if software_test_results else 'No packaged software test report found.'}</pre>
<h2>User-centred evaluation status</h2>
<p>The questionnaire template is included in this ZIP. Participant results are not auto-generated and must come from real voluntary participants before they are reported in Chapter 5.</p>
<div class="warning"><strong>Safety note:</strong> These outputs are decision support only. A human reviewer should check data quality, business context, model limitations and fairness/compliance implications before acting.</div>
</body>
</html>"""

    markdown = make_markdown_report(
        dataset_name,
        df,
        cleaning_report=cleaning_report,
        model_output=model_output,
        explanation_output=explanation_output,
        readiness=readiness,
        chat_history=chat_history,
        performance_events=performance_events,
        audit_events=audit_events,
    )
    metadata = {
        "dataset_name": dataset_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "readiness": readiness or {},
        "best_model": getattr(getattr(model_output, "best_result", None), "model_name", None) if model_output is not None else None,
        "classification_model_selection_rule": "F1-first; ROC-AUC, recall and precision used as tie-breakers",
        "implemented_memory_scope": "short-term session memory",
        "future_optional_memory_scope": "PostgreSQL/ChromaDB/document retrieval, disabled by default",
        "process_rss_mb_at_export": process_rss_mb(),
        "process_peak_rss_mb_at_export": process_peak_rss_mb(),
        "runtime_summary": runtime_summary_df.to_dict("records"),
        "copilot_evaluation_summary": evaluation_summary_df.to_dict("records"),
        "official_100_question_summary": reviewer_100_summary_df.to_dict("records"),
        "packaged_software_test_evidence_available": bool(software_test_results.strip()),
        "user_centred_evaluation_results_status": "NOT_AUTO_GENERATED; collect real participant responses separately",
    }

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", "Open full_results_report.html for the complete evidence report with graphs. The ZIP contains readiness/model/confusion/global-local explanation evidence, Copilot answer/grounding fields, an official 100-question regression summary, runtime/process evidence, memory-allocation evidence, session audit records and the packaged software-test report. Runtime totals measure computation and exclude human idle/typing time. The questionnaire template is included, but real participant results must be collected separately and are never fabricated by the application.\n")
        zf.writestr("full_results_report.html", html.encode("utf-8"))
        zf.writestr("full_results_report.md", markdown.encode("utf-8"))
        zf.writestr("metadata.json", json.dumps(metadata, indent=2, default=str).encode("utf-8"))
        zf.writestr("cleaned_dataset_preview.csv", df.head(5000).to_csv(index=False).encode("utf-8"))
        if not readiness_df.empty:
            zf.writestr("readiness_gates.csv", _safe_to_csv(readiness_df))
        if not cleaning_df.empty:
            zf.writestr("cleaning_report.csv", _safe_to_csv(cleaning_df))
        if leaderboard is not None and not leaderboard.empty:
            zf.writestr("model_leaderboard.csv", _safe_to_csv(leaderboard))
        if not confusion_df.empty:
            zf.writestr("best_model_confusion_matrix.csv", _safe_to_csv(confusion_df))
        if global_drivers is not None and not global_drivers.empty:
            zf.writestr("top_explanation_drivers.csv", _safe_to_csv(global_drivers))
        if local_drivers is not None and not local_drivers.empty:
            zf.writestr("local_prediction_explanation_drivers.csv", _safe_to_csv(local_drivers))
        if not chat_df.empty:
            zf.writestr("copilot_chat_answers.csv", _safe_to_csv(chat_df))
        if not audit_df.empty:
            zf.writestr("session_audit_log.csv", _safe_to_csv(audit_df))
        if not perf_df.empty:
            zf.writestr("runtime_process_evidence.csv", _safe_to_csv(perf_df))
        zf.writestr("runtime_summary.csv", _safe_to_csv(runtime_summary_df))
        zf.writestr("memory_allocation_evidence.csv", _safe_to_csv(memory_df))
        zf.writestr("copilot_evaluation_summary.csv", _safe_to_csv(evaluation_summary_df))
        zf.writestr("copilot_100_question_results.csv", _safe_to_csv(reviewer_100_results_df))
        zf.writestr("copilot_100_question_summary.csv", _safe_to_csv(reviewer_100_summary_df))
        if software_test_results:
            zf.writestr("software_test_results.txt", software_test_results.encode("utf-8"))
        if questionnaire_template:
            zf.writestr("user_evaluation_questionnaire_template.csv", questionnaire_template)
        zf.writestr(
            "user_evaluation_status.txt",
            (
                "Participant results are not auto-generated. Use the included questionnaire template with real voluntary participants, "
                "then analyse and report those genuine responses separately for the dissertation user-centred evaluation.\n"
            ).encode("utf-8"),
        )
    return buffer.getvalue()
