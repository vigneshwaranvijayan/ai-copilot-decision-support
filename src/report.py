"""Simple report export utilities."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd


def dataframe_to_markdown_safe(df: pd.DataFrame, index: bool = False) -> str:
    """Return a markdown table without requiring the optional tabulate package.

    Pandas DataFrame.to_markdown depends on tabulate. In student/local
    environments tabulate is sometimes missing, so this fallback keeps the
    Streamlit report export working instead of crashing.
    """
    if df is None or df.empty:
        return "_No rows available._"
    table = df.copy()
    if index:
        table = table.reset_index()
    table = table.astype(object).where(pd.notna(table), "")

    def clean_cell(value: object) -> str:
        text = str(value).replace("\n", " ").replace("|", "\\|")
        return text

    headers = [clean_cell(c) for c in table.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(clean_cell(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def make_markdown_report(
    dataset_name: str,
    df: pd.DataFrame,
    cleaning_report: Optional[object] = None,
    model_output: Optional[object] = None,
    explanation_output: Optional[object] = None,
) -> str:
    lines = [
        f"# Explainable AI Copilot Dataset Report",
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
            f"- Best model: `{best.model_name}`",
            f"- F1: {best.metrics.get('f1', float('nan')):.3f}",
            f"- Recall: {best.metrics.get('recall', float('nan')):.3f}",
            f"- Precision: {best.metrics.get('precision', float('nan')):.3f}",
            f"- ROC-AUC: {best.metrics.get('roc_auc', float('nan')):.3f}",
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
        lines.append(dataframe_to_markdown_safe(explanation_output.global_importance.head(10), index=False))
        lines.append("")
    lines.extend([
        "## Safety note",
        "",
        "The outputs are decision support only. A human reviewer should verify the context, data quality and business implications before taking action.",
    ])
    return "\n".join(lines)
