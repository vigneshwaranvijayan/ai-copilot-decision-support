"""Modern visual analytics helpers for the Explainable AI Copilot.

The goal of this module is not to reproduce a full BI platform such as
Power BI or Tableau. It creates presentation-ready, dataset-grounded charts
that support the dissertation contribution: explainable decision support from
uploaded structured data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .business_insights import detect_business_domain
from .eda import missing_value_table


@dataclass
class VisualCard:
    label: str
    value: str
    note: str = ""


@dataclass
class VisualFigure:
    title: str
    description: str
    fig: Any
    evidence: Optional[pd.DataFrame] = None


def _normalise(text: object) -> str:
    return str(text).lower().replace("_", "").replace("-", "").replace(" ", "")


def _find_column(df: pd.DataFrame, keywords: List[str], prefer_numeric: bool | None = None) -> Optional[str]:
    """Find a column using robust keyword matching."""
    norm_keywords = [_normalise(k) for k in keywords]
    candidates: List[Tuple[int, str]] = []
    for col in df.columns:
        norm_col = _normalise(col)
        if prefer_numeric is True and not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if prefer_numeric is False and pd.api.types.is_numeric_dtype(df[col]):
            continue
        score = 0
        for kw in norm_keywords:
            if norm_col == kw:
                score += 5
            elif kw in norm_col:
                score += 3
        if score:
            candidates.append((score, col))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def _date_columns(df: pd.DataFrame) -> List[str]:
    cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if cols:
        return cols
    possible = []
    for col in df.columns:
        norm = _normalise(col)
        if any(k in norm for k in ["date", "time", "month", "year", "invoice"]):
            converted = pd.to_datetime(df[col], errors="coerce")
            if converted.notna().mean() > 0.50:
                possible.append(col)
    return possible


def _categorical_columns(df: pd.DataFrame, max_unique_ratio: float = 0.50) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        unique = df[col].nunique(dropna=True)
        if unique >= 2 and unique <= max(30, int(len(df) * max_unique_ratio)):
            cols.append(col)
    return cols


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def _safe_sample(df: pd.DataFrame, max_rows: int = 50000) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=42)


def data_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a transparent quality score based on missingness and duplicates."""
    total_cells = max(df.shape[0] * df.shape[1], 1)
    missing_pct = float(df.isna().sum().sum() / total_cells * 100)
    duplicate_pct = float(df.duplicated().mean() * 100) if len(df) else 0.0
    empty_cols_pct = float(((df.isna().mean() == 1.0).sum() / max(df.shape[1], 1)) * 100)
    score = 100 - (missing_pct * 0.6) - (duplicate_pct * 0.3) - (empty_cols_pct * 0.1)
    score = max(0.0, min(100.0, score))
    if score >= 85:
        band = "Strong"
    elif score >= 70:
        band = "Acceptable"
    elif score >= 50:
        band = "Needs cleaning"
    else:
        band = "Weak"
    return {
        "score": round(score, 1),
        "band": band,
        "missing_percent": round(missing_pct, 2),
        "duplicate_percent": round(duplicate_pct, 2),
        "empty_columns_percent": round(empty_cols_pct, 2),
    }


def visual_kpi_cards(df: pd.DataFrame, domain: str, target_col: Optional[str] = None) -> List[VisualCard]:
    q = data_quality_score(df)
    cards = [
        VisualCard("Rows", f"{len(df):,}", "Records analysed"),
        VisualCard("Columns", f"{len(df.columns):,}", "Available fields"),
        VisualCard("Data quality", f"{q['score']}%", q["band"]),
        VisualCard("Missing cells", f"{int(df.isna().sum().sum()):,}", f"{q['missing_percent']}% of all cells"),
        VisualCard("Duplicate rows", f"{int(df.duplicated().sum()):,}", f"{q['duplicate_percent']}% of rows"),
        VisualCard("Detected domain", domain.replace("_", " ").title(), "Used for chart recommendations"),
    ]
    if target_col and target_col in df.columns:
        cards.append(VisualCard("Target column", str(target_col), f"{df[target_col].nunique(dropna=True)} distinct values"))
    return cards


def plot_quality_gauge(df: pd.DataFrame):
    q = data_quality_score(df)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=q["score"],
        number={"suffix": "%"},
        title={"text": f"Data quality score: {q['band']}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.22},
            "steps": [
                {"range": [0, 50], "color": "rgba(239, 68, 68, 0.18)"},
                {"range": [50, 70], "color": "rgba(245, 158, 11, 0.18)"},
                {"range": [70, 85], "color": "rgba(59, 130, 246, 0.18)"},
                {"range": [85, 100], "color": "rgba(16, 185, 129, 0.18)"},
            ],
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_missing_profile(df: pd.DataFrame):
    miss = missing_value_table(df)
    miss = miss[miss["missing_count"] > 0].head(20)
    if miss.empty:
        miss = pd.DataFrame({"column": ["No missing values"], "missing_percent": [0], "missing_count": [0]})
    fig = px.bar(
        miss.sort_values("missing_percent"),
        x="missing_percent",
        y="column",
        orientation="h",
        hover_data=["missing_count"],
        title="Missing-value profile",
        labels={"missing_percent": "Missing (%)", "column": "Column"},
    )
    return fig


def plot_dtype_mix(df: pd.DataFrame):
    rows = []
    for dtype_name, count in df.dtypes.astype(str).value_counts().items():
        rows.append({"dtype": dtype_name, "columns": int(count)})
    data = pd.DataFrame(rows)
    if data.empty:
        return None
    return px.pie(data, names="dtype", values="columns", hole=0.52, title="Column type mix")


def plot_top_category(df: pd.DataFrame, col: str, top_n: int = 12):
    if col not in df.columns:
        return None
    counts = df[col].astype("string").fillna("Missing").value_counts().head(top_n).reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(
        counts.sort_values("count"),
        x="count",
        y=col,
        orientation="h",
        title=f"Top values in {col}",
        labels={"count": "Record count", col: col},
        hover_data=["count"],
    )
    return fig


def plot_numeric_distribution(df: pd.DataFrame, col: str):
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return None
    return px.histogram(_safe_sample(df), x=col, marginal="box", nbins=40, title=f"Distribution of {col}")


def plot_target_distribution(df: pd.DataFrame, target_col: str):
    if not target_col or target_col not in df.columns:
        return None
    data = df[target_col].astype("string").fillna("Missing").value_counts().reset_index()
    data.columns = [target_col, "count"]
    if data.empty:
        return None
    return px.pie(data, names=target_col, values="count", hole=0.48, title=f"Target distribution: {target_col}")


def plot_target_rate_by_category(df: pd.DataFrame, target_col: str, category_col: str, positive_label: Optional[object] = None, top_n: int = 12):
    if not target_col or target_col not in df.columns or category_col not in df.columns:
        return None, pd.DataFrame()
    temp = df[[target_col, category_col]].copy()
    if positive_label is None:
        vc = temp[target_col].dropna().astype(str).value_counts()
        if vc.empty:
            return None, pd.DataFrame()
        # choose minority as likely positive when no explicit label is available
        positive_label = vc.index[-1]
    temp["positive"] = (temp[target_col].astype(str) == str(positive_label)).astype(int)
    grouped = temp.groupby(category_col, dropna=False)["positive"].agg(["mean", "count"]).reset_index()
    grouped["positive_rate_percent"] = (grouped["mean"] * 100).round(2)
    grouped = grouped[grouped["count"] >= max(3, int(len(df) * 0.005))]
    grouped = grouped.sort_values("positive_rate_percent", ascending=False).head(top_n)
    if grouped.empty:
        return None, grouped
    fig = px.bar(
        grouped.sort_values("positive_rate_percent"),
        x="positive_rate_percent",
        y=category_col,
        orientation="h",
        hover_data=["count"],
        title=f"{target_col} positive rate by {category_col}",
        labels={"positive_rate_percent": "Positive rate (%)", category_col: category_col},
    )
    return fig, grouped


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return None
    # prefer lower-cardinality and important-looking numeric fields to keep chart readable
    numeric = numeric.iloc[:, :25]
    corr = numeric.corr(numeric_only=True).round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Numeric correlation heatmap")
    return fig


def plot_time_trend(df: pd.DataFrame, date_col: str, value_col: Optional[str] = None):
    if not date_col or date_col not in df.columns:
        return None, pd.DataFrame()
    temp = df[[date_col] + ([value_col] if value_col and value_col in df.columns else [])].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])
    if temp.empty:
        return None, pd.DataFrame()
    temp["period"] = temp[date_col].dt.to_period("M").dt.to_timestamp()
    if value_col and value_col in temp.columns and pd.api.types.is_numeric_dtype(temp[value_col]):
        trend = temp.groupby("period", as_index=False)[value_col].sum()
        fig = px.line(trend, x="period", y=value_col, markers=True, title=f"Monthly trend of {value_col}")
    else:
        trend = temp.groupby("period").size().reset_index(name="records")
        fig = px.line(trend, x="period", y="records", markers=True, title="Monthly record volume trend")
    return fig, trend


def plot_top_numeric_by_category(df: pd.DataFrame, category_col: str, value_col: str, agg: str = "sum", top_n: int = 12):
    if category_col not in df.columns or value_col not in df.columns or not pd.api.types.is_numeric_dtype(df[value_col]):
        return None, pd.DataFrame()
    if agg not in {"sum", "mean", "median", "count"}:
        agg = "sum"
    data = getattr(df.groupby(category_col, dropna=False)[value_col], agg)().reset_index(name=f"{agg}_{value_col}")
    value_name = f"{agg}_{value_col}"
    data = data.sort_values(value_name, ascending=False).head(top_n)
    if data.empty:
        return None, data
    fig = px.bar(
        data.sort_values(value_name),
        x=value_name,
        y=category_col,
        orientation="h",
        title=f"Top {category_col} by {agg} {value_col}",
        labels={value_name: f"{agg.title()} {value_col}", category_col: category_col},
    )
    return fig, data


def plot_feedback_themes(df: pd.DataFrame, text_col: str):
    if not text_col or text_col not in df.columns:
        return None, pd.DataFrame()
    theme_keywords = {
        "Pricing / charges": ["price", "pricing", "charge", "charges", "expensive", "cost", "bill", "billing"],
        "Contract / cancellation": ["contract", "cancel", "cancelled", "canceled", "leave", "leaving", "switch", "churn"],
        "Service reliability": ["internet", "connection", "speed", "slow", "outage", "disconnect", "unreliable", "service"],
        "Support experience": ["support", "help", "agent", "call", "complaint", "delay", "rude", "response"],
        "Onboarding": ["new", "started", "joined", "first", "setup", "onboarding"],
        "Value / package fit": ["value", "package", "plan", "offer", "benefit", "feature"],
    }
    text = df[text_col].fillna("").astype(str).str.lower()
    rows = []
    for theme, keywords in theme_keywords.items():
        pattern = r"\b(?:" + "|".join([kw.replace(" ", r"\s+") for kw in keywords]) + r")\b"
        mask = text.str.contains(pattern, regex=True, na=False)
        rows.append({"theme": theme, "matched_rows": int(mask.sum()), "percent_rows": round(float(mask.mean() * 100), 2)})
    data = pd.DataFrame(rows).sort_values("matched_rows", ascending=False)
    fig = px.bar(
        data.sort_values("matched_rows"),
        x="matched_rows",
        y="theme",
        orientation="h",
        hover_data=["percent_rows"],
        title=f"Feedback themes detected in {text_col}",
        labels={"matched_rows": "Rows matched", "theme": "Theme"},
    )
    return fig, data


def build_visual_dashboard(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    target_col: Optional[str] = None,
    positive_label: Optional[object] = None,
    max_figures: int = 8,
) -> Dict[str, Any]:
    """Build a curated visual dashboard for the active dataset."""
    domain = detect_business_domain(df)
    figures: List[VisualFigure] = []

    figures.append(VisualFigure(
        "Data quality score",
        "Shows whether the active dataset is clean enough for reliable analysis.",
        plot_quality_gauge(df),
    ))
    figures.append(VisualFigure(
        "Missing-value profile",
        "Highlights the columns that need data quality review before modelling or decision-making.",
        plot_missing_profile(df),
        missing_value_table(df).head(20),
    ))
    dtype_fig = plot_dtype_mix(df)
    if dtype_fig:
        figures.append(VisualFigure("Column type mix", "Shows whether the dataset is mainly numeric, categorical, dates or mixed fields.", dtype_fig))

    if target_col and target_col in df.columns:
        tfig = plot_target_distribution(df, target_col)
        if tfig:
            figures.append(VisualFigure("Target distribution", "Checks class balance before model training and evaluation.", tfig))

    numeric_cols = _numeric_columns(df)
    categorical_cols = _categorical_columns(df)
    date_cols = _date_columns(df)

    # Domain-specific visuals first, then generic fallback.
    norm_cols = {_normalise(c): c for c in df.columns}

    if domain in {"customer_churn", "customer_feedback"}:
        for keywords in [["contract"], ["paymentmethod", "payment"], ["internetservice", "internet"], ["techsupport", "support"]]:
            cat = _find_column(df, keywords, prefer_numeric=False)
            if cat:
                if target_col:
                    fig, evid = plot_target_rate_by_category(df, target_col, cat, positive_label)
                    if fig:
                        figures.append(VisualFigure(f"Risk rate by {cat}", f"Compares the positive target rate across {cat} groups.", fig, evid))
                else:
                    fig = plot_top_category(df, cat)
                    if fig:
                        figures.append(VisualFigure(f"{cat} distribution", f"Shows the most common values in {cat}.", fig))
        text_col = _find_column(df, ["customerfeedback", "feedback", "comment", "review"], prefer_numeric=False)
        if text_col:
            fig, evid = plot_feedback_themes(df, text_col)
            if fig:
                figures.append(VisualFigure("Customer feedback themes", "Detects business themes from uploaded feedback text.", fig, evid))

    elif domain == "sales_retail":
        value_col = _find_column(df, ["sales", "revenue", "amount", "price", "total", "quantity", "profit"], prefer_numeric=True)
        product_col = _find_column(df, ["product", "stockcode", "description", "item", "sku"], prefer_numeric=False)
        date_col = date_cols[0] if date_cols else None
        if date_col:
            fig, evid = plot_time_trend(df, date_col, value_col)
            if fig:
                figures.append(VisualFigure("Sales/record trend", "Shows movement over time using monthly aggregation for scale-readiness.", fig, evid.head(24)))
        if product_col and value_col:
            fig, evid = plot_top_numeric_by_category(df, product_col, value_col, agg="sum")
            if fig:
                figures.append(VisualFigure("Top products/items", "Ranks products/items by uploaded sales, amount, revenue, quantity or profit.", fig, evid))
        elif product_col:
            fig = plot_top_category(df, product_col)
            if fig:
                figures.append(VisualFigure("Top products/items", "Ranks products/items by record count.", fig))

    elif domain == "bank_marketing":
        for cat in ["job", "education", "marital", "contact", "poutcome"]:
            col = _find_column(df, [cat], prefer_numeric=False)
            if col and target_col:
                fig, evid = plot_target_rate_by_category(df, target_col, col, positive_label)
                if fig:
                    figures.append(VisualFigure(f"Campaign response by {col}", f"Shows which {col} groups have higher positive response rates.", fig, evid))
            elif col:
                fig = plot_top_category(df, col)
                if fig:
                    figures.append(VisualFigure(f"{col} distribution", f"Shows distribution of {col}.", fig))

    elif domain == "employee_hr":
        for cat in ["department", "jobrole", "role", "overtime", "educationfield", "maritalstatus"]:
            col = _find_column(df, [cat], prefer_numeric=False)
            if col and target_col:
                fig, evid = plot_target_rate_by_category(df, target_col, col, positive_label)
                if fig:
                    figures.append(VisualFigure(f"HR risk by {col}", f"Compares attrition or risk rate across {col} groups.", fig, evid))
            elif col:
                fig = plot_top_category(df, col)
                if fig:
                    figures.append(VisualFigure(f"{col} distribution", f"Shows distribution of {col}.", fig))
        salary = _find_column(df, ["salary", "income", "monthlyincome", "wage", "pay"], prefer_numeric=True)
        performance = _find_column(df, ["performance", "rating", "satisfaction", "score"], prefer_numeric=True)
        if salary and performance:
            sample = _safe_sample(df)
            fig = px.scatter(sample, x=salary, y=performance, opacity=0.6, title=f"{salary} vs {performance}")
            figures.append(VisualFigure("Salary/performance relationship", "Explores whether pay and performance indicators move together.", fig))

    elif domain == "operations_service":
        for cat in ["category", "status", "type", "priority", "department", "location", "area"]:
            col = _find_column(df, [cat], prefer_numeric=False)
            if col:
                fig = plot_top_category(df, col)
                if fig:
                    figures.append(VisualFigure(f"Operational volume by {col}", f"Shows which {col} values appear most often.", fig))
        date_col = date_cols[0] if date_cols else None
        if date_col:
            fig, evid = plot_time_trend(df, date_col, None)
            if fig:
                figures.append(VisualFigure("Operational demand trend", "Shows monthly volume patterns for operational planning.", fig, evid.head(24)))

    # Generic visuals ensure every dataset gets useful charts.
    if categorical_cols:
        fig = plot_top_category(df, categorical_cols[0])
        if fig:
            figures.append(VisualFigure("Top category distribution", f"Shows the most common values in {categorical_cols[0]}.", fig))
    if numeric_cols:
        fig = plot_numeric_distribution(df, numeric_cols[0])
        if fig:
            figures.append(VisualFigure("Numeric distribution", f"Shows spread and outliers for {numeric_cols[0]}.", fig))
    corr_fig = plot_correlation_heatmap(df)
    if corr_fig:
        figures.append(VisualFigure("Correlation heatmap", "Highlights relationships between numeric columns.", corr_fig))

    # De-duplicate figures by title while keeping order.
    seen = set()
    unique_figures = []
    for vf in figures:
        if vf.fig is None or vf.title in seen:
            continue
        seen.add(vf.title)
        unique_figures.append(vf)

    return {
        "dataset_name": dataset_name,
        "domain": domain,
        "quality": data_quality_score(df),
        "cards": visual_kpi_cards(df, domain, target_col),
        "figures": unique_figures[:max_figures],
        "large_dataset_note": prepare_large_dataset_note(df),
    }


def prepare_large_dataset_note(df: pd.DataFrame) -> str:
    rows = len(df)
    if rows >= 1_000_000:
        return "Very large dataset: charts should use aggregation or sampling locally; production should use a database/query engine and cached visual aggregates."
    if rows >= 100_000:
        return "Large dataset: charts use aggregated views or sampling to remain responsive in Streamlit."
    return "Small/medium dataset: full local visual workflow is suitable for the proof-of-concept demo."
