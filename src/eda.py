"""Exploratory data analysis helpers."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def overview_metrics(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "Rows": int(df.shape[0]),
        "Columns": int(df.shape[1]),
        "Missing cells": int(df.isna().sum().sum()),
        "Duplicate rows": int(df.duplicated().sum()),
        "Numeric columns": int(df.select_dtypes(include=np.number).shape[1]),
        "Categorical/text columns": int(df.select_dtypes(include=["object", "category", "string"]).shape[1]),
        "Date/time columns": int(df.select_dtypes(include=["datetime", "datetimetz"]).shape[1]),
    }


def missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percent": (df.isna().mean().values * 100).round(2),
        "dtype": [str(t) for t in df.dtypes],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    return table.sort_values(["missing_count", "unique_values"], ascending=[False, False]).reset_index(drop=True)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number)
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T.reset_index().rename(columns={"index": "column"})


def categorical_summary(df: pd.DataFrame, max_unique: int = 30) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(include=["object", "category", "string", "bool"]).columns:
        nunique = int(df[col].nunique(dropna=True))
        top = df[col].mode(dropna=True)
        rows.append({
            "column": col,
            "unique_values": nunique,
            "top_value": top.iloc[0] if not top.empty else None,
            "top_frequency": int((df[col] == top.iloc[0]).sum()) if not top.empty else 0,
        })
    return pd.DataFrame(rows).sort_values("unique_values").reset_index(drop=True) if rows else pd.DataFrame()


def target_distribution(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        return pd.DataFrame()
    counts = df[target].value_counts(dropna=False).rename_axis(target).reset_index(name="count")
    counts["percent"] = (counts["count"] / max(counts["count"].sum(), 1) * 100).round(2)
    return counts


def figure_missing(df: pd.DataFrame):
    miss = missing_value_table(df)
    miss = miss[miss["missing_count"] > 0].head(30)
    if miss.empty:
        return None
    return px.bar(miss, x="column", y="missing_percent", title="Missing Values by Column (%)")


def figure_numeric_distribution(df: pd.DataFrame, column: str):
    return px.histogram(df, x=column, marginal="box", title=f"Distribution of {column}")


def figure_categorical_counts(df: pd.DataFrame, column: str, top_n: int = 20):
    counts = df[column].astype("string").fillna("Missing").value_counts().head(top_n).reset_index()
    counts.columns = [column, "count"]
    return px.bar(counts, x=column, y="count", title=f"Top {top_n} values in {column}")


def figure_correlation(df: pd.DataFrame, max_cols: int = 25):
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return None
    numeric = numeric.iloc[:, :max_cols]
    corr = numeric.corr(numeric_only=True).round(2)
    return px.imshow(corr, text_auto=True, aspect="auto", title="Numeric Correlation Heatmap")


def figure_target_by_category(df: pd.DataFrame, target: str, category: str, positive_value: Optional[object] = None, top_n: int = 15):
    if target not in df.columns or category not in df.columns:
        return None
    temp = df[[target, category]].copy()
    if positive_value is None:
        positive_value = temp[target].dropna().value_counts().index[0] if not temp[target].dropna().empty else None
    if positive_value is None:
        return None
    temp["target_positive"] = (temp[target].astype(str) == str(positive_value)).astype(int)
    grouped = temp.groupby(category, dropna=False)["target_positive"].agg(["mean", "count"]).reset_index()
    grouped["rate_percent"] = (grouped["mean"] * 100).round(2)
    grouped = grouped.sort_values("count", ascending=False).head(top_n)
    return px.bar(grouped, x=category, y="rate_percent", hover_data=["count"], title=f"{target} positive rate by {category}")


def figure_target_by_numeric(df: pd.DataFrame, target: str, numeric_col: str, positive_value: Optional[object] = None):
    if target not in df.columns or numeric_col not in df.columns:
        return None
    return px.box(df, x=target, y=numeric_col, title=f"{numeric_col} by {target}")


def figure_date_trend(df: pd.DataFrame, date_col: str, value_col: Optional[str] = None):
    if date_col not in df.columns:
        return None
    temp = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp[date_col]):
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])
    if temp.empty:
        return None
    temp["period"] = temp[date_col].dt.to_period("M").dt.to_timestamp()
    if value_col and value_col in temp.columns and pd.api.types.is_numeric_dtype(temp[value_col]):
        trend = temp.groupby("period")[value_col].sum().reset_index()
        return px.line(trend, x="period", y=value_col, markers=True, title=f"Monthly trend of {value_col}")
    trend = temp.groupby("period").size().reset_index(name="records")
    return px.line(trend, x="period", y="records", markers=True, title="Monthly record count trend")


def top_numeric_by_category(df: pd.DataFrame, category: str, numeric_col: str, agg: str = "sum", top_n: int = 15) -> pd.DataFrame:
    if category not in df.columns or numeric_col not in df.columns:
        return pd.DataFrame()
    if not pd.api.types.is_numeric_dtype(df[numeric_col]):
        return pd.DataFrame()
    if agg not in {"sum", "mean", "median", "count"}:
        agg = "sum"
    grouped = getattr(df.groupby(category, dropna=False)[numeric_col], agg)().reset_index(name=f"{agg}_{numeric_col}")
    return grouped.sort_values(f"{agg}_{numeric_col}", ascending=False).head(top_n)


def text_column_summary(df: pd.DataFrame, column: str, top_n: int = 20) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame()
    stopwords = {
        "the", "and", "for", "with", "this", "that", "from", "have", "has", "was", "were", "are", "not", "but", "you",
        "customer", "service", "very", "much", "will", "can", "our", "their", "they", "your", "been", "about", "because",
    }
    text = df[column].dropna().astype(str).str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True)
    words = text.str.split().explode()
    words = words[words.str.len() > 2]
    words = words[~words.isin(stopwords)]
    if words.empty:
        return pd.DataFrame()
    return words.value_counts().head(top_n).rename_axis("word").reset_index(name="count")


def simple_sentiment_summary(df: pd.DataFrame, column: str) -> Dict[str, int]:
    positive = {"good", "great", "excellent", "happy", "satisfied", "helpful", "fast", "easy", "love", "clear", "reliable"}
    negative = {"bad", "poor", "angry", "slow", "expensive", "problem", "issue", "cancel", "complaint", "unhappy", "difficult"}
    text = df[column].dropna().astype(str).str.lower()
    pos_count = int(text.apply(lambda s: any(w in s for w in positive)).sum())
    neg_count = int(text.apply(lambda s: any(w in s for w in negative)).sum())
    return {"positive_keyword_rows": pos_count, "negative_keyword_rows": neg_count, "analysed_rows": int(text.shape[0])}
