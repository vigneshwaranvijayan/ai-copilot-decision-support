from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px

import io
import json
from urllib.parse import urlparse

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls", "json"]


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _normalise_loaded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean DataFrame and remove fully empty rows/columns."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Loaded content is not a valid table.")
    df = _clean_columns(df)
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    if df.empty:
        raise ValueError("The uploaded file loaded successfully but contains no usable rows.")
    return df


def load_tabular_file(uploaded_file) -> pd.DataFrame:
    """
    Load CSV, Excel or JSON into a pandas DataFrame.

    This is a robustness feature. The dissertation's main evaluated case study remains customer churn.
    """
    name = getattr(uploaded_file, "name", "uploaded_file").lower()

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin1")
        return _normalise_loaded_dataframe(df)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        # Loads the first sheet by default to keep the prototype simple.
        df = pd.read_excel(uploaded_file, sheet_name=0)
        return _normalise_loaded_dataframe(df)

    if name.endswith(".json"):
        raw = uploaded_file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        data = json.loads(raw)

        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            # Common API shape: {"data": [...]} or {"records": [...]}.
            list_value = None
            for key in ["data", "records", "items", "results"]:
                if isinstance(data.get(key), list):
                    list_value = data[key]
                    break
            df = pd.json_normalize(list_value if list_value is not None else data)
        else:
            raise ValueError("JSON must contain an object or list of records.")
        return _normalise_loaded_dataframe(df)

    raise ValueError("Unsupported file type. Please upload CSV, Excel (.xlsx/.xls), or JSON.")


def load_from_api_url(url: str) -> pd.DataFrame:
    """
    Load tabular data from a public API/URL returning JSON or CSV.

    For safety and scope control, this proof-of-concept supports public, no-auth URLs only.
    """
    if requests is None:
        raise ValueError("The requests package is not installed. Run: python -m pip install -r requirements.txt")

    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Please enter a valid http or https URL.")

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    text = response.text

    if "json" in content_type or url.lower().endswith(".json"):
        data = response.json()
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            list_value = None
            for key in ["data", "records", "items", "results"]:
                if isinstance(data.get(key), list):
                    list_value = data[key]
                    break
            df = pd.json_normalize(list_value if list_value is not None else data)
        else:
            raise ValueError("API JSON must contain an object or list of records.")
        return _normalise_loaded_dataframe(df)

    # Treat as CSV/plain text if not JSON.
    df = pd.read_csv(io.StringIO(text))
    return _normalise_loaded_dataframe(df)


# Backward-compatible alias used by older app versions/tests.
def load_csv(uploaded_file) -> pd.DataFrame:
    return load_tabular_file(uploaded_file)


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_columns: List[str]
    target_col: str
    positive_label: Any


# load_csv is defined above as a backward-compatible alias for load_tabular_file.


def detect_target_column(df: pd.DataFrame, expected_target: str) -> str | None:
    if expected_target in df.columns:
        return expected_target
    lower_map = {c.lower(): c for c in df.columns}
    return lower_map.get(expected_target.lower())


def is_binary_target(series: pd.Series) -> bool:
    return series.dropna().astype(str).nunique() == 2


def likely_id_columns(df: pd.DataFrame) -> List[str]:
    id_like = []
    for col in df.columns:
        name = str(col).lower()
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
        if "id" in name or unique_ratio > 0.95:
            id_like.append(col)
    return id_like


def data_quality_report(df: pd.DataFrame, target_col: str | None = None) -> Dict[str, Any]:
    missing_by_column = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
    )
    missing_by_column["missing_percent"] = (
        missing_by_column["missing_count"] / max(len(df), 1) * 100
    ).round(2)

    dtypes = (
        df.dtypes.astype(str)
        .reset_index()
        .rename(columns={"index": "column", 0: "dtype"})
    )

    target_distribution = pd.DataFrame()
    if target_col and target_col in df.columns:
        target_distribution = (
            df[target_col].astype(str).value_counts(dropna=False)
            .reset_index()
        )
        target_distribution.columns = [target_col, "count"]

    return {
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": int(df.select_dtypes(include=np.number).shape[1]),
        "categorical_columns": int(df.select_dtypes(exclude=np.number).shape[1]),
        "id_like_columns": likely_id_columns(df),
        "missing_by_column": missing_by_column,
        "dtypes": dtypes,
        "target_distribution": target_distribution,
    }


def positive_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["yes", "1", "true", "y", "churn", "attrition"])


def _normalise_binary_target(y: pd.Series) -> Tuple[pd.Series, Any]:
    yes_like = {"yes", "true", "1", "y", "churn", "attrition", "positive"}
    no_like = {"no", "false", "0", "n", "not churn", "non-churn", "negative"}

    def convert_value(v):
        s = str(v).strip().lower()
        if s in yes_like:
            return 1
        if s in no_like:
            return 0
        return v

    converted = y.apply(convert_value)

    if converted.dropna().nunique() != 2:
        raise ValueError("The selected target column must contain exactly two classes for this proof-of-concept classifier.")

    if not pd.api.types.is_numeric_dtype(converted):
        codes, uniques = pd.factorize(converted)
        converted = pd.Series(codes, index=y.index)
        positive_label = uniques[1] if len(uniques) > 1 else uniques[0]
    else:
        unique_vals = sorted(converted.dropna().unique())
        if set(unique_vals).issubset({0, 1}):
            positive_label = 1
            converted = converted.astype(int)
        else:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            converted = converted.map(mapping).astype(int)
            positive_label = unique_vals[1]

    return converted.astype(int), positive_label


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    drop_columns: List[str] | None = None,
) -> PreparedData:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found.")
    if not is_binary_target(df[target_col]):
        raise ValueError("The selected target column is not binary. Please choose a Yes/No or 0/1 style target for prediction.")

    drop_columns = drop_columns or []
    work_df = df.copy()

    # Convert object numeric-looking columns.
    for col in work_df.columns:
        if work_df[col].dtype == "object":
            converted = pd.to_numeric(work_df[col], errors="coerce")
            # Convert only if most non-null values look numeric.
            valid_ratio = converted.notna().mean()
            if valid_ratio > 0.80:
                work_df[col] = converted

    y_raw = work_df[target_col]
    y, positive_label = _normalise_binary_target(y_raw)

    feature_df = work_df.drop(columns=[target_col], errors="ignore")
    feature_df = feature_df.drop(
        columns=[c for c in drop_columns if c in feature_df.columns],
        errors="ignore",
    )

    # Remove columns that look like unique IDs.
    id_like_cols = []
    for col in feature_df.columns:
        unique_ratio = feature_df[col].nunique(dropna=True) / max(len(feature_df), 1)
        if unique_ratio > 0.95 and (feature_df[col].dtype == "object" or "id" in str(col).lower()):
            id_like_cols.append(col)
    feature_df = feature_df.drop(columns=id_like_cols, errors="ignore")

    if feature_df.shape[1] == 0:
        raise ValueError("No usable feature columns remain after removing target and ID-like columns.")

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() == 2 and y.value_counts().min() >= 2 else None,
    )

    numeric_features = list(X_train.select_dtypes(include=np.number).columns)
    categorical_features = list(X_train.select_dtypes(exclude=np.number).columns)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_columns=list(feature_df.columns),
        target_col=target_col,
        positive_label=positive_label,
    )


def grouped_positive_rate(df: pd.DataFrame, target_col: str, group_col: str) -> pd.DataFrame:
    temp = df.copy()
    if group_col not in temp.columns or target_col not in temp.columns:
        return pd.DataFrame()

    if pd.api.types.is_numeric_dtype(temp[group_col]):
        try:
            temp[f"{group_col}_group"] = pd.qcut(temp[group_col], q=4, duplicates="drop").astype(str)
            group_col = f"{group_col}_group"
        except Exception:
            temp[f"{group_col}_group"] = temp[group_col].astype(str)
            group_col = f"{group_col}_group"

    grouped = (
        temp.groupby(group_col)[target_col]
        .apply(lambda x: positive_mask(x).mean() * 100 if x.astype(str).str.lower().isin(["yes", "no", "1", "0", "true", "false"]).any() else (x.astype(str).value_counts(normalize=True).max() * 100))
        .reset_index()
        .rename(columns={target_col: "positive_rate"})
        .sort_values("positive_rate", ascending=False)
    )
    return grouped


def create_generic_charts(df: pd.DataFrame):
    charts = {}
    summary_points = [
        f"The dataset contains {len(df)} records and {df.shape[1]} columns.",
        f"It has {df.select_dtypes(include=np.number).shape[1]} numeric columns and {df.select_dtypes(exclude=np.number).shape[1]} categorical/text columns.",
        f"It contains {int(df.isna().sum().sum())} missing cells and {int(df.duplicated().sum())} duplicate rows.",
    ]

    # First categorical distribution.
    cat_cols = [c for c in df.select_dtypes(exclude=np.number).columns if df[c].nunique(dropna=True) <= 30]
    if cat_cols:
        col = cat_cols[0]
        counts = df[col].astype(str).value_counts().head(15).reset_index()
        counts.columns = [col, "count"]
        charts["categorical_distribution"] = px.bar(counts, x=col, y="count", title=f"Distribution of {col}", text="count")
        summary_points.append(f"The most common {col} value is {counts.iloc[0][col]} ({counts.iloc[0]['count']} records).")

    # First numeric distribution.
    num_cols = list(df.select_dtypes(include=np.number).columns)
    if num_cols:
        col = num_cols[0]
        charts["numeric_distribution"] = px.histogram(df, x=col, title=f"Distribution of {col}")
        summary_points.append(f"The average {col} is approximately {df[col].mean():.2f}.")

    return summary_points, charts


def create_dashboard_metrics(df: pd.DataFrame, target_col: str | None = None):
    charts = {}

    # Generic metrics first.
    metrics = {
        "total_records": len(df),
        "total_columns": df.shape[1],
        "numeric_columns": int(df.select_dtypes(include=np.number).shape[1]),
        "categorical_columns": int(df.select_dtypes(exclude=np.number).shape[1]),
        "has_target": bool(target_col and target_col in df.columns),
        "positive_rate": None,
        "target_classes": None,
        "summary_points": [],
    }

    if target_col and target_col in df.columns:
        target_counts = df[target_col].astype(str).value_counts().reset_index()
        target_counts.columns = [target_col, "count"]
        metrics["target_classes"] = df[target_col].astype(str).nunique(dropna=False)

        charts["target_distribution"] = px.bar(
            target_counts,
            x=target_col,
            y="count",
            title=f"Distribution of {target_col}",
            text="count",
        )

        if is_binary_target(df[target_col]):
            rate = positive_mask(df[target_col]).mean() * 100
            metrics["positive_rate"] = rate
            metrics["summary_points"].append(f"The positive rate for {target_col} is approximately {rate:.2f}%.")

            preferred_cols = ["Contract", "tenure", "MonthlyCharges", "PaymentMethod", "InternetService"]
            possible_cols = [c for c in preferred_cols if c in df.columns]
            if not possible_cols:
                # Pick sensible non-target grouping columns.
                cat_cols = [c for c in df.select_dtypes(exclude=np.number).columns if c != target_col and df[c].nunique(dropna=True) <= 20]
                num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target_col]
                possible_cols = cat_cols[:2] + num_cols[:2]

            chart_num = 1
            for col in possible_cols[:3]:
                grouped = grouped_positive_rate(df, target_col, col)
                if grouped.empty:
                    continue
                group_col = grouped.columns[0]
                fig = px.bar(
                    grouped,
                    x=group_col,
                    y="positive_rate",
                    title=f"{target_col} positive rate by {col}",
                    text=grouped["positive_rate"].round(2),
                )
                fig.update_layout(yaxis_title=f"{target_col} positive rate (%)")
                charts[f"chart_{chart_num}"] = fig
                chart_num += 1

                highest = grouped.iloc[0]
                metrics["summary_points"].append(
                    f"Highest positive rate by {col}: {highest[group_col]} ({highest['positive_rate']:.2f}%)."
                )

    generic_points, generic_charts = create_generic_charts(df)
    metrics["summary_points"] = generic_points + metrics["summary_points"]
    charts.update({k: v for k, v in generic_charts.items() if k not in charts})

    return metrics, charts
