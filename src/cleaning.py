"""Automatic cleaning and schema utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class CleaningReport:
    original_shape: Tuple[int, int]
    cleaned_shape: Tuple[int, int]
    dropped_empty_rows: int = 0
    dropped_empty_columns: List[str] = field(default_factory=list)
    dropped_duplicate_rows: int = 0
    renamed_columns: Dict[str, str] = field(default_factory=dict)
    converted_numeric_columns: List[str] = field(default_factory=list)
    converted_datetime_columns: List[str] = field(default_factory=list)
    trimmed_text_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            ("Original shape", f"{self.original_shape[0]} rows × {self.original_shape[1]} columns"),
            ("Cleaned shape", f"{self.cleaned_shape[0]} rows × {self.cleaned_shape[1]} columns"),
            ("Dropped empty rows", self.dropped_empty_rows),
            ("Dropped duplicate rows", self.dropped_duplicate_rows),
            ("Dropped empty columns", ", ".join(self.dropped_empty_columns) or "None"),
            ("Numeric conversions", ", ".join(self.converted_numeric_columns) or "None"),
            ("Datetime conversions", ", ".join(self.converted_datetime_columns) or "None"),
            ("Text columns trimmed", ", ".join(self.trimmed_text_columns) or "None"),
            ("Warnings", "; ".join(self.warnings) or "None"),
        ]
        return pd.DataFrame(rows, columns=["Step", "Result"])


def clean_column_name(name: object) -> str:
    text = str(name).strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "column"


def make_unique_columns(columns: List[str]) -> List[str]:
    counts: Dict[str, int] = {}
    unique: List[str] = []
    for col in columns:
        base = clean_column_name(col)
        counts[base] = counts.get(base, 0) + 1
        unique.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return unique


def _try_convert_numeric(series: pd.Series, threshold: float = 0.85) -> Tuple[pd.Series, bool]:
    if pd.api.types.is_numeric_dtype(series):
        return series, False
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return series, False
    cleaned = (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA})
        .str.replace(r"[£$€,]", "", regex=True)
        .str.replace("%", "", regex=False)
    )
    converted = pd.to_numeric(cleaned, errors="coerce")
    non_missing = cleaned.notna().sum()
    if non_missing == 0:
        return series, False
    success_rate = converted.notna().sum() / non_missing
    if success_rate >= threshold:
        return converted, True
    return series, False


def _try_convert_datetime(series: pd.Series, threshold: float = 0.85) -> Tuple[pd.Series, bool]:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, False
    name_hint = any(token in str(series.name).lower() for token in ["date", "time", "created", "updated", "invoice"])
    if not name_hint and series.dropna().astype(str).head(20).str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}").mean() < 0.5:
        return series, False
    converted = pd.to_datetime(series, errors="coerce", dayfirst=True)
    non_missing = series.notna().sum()
    if non_missing == 0:
        return series, False
    success_rate = converted.notna().sum() / non_missing
    if success_rate >= threshold:
        return converted, True
    return series, False


def clean_dataframe(df: pd.DataFrame, drop_duplicates: bool = True) -> Tuple[pd.DataFrame, CleaningReport]:
    """Clean tabular business data without hard-coding a dataset."""
    if df is None:
        raise ValueError("No dataframe supplied")
    original_shape = df.shape
    cleaned = df.copy()

    renamed = dict(zip(list(cleaned.columns), make_unique_columns(list(cleaned.columns))))
    cleaned.columns = list(renamed.values())

    before_rows = cleaned.shape[0]
    cleaned = cleaned.dropna(how="all")
    dropped_empty_rows = before_rows - cleaned.shape[0]

    empty_cols = [c for c in cleaned.columns if cleaned[c].isna().all()]
    if empty_cols:
        cleaned = cleaned.drop(columns=empty_cols)

    trimmed_cols: List[str] = []
    numeric_cols: List[str] = []
    datetime_cols: List[str] = []

    for col in list(cleaned.columns):
        if pd.api.types.is_object_dtype(cleaned[col]) or pd.api.types.is_string_dtype(cleaned[col]):
            before = cleaned[col].copy()
            cleaned[col] = cleaned[col].astype("string").str.strip()
            cleaned[col] = cleaned[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA, "NULL": pd.NA})
            if not cleaned[col].equals(before):
                trimmed_cols.append(col)

        converted_num, changed_num = _try_convert_numeric(cleaned[col])
        if changed_num:
            cleaned[col] = converted_num
            numeric_cols.append(col)
            continue

        converted_dt, changed_dt = _try_convert_datetime(cleaned[col])
        if changed_dt:
            cleaned[col] = converted_dt
            datetime_cols.append(col)

    before_dupes = cleaned.shape[0]
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()
    dropped_dupes = before_dupes - cleaned.shape[0]

    warnings: List[str] = []
    if cleaned.shape[0] == 0:
        warnings.append("Dataset is empty after cleaning.")
    if cleaned.shape[1] == 0:
        warnings.append("No usable columns remain after cleaning.")
    if cleaned.shape[0] > 250_000:
        warnings.append("Large dataset detected. Modelling may use sampling for speed, while EDA still profiles the active data.")

    report = CleaningReport(
        original_shape=original_shape,
        cleaned_shape=cleaned.shape,
        dropped_empty_rows=dropped_empty_rows,
        dropped_empty_columns=empty_cols,
        dropped_duplicate_rows=dropped_dupes,
        renamed_columns={k: v for k, v in renamed.items() if str(k) != v},
        converted_numeric_columns=numeric_cols,
        converted_datetime_columns=datetime_cols,
        trimmed_text_columns=trimmed_cols,
        warnings=warnings,
    )
    return cleaned.reset_index(drop=True), report


def detect_column_roles(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric + datetime_cols and df[c].nunique(dropna=True) <= max(100, min(500, df.shape[0] // 10 if df.shape[0] else 100))]
    text = [c for c in df.columns if c not in numeric + datetime_cols + categorical]
    ids = [c for c in df.columns if any(token in c.lower() for token in ["id", "customer", "client", "account", "user"]) and df[c].nunique(dropna=True) > 0.7 * max(len(df), 1)]
    return {"numeric": numeric, "categorical": categorical, "datetime": datetime_cols, "text": text, "id_like": ids}
