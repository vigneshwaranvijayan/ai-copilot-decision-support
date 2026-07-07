"""Data loading utilities for files and public URL/API links."""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import requests

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json"}


@dataclass
class LoadedDataset:
    name: str
    dataframe: pd.DataFrame
    source_type: str
    notes: str = ""


def _normalise_name(name: str) -> str:
    return Path(name).name.replace(" ", "_")


def _read_csv_bytes(data: bytes, name: str) -> pd.DataFrame:
    suffix = Path(name.lower()).suffix
    sep = "\t" if suffix == ".tsv" else None
    # sep=None lets pandas sniff comma/semicolon where possible.
    return pd.read_csv(io.BytesIO(data), sep=sep, engine="python")


def _json_to_dataframe(data: Union[dict, list]) -> pd.DataFrame:
    if isinstance(data, list):
        return pd.json_normalize(data)
    if isinstance(data, dict):
        # Common API response shapes: {data:[...]}, {results:[...]}, {items:[...]}
        for key in ["data", "results", "items", "records", "rows"]:
            if key in data and isinstance(data[key], list):
                return pd.json_normalize(data[key])
        return pd.json_normalize(data)
    raise ValueError("Unsupported JSON structure")


def read_bytes_to_dataframe(data: bytes, name: str) -> pd.DataFrame:
    """Read supported byte content to a DataFrame."""
    suffix = Path(name.lower()).suffix
    if suffix in {".csv", ".tsv", ".txt"}:
        return _read_csv_bytes(data, name)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(data))
    if suffix == ".json":
        parsed = json.loads(data.decode("utf-8"))
        return _json_to_dataframe(parsed)
    raise ValueError(f"Unsupported file type: {suffix}")


def load_uploaded_files(uploaded_files: Iterable) -> List[LoadedDataset]:
    """Load Streamlit uploaded files, including ZIPs with supported files inside."""
    datasets: List[LoadedDataset] = []
    for uploaded in uploaded_files or []:
        raw_name = getattr(uploaded, "name", "uploaded_file")
        data = uploaded.getvalue()
        suffix = Path(raw_name.lower()).suffix
        if suffix == ".zip":
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    inner_name = info.filename
                    inner_suffix = Path(inner_name.lower()).suffix
                    if inner_suffix not in SUPPORTED_SUFFIXES:
                        continue
                    try:
                        df = read_bytes_to_dataframe(zf.read(info), inner_name)
                        datasets.append(
                            LoadedDataset(
                                name=_normalise_name(inner_name),
                                dataframe=df,
                                source_type="zip_upload",
                                notes=f"Loaded from {raw_name}",
                            )
                        )
                    except Exception as exc:  # pragma: no cover - UI displays note
                        datasets.append(
                            LoadedDataset(
                                name=_normalise_name(inner_name),
                                dataframe=pd.DataFrame(),
                                source_type="zip_upload_error",
                                notes=str(exc),
                            )
                        )
        else:
            if suffix not in SUPPORTED_SUFFIXES:
                continue
            df = read_bytes_to_dataframe(data, raw_name)
            datasets.append(LoadedDataset(_normalise_name(raw_name), df, "upload"))
    return datasets


def load_public_url(url: str, timeout: int = 30) -> LoadedDataset:
    """Load a public CSV/JSON/Excel URL or API endpoint into a DataFrame."""
    if not url or not url.strip():
        raise ValueError("URL is empty")
    url = url.strip()
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    suffix = Path(url.split("?")[0].lower()).suffix
    name = _normalise_name(Path(url.split("?")[0]).name or "api_dataset.json")

    if "json" in content_type or suffix == ".json":
        parsed = response.json()
        return LoadedDataset(name=name if suffix else "api_dataset.json", dataframe=_json_to_dataframe(parsed), source_type="url_api")
    if suffix in {".xlsx", ".xls"} or "spreadsheet" in content_type or "excel" in content_type:
        return LoadedDataset(name=name, dataframe=pd.read_excel(io.BytesIO(response.content)), source_type="url_excel")
    # Default to CSV-style table for public API links and CSV URLs.
    return LoadedDataset(name=name if suffix else "api_dataset.csv", dataframe=_read_csv_bytes(response.content, name), source_type="url_csv")


def dataset_profile(df: pd.DataFrame) -> Dict[str, object]:
    """Return quick profile metadata for a dataset."""
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024)) if not df.empty else 0.0
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_mb": round(memory_mb, 2),
        "numeric_columns": int(df.select_dtypes(include="number").shape[1]),
        "object_columns": int(df.select_dtypes(include=["object", "category", "string"]).shape[1]),
        "datetime_columns": int(df.select_dtypes(include=["datetime", "datetimetz"]).shape[1]),
        "missing_cells": int(df.isna().sum().sum()) if not df.empty else 0,
        "duplicate_rows": int(df.duplicated().sum()) if not df.empty else 0,
    }


def find_common_columns(left: pd.DataFrame, right: pd.DataFrame) -> List[str]:
    """Return likely shared key columns between two datasets."""
    common = sorted(set(left.columns).intersection(set(right.columns)))
    preferred = [c for c in common if any(token in c.lower() for token in ["id", "customer", "client", "account", "user"])]
    return preferred + [c for c in common if c not in preferred]


def merge_datasets(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_key: str,
    right_key: str,
    how: str = "left",
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Merge two datasets by selected keys, with optional safety cap."""
    if left_key not in left.columns or right_key not in right.columns:
        raise KeyError("Merge key not found in one of the datasets")
    merged = pd.merge(left, right, left_on=left_key, right_on=right_key, how=how, suffixes=("", "_right"))
    if max_rows is not None and merged.shape[0] > max_rows:
        merged = merged.sample(max_rows, random_state=42)
    return merged
