"""Local audit persistence for the proof-of-concept.

SQLite is used for the local prototype because it is easy to run without
separate infrastructure. The architecture remains PostgreSQL-ready and uses ChromaDB as the semantic long-term memory layer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional
import json
import sqlite3
import time

import pandas as pd

DEFAULT_DB_PATH = Path("audit_store.db")


def init_audit_store(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    path = Path(db_path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                event_type TEXT NOT NULL,
                dataset_name TEXT,
                status TEXT,
                question TEXT,
                details_json TEXT
            )
            """
        )
        conn.commit()


def record_audit_event(
    event_type: str,
    dataset_name: str = "",
    status: str = "INFO",
    question: str = "",
    details: Optional[Dict[str, object]] = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    init_audit_store(db_path)
    with sqlite3.connect(Path(db_path)) as conn:
        conn.execute(
            "INSERT INTO audit_events (created_at, event_type, dataset_name, status, question, details_json) VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), event_type, dataset_name, status, question, json.dumps(details or {}, default=str)),
        )
        conn.commit()


def read_audit_events(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    init_audit_store(db_path)
    with sqlite3.connect(Path(db_path)) as conn:
        df = pd.read_sql_query("SELECT * FROM audit_events ORDER BY id DESC", conn)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], unit="s")
    return df
