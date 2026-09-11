from pathlib import Path
import pandas as pd

from src.validation import build_readiness_report
from src.readiness_gates import PASS, WARNING, FAIL, answer_readiness_status
from src.research_contribution import contribution_table
from src.evaluation_framework import evaluation_metrics_table
from src.vector_memory import add_memory_document, query_memory


def test_v15_readiness_report_pass_for_clean_dataset():
    df = pd.DataFrame({"tenure": [1, 2, 3, 4, 5, 6], "contract": ["m", "y", "m", "y", "m", "y"], "churn": [0, 1, 0, 1, 0, 1]})
    report = build_readiness_report(df, dataset_name="tiny", target_column="churn")
    assert report.rows == 6
    assert report.quality_score >= 80
    assert any(g.gate == "Target column" and g.status == PASS for g in report.gates)
    assert any(g.gate == "Target validity" and g.status == PASS for g in report.gates)


def test_v15_answer_readiness_fail_when_required_column_missing():
    assert answer_readiness_status(has_required_columns=False, has_evidence=True) == FAIL
    assert answer_readiness_status(has_required_columns=True, has_evidence=True) == PASS
    assert answer_readiness_status(has_required_columns=True, has_evidence=True, has_model_or_eda=False) == WARNING


def test_v15_research_and_evaluation_tables_non_empty():
    assert len(contribution_table()) >= 5
    assert len(evaluation_metrics_table()) >= 6


def test_v15_vector_memory_fallback_or_chromadb(tmp_path):
    record_id = add_memory_document("churn readiness PASS with random forest evidence", metadata={"dataset": "telco"}, memory_dir=tmp_path)
    assert record_id
    results = query_memory("churn evidence", dataset_name="telco", memory_dir=tmp_path)
    assert isinstance(results, list)
    assert len(results) >= 1
