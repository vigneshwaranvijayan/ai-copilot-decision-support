import pandas as pd

from src.document_retrieval import (
    chunk_text,
    chunks_to_dataframe,
    hybrid_search,
    evaluate_retrieval,
    filter_chunks_by_metadata,
)
from src.text_utils import normalise_question


def test_overlapping_chunking_preserves_metadata_and_overlap():
    text = " ".join([f"word{i}" for i in range(300)])
    chunks = chunk_text(text, document_name="report.txt", chunk_words=100, overlap_words=20, base_metadata={"department": "sales", "topic": "churn"})
    assert len(chunks) >= 3
    assert chunks[0].metadata["department"] == "sales"
    assert chunks[1].metadata["word_start"] == 80
    df = chunks_to_dataframe(chunks)
    assert "preview" in df.columns


def test_metadata_filter_then_hybrid_retrieval_finds_relevant_chunk():
    docs = []
    docs.extend(chunk_text("Pricing complaints and monthly charges cause customer churn risk.", document_name="retention.md", chunk_words=60, overlap_words=10, base_metadata={"department": "customer", "topic": "retention"}))
    docs.extend(chunk_text("Warehouse stock levels and delivery operations need monitoring.", document_name="ops.md", chunk_words=60, overlap_words=10, base_metadata={"department": "operations", "topic": "stock"}))
    filtered = filter_chunks_by_metadata(docs, {"department": "customer"})
    assert len(filtered) == 1
    results = hybrid_search("monthly charge churn", docs, metadata_filters={"department": "customer"}, top_k=3)
    assert not results.empty
    assert "retention.md" in results.iloc[0]["document_name"]


def test_retrieval_metrics():
    metrics = evaluate_retrieval(["a", "b", "c"], ["b", "x"], k=2)
    assert metrics["precision@2"] == 0.5
    assert metrics["recall@2"] == 0.5
    assert metrics["mrr"] == 0.5
    assert "ndcg@2" in metrics


def test_user_typo_autocorrect_for_retrieval_terms():
    fixed, corrections = normalise_question("visulation precison reterival qusstion", ["visualisation", "precision", "retrieval", "question"])
    assert "visualisation" in fixed
    assert "precision" in fixed
    assert "retrieval" in fixed
    assert corrections
