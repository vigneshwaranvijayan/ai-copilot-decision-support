from pathlib import Path
import pandas as pd

from src.modeling import infer_clear_binary_target, infer_positive_label
from src.validation import build_readiness_report


def _telco_like_df(n=600):
    # Multiple binary columns are deliberate: churn must win by target-name hint.
    return pd.DataFrame({
        "gender": ["Female", "Male"] * (n // 2),
        "partner": ["Yes", "No"] * (n // 2),
        "tenure": list(range(n)),
        "monthlycharges": [50.0 + (i % 20) for i in range(n)],
        "churn": (["No"] * 440 + ["Yes"] * 160)[:n],
    })


def test_telco_readiness_auto_detects_churn_and_passes_before_model_training():
    df = _telco_like_df()
    target = infer_clear_binary_target(df)
    assert target == "churn"
    report = build_readiness_report(df, dataset_name="telco.csv", target_column=target)
    assert report.overall_status == "PASS"
    gates = {g.gate: g.status for g in report.gates}
    assert gates["Target column"] == "PASS"
    assert gates["Target validity"] == "PASS"
    assert gates["Class balance"] == "PASS"
    assert gates["Evidence availability"] == "PASS"


def test_positive_class_for_telco_is_yes():
    df = _telco_like_df()
    assert infer_positive_label(df["churn"]) == "Yes"


def test_app_defaults_auto_model_on_and_uses_dataset_specific_positive_class_key():
    app_text = Path(__file__).parents[1].joinpath("app.py").read_text(encoding="utf-8")
    assert '"auto_model_enabled": True' in app_text
    assert 'key=f"model_pos::{name}::{target_col}"' in app_text
    assert 'format_func=lambda value: str(value)' in app_text


def test_launcher_reuses_shared_runtime_between_extracted_versions():
    text = Path(__file__).parents[1].joinpath("run_app.bat").read_text(encoding="utf-8").lower()
    assert "%localappdata%\\datasetgroundedaicopilot" in text
    assert "%copilot_runtime%\\.runtime_ready" in text
    assert "--server.filewatchertype none" in text
