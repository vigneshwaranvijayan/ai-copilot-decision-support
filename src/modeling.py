"""Model training, model comparison and explanation support."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # optional dependency
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None
    XGBOOST_AVAILABLE = False


TARGET_NAME_HINTS = [
    "churn",
    "attrition",
    "exited",
    "target",
    "label",
    "class",
    "y",
    "subscribed",
    "default",
    "fraud",
    "returned",
    "cancelled",
    "late_delivery",
    "high_risk",
]



ID_COLUMN_NAME_HINTS = {"id", "customerid", "clientid", "accountid", "userid", "rowid", "caseid", "recordid"}


def is_identifier_like_column(series: pd.Series, column_name: str) -> bool:
    """Detect identifier columns that should not be used as predictive features.

    The Telco churn file includes `customerID`. Keeping this as a one-hot encoded
    model feature is academically risky because identifiers are not business
    drivers. The prototype removes clear ID-like columns before training so that
    the leaderboard and explanation drivers reflect meaningful customer features.
    """
    name = str(column_name).strip().lower().replace("_", "").replace("-", "")
    if name in ID_COLUMN_NAME_HINTS or name.endswith("id"):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_ratio = non_null.astype(str).nunique(dropna=True) / max(len(non_null), 1)
    avg_len = non_null.astype(str).str.len().mean()
    # High-cardinality text columns with almost one value per row are usually
    # identifiers, especially when values are code-like rather than categories.
    if unique_ratio >= 0.90 and avg_len >= 6 and not pd.api.types.is_numeric_dtype(non_null):
        return True
    return False


@dataclass
class ModelResult:
    model_name: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    confusion: np.ndarray
    positive_label: object
    target_column: str
    feature_columns: List[str]
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]
    sampled_training_rows: int
    training_seconds: float = 0.0
    notes: str = ""


@dataclass
class TrainingOutput:
    results: List[ModelResult]
    best_result: Optional[ModelResult]
    leaderboard: pd.DataFrame
    skipped_models: Dict[str, str]
    target_column: str
    positive_label: object
    class_distribution: pd.DataFrame
    total_training_seconds: float = 0.0


def detect_binary_targets(df: pd.DataFrame) -> List[str]:
    candidates: List[Tuple[int, str]] = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        unique_count = series.nunique(dropna=True)
        if unique_count != 2:
            continue
        lower = col.lower()
        score = 0
        for idx, hint in enumerate(TARGET_NAME_HINTS):
            if lower == hint:
                score += 100 - idx
            elif hint in lower:
                score += 50 - idx
        # ID-like binary columns are usually not target columns.
        if any(token in lower for token in ["id", "phone", "postcode", "zip"]):
            score -= 30
        candidates.append((score, col))
    return [col for _, col in sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True)]


def infer_clear_binary_target(df: pd.DataFrame) -> Optional[str]:
    """Return a safe auto-detected binary target when the intent is clear.

    Exact business target names such as ``churn``/``attrition``/``target`` are
    preferred. If a dataset contains exactly one binary column, that single
    candidate can also be used provisionally. Multiple unrelated binary fields
    are left for explicit user confirmation.
    """
    candidates = detect_binary_targets(df)
    if not candidates:
        return None
    normalized_hints = {
        str(h).lower().replace("_", "").replace("-", "").replace(" ", "")
        for h in TARGET_NAME_HINTS
    }
    for col in candidates:
        normalized = str(col).lower().replace("_", "").replace("-", "").replace(" ", "")
        if normalized in normalized_hints:
            return col
    return candidates[0] if len(candidates) == 1 else None


def infer_positive_label(series: pd.Series) -> object:
    values = list(series.dropna().unique())
    if not values:
        return 1
    positive_tokens = {"yes", "true", "1", "churn", "exited", "attrition", "positive", "subscribed", "default", "fraud", "returned", "cancelled", "late", "high", "highrisk"}
    for value in values:
        if str(value).strip().lower().replace(" ", "") in positive_tokens:
            return value
    # Use minority class as positive by default for risk-style classification.
    counts = series.value_counts(dropna=True)
    return counts.idxmin() if not counts.empty else values[-1]


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_ohe()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_candidate_models(include_xgboost: bool = False, random_state: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "MLP Neural Network Baseline": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.15,
        ),
    }
    if include_xgboost and XGBOOST_AVAILABLE and XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            learning_rate=0.06,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    return models


def _safe_proba(pipeline: Pipeline, X_test: pd.DataFrame) -> Optional[np.ndarray]:
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
    if hasattr(pipeline, "decision_function"):
        score = pipeline.decision_function(X_test)
        score = np.asarray(score, dtype=float)
        return (score - score.min()) / max(score.max() - score.min(), 1e-9)
    return None


def _metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: Optional[np.ndarray], positive_label: object) -> Dict[str, float]:
    y_true_bin = (y_true.astype(str) == str(positive_label)).astype(int)
    # sklearn pipeline predicts original labels, except XGBoost may predict encoded integers if target encoded.
    if set(np.unique(y_pred)).issubset({0, 1}) and str(positive_label) not in {"1", "True", "true", "yes", "Yes"}:
        y_pred_bin = np.asarray(y_pred).astype(int)
    else:
        y_pred_bin = (pd.Series(y_pred).astype(str) == str(positive_label)).astype(int).to_numpy()
    output = {
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "roc_auc": np.nan,
    }
    if y_proba is not None and len(np.unique(y_true_bin)) == 2:
        try:
            output["roc_auc"] = float(roc_auc_score(y_true_bin, y_proba))
        except Exception:
            output["roc_auc"] = np.nan
    return output


def _encode_target_for_xgb(y: pd.Series, positive_label: object) -> pd.Series:
    return (y.astype(str) == str(positive_label)).astype(int)


def _sample_for_training(df: pd.DataFrame, target: str, max_rows: int, random_state: int) -> pd.DataFrame:
    if df.shape[0] <= max_rows:
        return df
    # Stratified sampling when possible.
    counts = df[target].value_counts(dropna=False)
    if counts.shape[0] == 2 and counts.min() >= 2:
        frac = max_rows / df.shape[0]
        return df.groupby(target, group_keys=False).apply(lambda x: x.sample(max(1, int(len(x) * frac)), random_state=random_state)).head(max_rows)
    return df.sample(max_rows, random_state=random_state)



def prepare_features_for_model(X: pd.DataFrame) -> pd.DataFrame:
    """Convert date/time columns to compact numeric features and normalise text columns."""
    out = X.copy()
    for col in list(out.columns):
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[f"{col}_year"] = out[col].dt.year
            out[f"{col}_month"] = out[col].dt.month
            out[f"{col}_dayofweek"] = out[col].dt.dayofweek
            out = out.drop(columns=[col])
        elif pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]) or isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype("string").fillna("Missing")
    return out

def train_models(
    df: pd.DataFrame,
    target_column: str,
    positive_label: Optional[object] = None,
    include_xgboost: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    max_training_rows: int = 120_000,
) -> TrainingOutput:
    """Train the final four assessed classification models on the uploaded dataset and select the best.

    The assessed dissertation workflow uses exactly four models by default:
    Logistic Regression, Random Forest, Gradient Boosting, and MLP Neural
    Network Baseline. XGBoost remains disabled unless explicitly enabled in
    code for exploratory comparison outside the assessed workflow.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column not found: {target_column}")
    working = df.dropna(subset=[target_column]).copy()
    if working[target_column].nunique(dropna=True) != 2:
        raise ValueError("The selected target must be binary for this proof-of-concept model workflow.")

    positive_label = positive_label if positive_label is not None else infer_positive_label(working[target_column])
    class_dist = working[target_column].value_counts(dropna=False).rename_axis(target_column).reset_index(name="count")
    class_dist["percent"] = (class_dist["count"] / class_dist["count"].sum() * 100).round(2)

    working = _sample_for_training(working, target_column, max_training_rows, random_state)
    # Exclude identifier-like columns such as customerID from modelling. They can
    # memorise individual records and create poor explanation evidence without
    # providing a reusable business driver.
    feature_columns = [
        c for c in working.columns
        if c != target_column and not is_identifier_like_column(working[c], c)
    ]
    X = prepare_features_for_model(working[feature_columns].copy())
    # Drop columns that are completely missing or have one unique value.
    keep_cols = [c for c in X.columns if X[c].notna().sum() > 0 and X[c].nunique(dropna=True) > 1]
    X = X[keep_cols]
    y_original = working[target_column].copy()
    y_bin = _encode_target_for_xgb(y_original, positive_label)

    stratify = y_original if y_original.value_counts().min() >= 2 else None
    X_train, X_test, y_train_original, y_test_original, y_train_bin, y_test_bin = train_test_split(
        X,
        y_original,
        y_bin,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    models = get_candidate_models(include_xgboost=include_xgboost, random_state=random_state)
    results: List[ModelResult] = []
    skipped: Dict[str, str] = {}
    total_timer_start = time.perf_counter()

    for name, estimator in models.items():
        model_timer_start = time.perf_counter()
        try:
            pipeline = Pipeline(steps=[("preprocess", build_preprocessor(X_train)), ("model", estimator)])
            if name in {"XGBoost", "MLP Neural Network Baseline"}:
                # XGBoost and recent scikit-learn MLP early-stopping behave more
                # reliably with encoded binary labels. Metrics are still reported
                # against the same positive class used by the dissertation workflow.
                pipeline.fit(X_train, y_train_bin)
                y_pred = pipeline.predict(X_test)
                y_proba = _safe_proba(pipeline, X_test)
                metrics = _metrics(y_test_bin, y_pred, y_proba, positive_label=1)
                y_true_for_conf = y_test_bin
                y_pred_for_conf = np.asarray(y_pred).astype(int)
                positive_for_result = 1
            else:
                pipeline.fit(X_train, y_train_original)
                y_pred = pipeline.predict(X_test)
                y_proba = _safe_proba(pipeline, X_test)
                metrics = _metrics(y_test_original, y_pred, y_proba, positive_label)
                y_true_for_conf = (y_test_original.astype(str) == str(positive_label)).astype(int)
                y_pred_for_conf = (pd.Series(y_pred).astype(str) == str(positive_label)).astype(int).to_numpy()
                positive_for_result = positive_label

            conf = confusion_matrix(y_true_for_conf, y_pred_for_conf, labels=[0, 1])
            training_seconds = round(time.perf_counter() - model_timer_start, 3)
            metrics["training_seconds"] = training_seconds
            results.append(
                ModelResult(
                    model_name=name,
                    pipeline=pipeline,
                    metrics=metrics,
                    confusion=conf,
                    positive_label=positive_for_result,
                    target_column=target_column,
                    feature_columns=keep_cols,
                    X_test=X_test,
                    y_test=y_test_original,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    sampled_training_rows=working.shape[0],
                    training_seconds=training_seconds,
                    notes="Target encoded as 1=positive class for this model." if name in {"XGBoost", "MLP Neural Network Baseline"} else "",
                )
            )
        except Exception as exc:
            skipped[name] = str(exc)

    total_training_seconds = round(time.perf_counter() - total_timer_start, 3)

    if not results:
        raise RuntimeError("No model could be trained successfully. Check the dataset and target column.")

    leaderboard = pd.DataFrame([
        {
            "model": r.model_name,
            "accuracy": round(r.metrics["accuracy"], 4),
            "precision": round(r.metrics["precision"], 4),
            "recall": round(r.metrics["recall"], 4),
            "f1": round(r.metrics["f1"], 4),
            "roc_auc": round(r.metrics["roc_auc"], 4) if not np.isnan(r.metrics["roc_auc"]) else np.nan,
            "modeling_rows": r.sampled_training_rows,
            "training_rows": int(r.sampled_training_rows - len(r.X_test)),
            "test_rows": int(len(r.X_test)),
            "training_time_sec": round(float(getattr(r, "training_seconds", 0.0)), 3),
        }
        for r in results
    ])
    # Select the best model in a way that favours balanced predictive quality.
    # Earlier versions averaged F1, recall and ROC-AUC, which could select a
    # model that predicts almost every customer as positive because recall is 1.0.
    # For a decision-support prototype, F1 is the main selection metric because
    # it balances precision and recall; ROC-AUC, recall and precision are used
    # only as tie-breakers.
    leaderboard = leaderboard.sort_values(["f1", "roc_auc", "recall", "precision"], ascending=False).reset_index(drop=True)
    best_model_name = leaderboard.loc[0, "model"]
    best = next(r for r in results if r.model_name == best_model_name)
    return TrainingOutput(results, best, leaderboard, skipped, target_column, positive_label, class_dist, total_training_seconds)


def get_feature_names(pipeline: Pipeline) -> List[str]:
    preprocess = pipeline.named_steps.get("preprocess")
    if preprocess is None:
        return []
    try:
        return list(preprocess.get_feature_names_out())
    except Exception:
        return []


def transformed_matrix(pipeline: Pipeline, X: pd.DataFrame):
    preprocess = pipeline.named_steps.get("preprocess")
    if preprocess is None:
        raise ValueError("Pipeline has no preprocessor")
    return preprocess.transform(X)

# ---------------------------------------------------------------------------
# Optional regression mode for multi-dataset robustness
# ---------------------------------------------------------------------------

@dataclass
class RegressionModelResult:
    model_name: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    target_column: str
    feature_columns: List[str]
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    sampled_training_rows: int
    training_seconds: float = 0.0
    notes: str = ""


@dataclass
class RegressionTrainingOutput:
    results: List[RegressionModelResult]
    best_result: Optional[RegressionModelResult]
    leaderboard: pd.DataFrame
    skipped_models: Dict[str, str]
    target_column: str
    target_summary: pd.DataFrame


REGRESSION_TARGET_HINTS = [
    "sales", "revenue", "profit", "amount", "total", "price", "monthlycharges",
    "income", "salary", "score", "rating", "quantity", "units", "duration",
    "deliverytime", "resolutiontime", "cost", "value", "balance",
]


def detect_regression_targets(df: pd.DataFrame, max_unique_ratio: float = 0.05) -> List[str]:
    """Detect numeric columns that are plausible continuous regression targets."""
    candidates: List[Tuple[int, str]] = []
    n = max(len(df), 1)
    for col in df.select_dtypes(include=np.number).columns:
        s_col = df[col].dropna()
        if s_col.empty:
            continue
        unique_count = int(s_col.nunique(dropna=True))
        if unique_count <= 2:
            continue
        lower = col.lower().replace("_", "")
        if any(token in lower for token in ["id", "phone", "postcode", "zip"]):
            continue
        unique_ratio = unique_count / n
        score = unique_count
        for idx, hint in enumerate(REGRESSION_TARGET_HINTS):
            if hint in lower:
                score += 100 - idx
        if unique_ratio < max_unique_ratio and score < 80:
            continue
        candidates.append((score, col))
    return [c for _, c in sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True)]


def get_candidate_regression_models(include_xgboost: bool = False, random_state: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=random_state, n_jobs=-1,
        ),
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=random_state, n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=random_state),
    }
    if include_xgboost and XGBOOST_AVAILABLE and "XGBRegressor" in globals() and XGBRegressor is not None:
        models["XGBoost Regressor"] = XGBRegressor(
            n_estimators=250,
            learning_rate=0.06,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    return models


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_num = pd.to_numeric(y_true, errors="coerce")
    y_pred_num = np.asarray(y_pred, dtype=float)
    mask = y_true_num.notna() & np.isfinite(y_pred_num)
    if not mask.any():
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    yt = y_true_num[mask].to_numpy(dtype=float)
    yp = y_pred_num[mask]
    try:
        rmse = float(mean_squared_error(yt, yp, squared=False))
    except TypeError:
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": rmse,
        "r2": float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan,
    }


def _sample_for_regression(df: pd.DataFrame, target: str, max_rows: int, random_state: int) -> pd.DataFrame:
    working = df.dropna(subset=[target]).copy()
    if working.shape[0] <= max_rows:
        return working
    return working.sample(max_rows, random_state=random_state)


def train_regression_models(
    df: pd.DataFrame,
    target_column: str,
    include_xgboost: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    max_training_rows: int = 120_000,
) -> RegressionTrainingOutput:
    """Train candidate regression models for numeric business outcomes.

    This is an optional robustness mode. The main dissertation evaluation remains
    binary customer churn classification with SHAP and Copilot explanation.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column not found: {target_column}")
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError("The regression target must be numeric.")
    working = _sample_for_regression(df, target_column, max_training_rows, random_state)
    y = pd.to_numeric(working[target_column], errors="coerce")
    working = working[y.notna()].copy()
    y = y[y.notna()]
    if working.shape[0] < 20:
        raise ValueError("Need at least 20 non-missing rows for regression mode.")
    if y.nunique(dropna=True) <= 2:
        raise ValueError("This looks like a binary target. Use classification mode instead.")

    target_summary = pd.DataFrame([
        {"statistic": "rows_used", "value": int(working.shape[0])},
        {"statistic": "mean", "value": float(y.mean())},
        {"statistic": "median", "value": float(y.median())},
        {"statistic": "std", "value": float(y.std())},
        {"statistic": "min", "value": float(y.min())},
        {"statistic": "max", "value": float(y.max())},
    ])

    feature_columns = [c for c in working.columns if c != target_column]
    X = prepare_features_for_model(working[feature_columns].copy())
    keep_cols = [c for c in X.columns if X[c].notna().sum() > 0 and X[c].nunique(dropna=True) > 1]
    X = X[keep_cols]
    if not keep_cols:
        raise ValueError("No usable feature columns remain after preprocessing.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    models = get_candidate_regression_models(include_xgboost=include_xgboost, random_state=random_state)
    results: List[RegressionModelResult] = []
    skipped: Dict[str, str] = {}

    for model_name, estimator in models.items():
        try:
            pipeline = Pipeline(steps=[("preprocess", build_preprocessor(X_train)), ("model", estimator)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            metrics = _regression_metrics(y_test, y_pred)
            results.append(
                RegressionModelResult(
                    model_name=model_name,
                    pipeline=pipeline,
                    metrics=metrics,
                    target_column=target_column,
                    feature_columns=keep_cols,
                    X_test=X_test,
                    y_test=y_test,
                    y_pred=np.asarray(y_pred),
                    sampled_training_rows=working.shape[0],
                    notes="Optional regression robustness mode; not the main churn evaluation workflow.",
                )
            )
        except Exception as exc:
            skipped[model_name] = str(exc)

    if not results:
        raise RuntimeError("No regression model could be trained successfully. Check the dataset and numeric target column.")

    leaderboard = pd.DataFrame([
        {
            "model": r.model_name,
            "mae": round(r.metrics["mae"], 4) if not np.isnan(r.metrics["mae"]) else np.nan,
            "rmse": round(r.metrics["rmse"], 4) if not np.isnan(r.metrics["rmse"]) else np.nan,
            "r2": round(r.metrics["r2"], 4) if not np.isnan(r.metrics["r2"]) else np.nan,
            "modeling_rows": r.sampled_training_rows,
            "training_rows": int(r.sampled_training_rows - len(r.X_test)),
            "test_rows": int(len(r.X_test)),
            "training_time_sec": round(float(getattr(r, "training_seconds", 0.0)), 3),
        }
        for r in results
    ])
    leaderboard["r2_sort"] = leaderboard["r2"].fillna(-999)
    leaderboard = leaderboard.sort_values(["r2_sort", "rmse", "mae"], ascending=[False, True, True]).drop(columns=["r2_sort"]).reset_index(drop=True)
    best_name = leaderboard.loc[0, "model"]
    best = next(r for r in results if r.model_name == best_name)
    return RegressionTrainingOutput(results, best, leaderboard, skipped, target_column, target_summary)
