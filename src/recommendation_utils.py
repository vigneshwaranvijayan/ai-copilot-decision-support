from __future__ import annotations

import re
import pandas as pd
import plotly.express as px

from src.data_utils import grouped_positive_rate, positive_mask, is_binary_target
from src.nlp_utils import best_column_match, correction_prefix, interpret_question


BUSINESS_RULES = {
    "Contract": [
        "Consider offering annual-contract incentives or loyalty benefits for month-to-month customers.",
        "Review whether flexible contract customers need stronger retention engagement.",
    ],
    "tenure": [
        "Improve onboarding and proactive support for short-tenure customers.",
        "Send early check-in messages during the first months of the customer relationship.",
    ],
    "MonthlyCharges": [
        "Review price satisfaction and consider personalised bundle or loyalty offers.",
        "Check whether high charges are linked with dissatisfaction or low perceived value.",
    ],
    "TotalCharges": [
        "Review customer value history and prioritise high-value at-risk customers for human follow-up.",
    ],
    "OnlineSecurity": [
        "Offer a security-package trial or clearly explain the value of protection services.",
    ],
    "TechSupport": [
        "Provide proactive technical support calls or service-health checks.",
    ],
    "PaymentMethod": [
        "Review billing friction and encourage easier automated payment options where appropriate.",
    ],
    "InternetService": [
        "Review service quality, package fit and support experience for customers in high-risk service groups.",
    ],
}


KNOWN_ONE_HOT_PREFIXES = [
    "Contract",
    "PaymentMethod",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PhoneService",
    "MultipleLines",
    "OnlineBackup",
    "DeviceProtection",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "gender",
    "Partner",
    "Dependents",
]


def humanise_feature_name(feature_name: str) -> str:
    """
    Convert technical SHAP/encoded feature names into more readable business language.

    Examples:
    - Contract_Month-to-month -> Contract is Month-to-month
    - MonthlyCharges -> Monthly charges
    - OnlineSecurity_No -> Online security is No
    """
    raw = str(feature_name)
    raw = raw.replace("numeric__", "").replace("categorical__", "")

    # Some encoders return names such as Contract_Month-to-month.
    for prefix in KNOWN_ONE_HOT_PREFIXES:
        if raw.startswith(prefix + "_"):
            value = raw[len(prefix) + 1:].replace("_", " ")
            readable_prefix = re.sub(r"(?<!^)(?=[A-Z])", " ", prefix).strip().capitalize()
            return f"{readable_prefix} is {value}"

    # Convert camel case to words.
    readable = re.sub(r"(?<!^)(?=[A-Z])", " ", raw)
    readable = readable.replace("_", " ").strip()

    # A few specific business-friendly labels.
    replacements = {
        "Monthly Charges": "Monthly charges",
        "Total Charges": "Total charges",
        "tenure": "Customer tenure",
        "Senior Citizen": "Senior citizen status",
    }
    return replacements.get(readable, readable)


def humanise_top_features(shap_result, max_features: int = 5):
    if not shap_result or not shap_result.get("available"):
        return []

    readable_items = []
    top_df = shap_result["top_features"].head(max_features)

    for _, row in top_df.iterrows():
        feature = humanise_feature_name(row["feature"])
        impact = row.get("impact", "")
        readable_items.append(f"{feature} ({impact})")

    return readable_items


def _is_numeric_column(df: pd.DataFrame, col: str | None) -> bool:
    return bool(col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]))


def _display_columns(df: pd.DataFrame, target_col: str | None = None, key_col: str | None = None):
    preferred = ["customerID", "CustomerID", key_col, "Contract", "tenure", "MonthlyCharges", "TotalCharges", target_col]
    cols = []
    for col in preferred:
        if col and col in df.columns and col not in cols:
            cols.append(col)
    if not cols:
        cols = list(df.columns[:8])
    return cols


def _match_rules(feature_name: str):
    matches = []
    clean = str(feature_name).replace("numeric__", "").replace("categorical__", "").lower()

    for key, recs in BUSINESS_RULES.items():
        if key.lower() in clean:
            matches.extend(recs)

    return matches


def generate_plain_english_explanation(prediction_result, shap_result):
    risk_label = prediction_result["risk_label"]
    prob = prediction_result["positive_probability"] * 100
    target_col = prediction_result.get("target_col", "target")

    if shap_result and shap_result.get("available"):
        readable_features = humanise_top_features(shap_result, max_features=5)
        readable = "; ".join(readable_features)

        if str(target_col).lower() == "churn":
            return (
                f"The model predicts **{risk_label.lower()}** with a churn probability of **{prob:.2f}%**. "
                f"The main business-readable factors are: **{readable}**. "
                "This means the customer may need human review for possible retention support. "
                "The recommendation should be checked by a manager because the model shows patterns, not guaranteed causes."
            )

        return (
            f"The model predicts **{risk_label.lower()}** with a positive-class probability of **{prob:.2f}%** for target **{target_col}**. "
            f"The main readable factors are: **{readable}**. "
            "This is a data-grounded explanation and should be treated as decision support, not an automatic decision."
        )

    return (
        f"The model predicts **{risk_label.lower()}** with a probability of **{prob:.2f}%**. "
        "SHAP details are unavailable in this run, so the system cannot provide full feature-level explanation. "
        "The result can still be used as a decision-support signal with caution."
    )


def generate_recommendations(shap_result, selected_raw=None):
    recommendations = []

    if shap_result and shap_result.get("available"):
        for feature in shap_result["top_features"]["feature"].tolist():
            recommendations.extend(_match_rules(feature))

    if not recommendations:
        recommendations = [
            "Review the highest-risk or highest-value groups in the dashboard.",
            "Investigate the main data patterns before taking action.",
            "Use the model output as decision support and combine it with human context.",
        ]

    recommendations.append(
        "Safety warning: this is a decision-support recommendation only. A human user should review the context before action."
    )

    seen = set()
    unique_recs = []
    for item in recommendations:
        if item not in seen:
            unique_recs.append(item)
            seen.add(item)

    return unique_recs[:6]


def supported_question_examples():
    return [
        "Give me a dataset summary",
        "What columns are in this file?",
        "Check missing values",
        "Check duplicates",
        "Show distribution of Contract",
        "What is the average MonthlyCharges?",
        "What is the highest TotalCharges?",
        "What is the churn rate?",
        "Show churn by contract",
        "What should we improve?",
        "Why is this record high risk?",
    ]


def _distribution_chart(df: pd.DataFrame, column: str):
    if pd.api.types.is_numeric_dtype(df[column]):
        fig = px.histogram(df, x=column, title=f"Distribution of {column}")
        summary = f"Here is the numeric distribution for **{column}**. The average is approximately **{df[column].mean():.2f}**."
        return summary, fig, None

    counts = df[column].astype(str).value_counts().head(20).reset_index()
    counts.columns = [column, "count"]
    fig = px.bar(counts, x=column, y="count", title=f"Distribution of {column}", text="count")
    summary = f"Here is the distribution for **{column}**. The most common value is **{counts.iloc[0][column]}**."
    return summary, fig, counts


def answer_copilot_question(
    question,
    df,
    target_col=None,
    prediction_pack=None,
    explanation_pack=None,
):
    interpretation = interpret_question(question, df.columns)
    q = interpretation.corrected
    prefix = correction_prefix(interpretation)

    if not q:
        return {
            "text": "Please type a question such as: What columns are in this file? Check missing values. Show distribution of Contract.",
            "suggestions": supported_question_examples()[:5],
        }

    unsafe_terms = ["automatic decision", "auto decision", "replace manager", "decide alone", "make decision", "decide customer"]
    if any(term in q for term in unsafe_terms):
        return {
            "text": prefix + (
                "No. This prototype is designed for **decision support only**. It can explain patterns and suggest possible actions, "
                "but it must not replace human judgement or automatically decide treatment."
            )
        }

    if "missing" in q or "null" in q or "empty" in q:
        missing = df.isna().sum().reset_index()
        missing.columns = ["column", "missing_count"]
        missing = missing[missing["missing_count"] > 0].sort_values("missing_count", ascending=False)
        if missing.empty:
            return {"text": prefix + "I did not find missing values in the uploaded dataset."}
        return {"text": prefix + "These columns have missing values:", "table": missing}

    if "duplicate" in q or "duplicates" in q or "repeated" in q:
        dup = int(df.duplicated().sum())
        return {"text": prefix + f"The uploaded dataset contains **{dup} duplicate rows**."}

    if "columns" in q or "column" in q or "fields" in q or "field" in q:
        columns = list(df.columns)
        preview_cols = ", ".join([f"**{c}**" for c in columns[:30]])
        extra = "" if len(columns) <= 30 else f" and {len(columns) - 30} more columns"
        return {"text": prefix + f"The uploaded file contains these columns: {preview_cols}{extra}."}

    mentioned_col = best_column_match(q, df.columns)

    if "summary" in q or "overview" in q or "dataset" in q or "about data" in q:
        text = (
            f"This dataset contains **{len(df)} records** and **{df.shape[1]} columns**. "
            f"It has **{df.select_dtypes(include='number').shape[1]} numeric** columns and "
            f"**{df.select_dtypes(exclude='number').shape[1]} categorical/text** columns. "
            f"There are **{int(df.isna().sum().sum())} missing cells** and **{int(df.duplicated().sum())} duplicate rows**."
        )
        if target_col and target_col in df.columns and is_binary_target(df[target_col]):
            rate = positive_mask(df[target_col]).mean() * 100
            text += f" The selected binary target is **{target_col}**, with a positive rate of approximately **{rate:.2f}%**."
        return {"text": prefix + text}

    if "distribution" in q or "show" in q or "chart" in q or "graph" in q:
        asks_for_target = target_col and target_col in df.columns and (str(target_col).lower() in q or "churn" in q or "rate" in q)
        asks_by_group = any(word in q for word in ["by", "compare", "group", "segment"]) or ("rate" in q and mentioned_col)

        if asks_for_target and asks_by_group and is_binary_target(df[target_col]):
            group_col = mentioned_col
            if not group_col or group_col == target_col or "id" in str(group_col).lower():
                non_target_cols = [c for c in df.columns if c != target_col and df[c].nunique(dropna=True) <= 30]
                group_col = non_target_cols[0] if non_target_cols else None

            if group_col:
                grouped = grouped_positive_rate(df, target_col, group_col)
                if not grouped.empty:
                    group_label_col = grouped.columns[0]
                    fig = px.bar(
                        grouped,
                        x=group_label_col,
                        y="positive_rate",
                        title=f"{target_col} positive rate by {group_col}",
                        text=grouped["positive_rate"].round(2),
                    )
                    fig.update_layout(yaxis_title=f"{target_col} positive rate (%)")
                    top = grouped.iloc[0]
                    return {
                        "text": prefix + (
                            f"The highest observed **{target_col}** positive rate by **{group_col}** is for **{top[group_label_col]}**, "
                            f"with approximately **{top['positive_rate']:.2f}%**."
                        ),
                        "chart": fig,
                        "table": grouped,
                    }

        if mentioned_col and mentioned_col in df.columns:
            summary, fig, table = _distribution_chart(df, mentioned_col)
            ans = {"text": prefix + summary, "chart": fig}
            if table is not None:
                ans["table"] = table
            return ans

    wants_top = any(word in q for word in ["highest", "top", "maximum", "max", "most"])
    wants_low = any(word in q for word in ["lowest", "minimum", "min", "least"])
    wants_average = any(word in q for word in ["average", "mean", "avg"])

    if mentioned_col and mentioned_col in df.columns and pd.api.types.is_numeric_dtype(df[mentioned_col]) and (wants_top or wants_low):
        ascending = wants_low
        cols = _display_columns(df, target_col, mentioned_col)
        table = df.sort_values(mentioned_col, ascending=ascending).head(5)[cols].reset_index(drop=True)
        direction = "lowest" if wants_low else "highest"
        value = table.iloc[0][mentioned_col]
        return {
            "text": prefix + f"The **{direction}** value found for **{mentioned_col}** is **{value}**. Here are the top matching records.",
            "table": table,
        }

    if mentioned_col and mentioned_col in df.columns and pd.api.types.is_numeric_dtype(df[mentioned_col]) and wants_average:
        avg = df[mentioned_col].mean()
        return {"text": prefix + f"The average **{mentioned_col}** is approximately **{avg:.2f}**."}

    if target_col and target_col in df.columns and is_binary_target(df[target_col]) and ("rate" in q or "percentage" in q or "how many" in q or str(target_col).lower() in q or "churn" in q):
        rate = positive_mask(df[target_col]).mean() * 100
        positive_count = int(positive_mask(df[target_col]).sum())
        return {
            "text": prefix + (
                f"The positive rate for **{target_col}** is approximately **{rate:.2f}%** "
                f"(**{positive_count}** positive cases out of **{len(df)}** records)."
            )
        }

    if "driver" in q or "drivers" in q or "reason" in q or "reasons" in q or "factor" in q or "factors" in q:
        if explanation_pack and explanation_pack.get("shap_result", {}).get("available"):
            top_features = explanation_pack["shap_result"]["top_features"].copy()
            top_features["plain_english_feature"] = top_features["feature"].apply(humanise_feature_name)
            return {
                "text": prefix + "The current selected prediction is mainly influenced by these SHAP drivers:",
                "table": top_features,
            }
        return {
            "text": prefix + (
                "To show model drivers, first choose a binary target, train the model, then go to **Prediction & SHAP Explanation**, select a record, and click **Predict and Explain**."
            )
        }

    if "why" in q or "explain" in q or "explanation" in q:
        if explanation_pack:
            return {"text": prefix + explanation_pack.get("explanation_text", "No explanation is available yet.")}
        return {
            "text": prefix + (
                "For an individual model explanation, first train a binary classification model, then go to **Prediction & SHAP Explanation**, select a row, and click **Predict and Explain**."
            )
        }

    if "recommend" in q or "improve" in q or "what should" in q or "solution" in q or "action" in q:
        recs = explanation_pack.get("recommendations", []) if explanation_pack else generate_recommendations(shap_result=None)
        return {
            "text": prefix + "Based on the current analysis, these actions are suggested:",
            "recommendations": recs,
        }

    if mentioned_col:
        examples = [f"Show distribution of {mentioned_col}"]
        if pd.api.types.is_numeric_dtype(df[mentioned_col]):
            examples += [f"Highest {mentioned_col}", f"Average {mentioned_col}"]
        if target_col and target_col in df.columns and target_col != mentioned_col:
            examples.append(f"Show {target_col} by {mentioned_col}")

        return {
            "text": prefix + (
                f"I noticed the column **{mentioned_col}**, but I need a clearer action. Try one of these questions."
            ),
            "suggestions": examples,
        }

    return {
        "text": prefix + (
            "I did not fully understand that question, but I can help with data-grounded analysis of the uploaded CSV. "
            "Try one of the suggested questions below."
        ),
        "suggestions": supported_question_examples(),
    }
