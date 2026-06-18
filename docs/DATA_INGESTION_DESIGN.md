# Data Ingestion Design

## Purpose

This version supports more than CSV upload because real organisational data may arrive through different sources.

Supported data sources:

- CSV file upload
- Excel file upload (`.xlsx` / `.xls`)
- JSON file upload
- public CSV/JSON API or URL

## Scope control

This does **not** change the dissertation topic.

The main dissertation evaluation remains focused on the **customer churn case study**. Multi-source upload is included as a robustness and usability feature to show how the Copilot could accept structured business data from different sources.

## What works for any source

After the data is loaded into a table, the same pipeline is used:

```text
data ingestion
→ table normalisation
→ data quality
→ dashboard
→ controlled Copilot
```

## What requires a binary target

Machine learning, prediction and SHAP explanation require a selected binary target column, for example:

- `Churn`: Yes / No
- `Attrition`: Yes / No
- `Purchased`: 1 / 0
- `Default`: True / False

## API limitation

The proof-of-concept supports public no-auth URLs only. Authenticated enterprise APIs are discussed in the large-scale architecture, not implemented in the MVP.
