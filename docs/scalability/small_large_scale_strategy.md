# Small-Scale and Large-Scale Strategy

## Purpose

The supervisor requested improvement for both small-scale and large-scale operation. This document explains the two-level strategy used in the project.

## Small-scale proof-of-concept

The small-scale implementation is the fully working Streamlit prototype.

Recommended main case study:

```text
IBM/Telco Customer Churn
```

Small-scale workflow:

```text
Upload → clean → EDA → classification → SHAP/fallback explanation → Copilot response → recommendation → safety warning → evaluation
```

Evidence to collect:

- dataset upload screenshot;
- cleaning report screenshot;
- EDA charts;
- model leaderboard;
- SHAP/fallback explanation;
- Business Copilot recommendation;
- AI Copilot answer frame;
- evaluation questionnaire responses.

## Large-scale robustness testing

Use a larger structured business dataset such as:

- UCI Online Retail;
- UCI Online Retail II;
- bank marketing data;
- operations/service request data.

Large-scale local tests should focus on:

- file loading;
- row/column/memory profiling;
- cleaning and missing-value reporting;
- aggregated EDA;
- business insight generation;
- sampled model training where a target exists;
- exported scale evidence.

## Why sampling is acceptable locally

Streamlit is suitable for proof-of-concept interaction but not a distributed training engine. For very large datasets, the prototype uses sampling or aggregation for responsiveness. This is not a weakness if it is explained correctly: it shows the boundary between a local research prototype and an enterprise deployment.

## Enterprise-scale extension

A full large-scale version should use:

```text
Frontend UI
→ API gateway / FastAPI backend
→ data ingestion service
→ validation and cleaning service
→ database / data warehouse / data lake
→ model training service
→ model registry
→ explanation service
→ recommendation service
→ audit log
→ role-based access control
→ monitoring and drift detection
→ human approval workflow
```

## Dissertation wording

> The prototype is evaluated at two levels. First, the full explainable decision-support workflow is implemented and evaluated using a customer churn dataset. Second, larger structured business datasets are used to test robustness of ingestion, cleaning, EDA and business insight generation. For full enterprise-scale deployment, the dissertation proposes a scalable architecture separating the UI, API, data, model, explanation, recommendation, audit and governance layers.
