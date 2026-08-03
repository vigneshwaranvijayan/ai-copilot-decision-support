# v15.3 Reviewer Stability Change Log

## Why this version was needed

A Copilot chat export was reviewed and showed that several high-standard reviewer questions were not answered well enough. The problems were not with the dataset upload or modelling itself; the problems were mainly answer routing and supported system/meta question handling.

## Issues found in the uploaded chat export

- Tenure/payment-method relationship questions returned only the overall churn rate.
- Some memory, audit, MongoDB and graph-grounding questions were refused as unsupported.
- Legal-advice requests were refused too generically instead of explicitly.
- Retention-action questions were too generic.
- Top-driver questions could be captured by model-performance logic before explanation-driver logic.

## Fixes implemented

1. Feature-vs-target relationship routing
   - Numeric examples: tenure vs churn
   - Categorical examples: payment method vs churn
   - Output: positive-rate table and chart

2. Reviewer/system question handlers
   - short-term memory
   - audit trail
   - answer logging
   - MongoDB removal / ChromaDB reasoning
   - graph/source grounding
   - dataset RAM consumption

3. Safety refusal
   - Legal-advice questions now receive an explicit refusal and human-review message.

4. Retention-action routing
   - Retention priority questions now use churn segment evidence, feedback evidence and model evidence when available.

5. Prediction evidence
   - Model metrics, selected target and SHAP/fallback explanation status are returned for prediction-evidence questions.

6. Driver intent priority
   - Top-driver/risk-factor questions are checked before generic model-performance questions.

## Test result

- `python -m py_compile app.py src/*.py`: passed
- `pytest -q`: 40 passed

## Recommended demo workflow

1. Upload Telco churn dataset.
2. Let auto-modelling run.
3. Open Model page and generate SHAP/fallback explanation.
4. Ask the v15.3 screenshot questions.
5. Export chat answers and audit evidence.
