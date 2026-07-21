# v13 Data-Integrated Dataset-Grounded Architecture

This upgrade keeps the previous customer-churn proof-of-concept and adds the supervisor-requested data-integration and evidence-grounding layer.

## Main change
The Copilot should not answer from generic text. It must:

1. identify the active data source,
2. validate the dataset readiness,
3. retrieve evidence from the integrated dataset, EDA summaries, model outputs and explanation artefacts,
4. produce an answer with evidence, recommendation, limitation and warning,
5. log the interaction for evaluation evidence.

## Supported source types
- Local upload: CSV, TSV, Excel, JSON and ZIP.
- Public URL/API: CSV, JSON and Excel links.
- Google Drive / Google Sheets: public or shared file links.
- PostgreSQL: optional live structured-data connector.
- MongoDB: optional document/collection connector.

## Proof-of-concept scope
Customer churn remains the primary evaluated case study. Bank marketing, online retail/sales and customer feedback datasets are used as robustness and integration evidence.

## PASS / WARNING / FAIL gates
The readiness layer checks data availability, schema readability, missing-value quality, duplicate-row quality, model-target readiness and security/privacy signals.

## Auto-modelling behaviour
When a binary target is detected, classification training runs once after data load and is cached. Manual retraining remains available in the Models tab.

## Traceability
Each Copilot answer includes dataset, readiness status, columns/artefacts used, evidence, business use, limitation and safety warning. Audit exports are available in the Export tab.
