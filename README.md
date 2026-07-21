
## v13.3 UI and workflow upgrade

This version upgrades the interface based on supervisor/testing feedback:

- Replaced crowded top title and main tab bar with a compact dashboard and button-style navigation.
- Dashboard now focuses on dataset readiness, key metrics, quick model status and automatic visual preview.
- Combined related pages into clearer workflow sections:
  - Cleaning & EDA
  - Modeling & Prediction Explanation
  - Business Improvements
  - Chat
  - Evaluation
  - Export Data
  - Visualization
- Improved Copilot chat presentation with latest-first answers and fixed bottom chat input.
- Added export of Copilot chat questions, answers, warnings and evidence row counts.
- Kept automatic modelling once after upload and manual model retraining.
- Kept Google Drive/URL/API/PostgreSQL/MongoDB integration support.
- Fixed Streamlit sidebar collapse/expand usability by not hiding the Streamlit header element required for sidebar controls.


# Dataset-Grounded Explainable AI Copilot for Operational Business Decision Support

## v13.2 QA and reviewer-readiness fixes

This version improves the Copilot's reliability during supervisor/reviewer testing:

- Refuses bank-marketing/campaign questions when the active dataset is actually Telco churn.
- Returns a clear FAIL-style explanation when requested evidence such as `balance` is missing.
- Answers `target column` questions directly from the selected modelling target.
- Routes churn-risk factor questions to SHAP/fallback model drivers instead of simple target distribution.
- Adds a decision-support suitability verdict for model-quality questions.
- Adds numeric-to-target relationship evidence for questions such as `does balance affect campaign response`.
- Tightens customer-feedback concern detection so neutral service/charge words are not counted as negative by themselves.


## v13.1 Feedback Evidence Fix

This update improves the customer-feedback Copilot answer quality. Negative feedback evidence is now stricter and more explainable: the system avoids treating broad service words as negative by default, avoids counting standalone "not" as negative, and adds a clearer concern-evidence trace. Feedback improvement answers now show the top improvement priorities directly in the visible Copilot response, followed by the detailed evidence table and chart.

Validation evidence: `python -m py_compile app.py src/*.py` passed and `python -m pytest -q` returned 25 passed.

## Project title

**Design and Evaluation of a Dataset-Grounded Explainable AI Copilot for Operational Business Decision Support**

This repository contains a research-gap driven Streamlit prototype for an Explainable AI Copilot that supports operational business decision-making from uploaded structured datasets. It combines data ingestion, automated cleaning, exploratory data analysis, predictive modelling, SHAP/fallback explainability, controlled Copilot responses, business improvement recommendations, safety warnings, evaluation evidence and scale-readiness documentation.

The project is designed for an MSc dissertation. The **main evaluated case study remains customer churn**, while sales/retail, bank marketing, employee/HR, feedback and operations datasets are used only as robustness and wider-relevance tests.

---

## Supervisor feedback addressed in v12.2

This version explicitly addresses the research concerns raised during supervision:

1. **Market comparison:** the project does not claim that AI analytics Copilots are completely new. It compares against commercial BI Copilot-style tools and academic XAI/conversational analytics research.
2. **Research gap:** the dissertation gap is now framed around the missing integration of dataset-grounded analytics, explainable prediction, action recommendation, safety warnings and human evaluation in one reproducible workflow.
3. **40+ paper evidence:** `docs/research/literature_review_matrix.md` includes a structured 50-source literature matrix across XAI, human-AI decision support, conversational analytics, AutoML/data quality, decision-support systems and responsible AI.
4. **Novelty:** the contribution is not a new ML algorithm; it is a controlled, explainable, dataset-grounded and evaluable decision-support workflow.
5. **Small-scale and large-scale readiness:** the app now includes a **Scale Readiness** tab and supporting documentation for local prototype testing and enterprise-scale extension.

---

## Research gap and contribution

### Main research gap

Existing BI Copilot platforms and academic XAI tools often address one part of the problem: dashboard generation, natural-language querying, model explanation or automated model selection. The gap targeted here is the **integration gap**:

> How can uploaded structured business data be converted into understandable, explainable, action-oriented and safety-aware decision support for non-technical users in a controlled and reproducible prototype?

### Contribution

This dissertation contributes a proof-of-concept framework that integrates:

- dataset upload from CSV, Excel, JSON, ZIP and public API/URL sources;
- automatic cleaning and data profiling;
- modern interactive EDA;
- classification-first supervised modelling for churn-style business outcomes;
- optional regression robustness mode for numeric targets such as sales or revenue;
- SHAP/fallback explanations;
- controlled Copilot responses grounded only in uploaded data and model artefacts;
- business recommendation and improvement suggestions;
- human-review safety warnings;
- a built-in evaluation workspace;
- small-scale and large-scale readiness documentation.

The system is therefore positioned as a **research prototype**, not as a direct commercial competitor to Power BI, Tableau or ThoughtSpot.

---

## Main workflow

```text
Upload CSV / Excel / JSON / ZIP / public API URL
→ automatic data cleaning
→ data profiling and EDA
→ detect target type
→ train classification or optional regression models
→ compare model performance
→ explain prediction with SHAP/fallback feature importance
→ controlled AI Copilot response
→ business improvement suggestions
→ human-review safety warning
→ evaluation evidence export
→ scale-readiness evidence export
```

---

## Supported input types

The app supports:

- CSV
- TSV / TXT table files
- Excel (`.xlsx`, `.xls`)
- JSON
- ZIP files containing CSV/Excel/JSON files
- Public CSV/JSON/Excel/API URLs

Multiple datasets can be loaded. The user selects an **active dataset** in the Dataset Hub. Copilot answers are restricted to the selected active dataset.

---

## Dataset strategy

### Main evaluated dataset

Use a customer churn dataset such as IBM/Telco Customer Churn for the full dissertation workflow:

```text
cleaning → EDA → classification → SHAP → Copilot explanation → recommendation → safety warning → evaluation
```

### Robustness datasets

Other datasets can be used to demonstrate wider business relevance and scale readiness:

| Dataset type | Example use | Model mode |
|---|---|---|
| Customer churn | churn risk, retention actions | Classification |
| Bank marketing | campaign response prediction | Classification |
| Employee/HR public or synthetic data | attrition/support patterns | Classification or EDA |
| Sales/retail data | product trends and stock suggestions | EDA + optional regression |
| Feedback/comment data | worst feedback and improvement themes | Text/EDA + business insight |
| Operations/service data | complaint or ticket improvement | EDA + optional classification |
| Large retail transaction data | large upload, cleaning, aggregation and sales insight | Large-scale robustness |

The dissertation should state that additional datasets are used for robustness and demonstration only. The primary evaluation remains customer churn.

---

## Model modes

| Target type | System mode | Example | Dissertation status |
|---|---|---|---|
| Yes/No, 0/1, True/False | Classification | Churn, Attrition, Exited, `y` | Main implemented and evaluated workflow |
| Continuous numeric | Regression | Sales, revenue, profit, delivery time | Optional robustness mode |
| No target column | EDA + Business Copilot | Retail transactions, feedback, operations logs | Implemented insight workflow |
| Action/reward history | RL/bandit extension | Retention action feedback | Future work only |

Reinforcement learning is not implemented as a main model because the public/static datasets used in the project do not normally contain real action-reward feedback loops. It is discussed as a future enterprise extension.

---

## Classification models

The classification workflow trains a controlled set of candidate models on the uploaded dataset:

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- Optional XGBoost, if installed

The best model is selected using a balanced metric priority:

```text
F1-score → ROC-AUC → Recall → Precision
```

This avoids selecting a model that only has high recall but poor precision.

---

## Optional regression models

For numeric target columns, the app can optionally train:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- Optional XGBoost Regressor, if installed

Regression metrics:

- MAE
- RMSE
- R²

Regression is included as robustness functionality for sales/revenue/profit-style datasets. It is not the main evaluated workflow.

---

## Explainability

The Explain page uses SHAP where possible. If SHAP fails or is not suitable for the selected model, the system falls back to feature importance or model-supported coefficients where available.

The explanation output supports:

- global feature drivers;
- local row-level drivers;
- plain-English driver sentences;
- Copilot recommendations based on drivers;
- safety warnings.

---

## AI Copilot response frame

Every Copilot answer is framed in a consistent academic format:

```text
Question understood as:
Dataset used:
Columns/artefacts used:
Evidence from the uploaded data:
Business / decision-support use:
Evidence format:
Safety warning:
```

This makes the Copilot easier to evaluate and helps show that responses are grounded in the uploaded data.

---

## Business Insight Engine

The Business Copilot tab provides domain-aware suggestions using only the active dataset.

Supported domains include:

- customer churn and feedback;
- sales and retail;
- employee/HR public or synthetic data;
- bank marketing;
- operations and service datasets;
- general structured business datasets.

Example questions:

```text
how to improve customerfeedback?
who gave worst customerfeedback?
which products should we purchase more?
which products are declining?
employee support and retention suggestions
marketing campaign suggestions
operations improvement suggestions
```

The system provides suggestions, but it does **not** make final business, financial or HR decisions automatically.

---

## Visual Analytics dashboard

The v12.2 version adds a **Visual Analytics** tab so the prototype can be demonstrated more like a modern analytics product while staying academically controlled.

The dashboard includes:

- executive KPI cards;
- a data-quality gauge;
- missing-value profile chart;
- column-type mix chart;
- target distribution chart where a target exists;
- domain-specific charts for churn, customer feedback, sales/retail, bank marketing, HR and operations datasets;
- correlation heatmaps and numeric distributions where appropriate;
- chart evidence tables;
- scale-aware notes for large datasets;
- a **custom visual builder** where users can choose X/category/date fields, numeric Y fields, aggregation, optional group/colour field and chart type.

This does not claim to replace Power BI or Tableau. The research contribution is that each visual output is linked to uploaded-data evidence, model explanations, Copilot interpretation, business suggestions and safety warnings.

---

## Research gap evidence

Research-gap and novelty material is kept in the GitHub documentation rather than shown as a main app tab. This keeps the live prototype focused on user-facing decision support while still providing dissertation evidence.

Key files:

```text
docs/research/literature_review_matrix.md
docs/research/research_gap_and_novelty.md
docs/research/market_comparison.md
```

---

## Small-scale and large-scale readiness

The app includes a **Scale Readiness** tab.

### Small-scale implementation

Small-scale evaluation uses the customer churn workflow:

```text
Telco churn dataset → full ML + SHAP + Copilot + user evaluation
```

### Large-scale robustness

Large datasets such as UCI Online Retail or Online Retail II can be used to test:

- upload and data loading;
- cleaning and profiling;
- memory/row/column reporting;
- aggregated EDA;
- sampled model training where relevant;
- business insight generation;
- exportable scale evidence.

### Enterprise-scale design

The dissertation should explain how the prototype can scale into:

```text
Frontend UI
→ API gateway / FastAPI backend
→ data ingestion and validation service
→ database / data lake / warehouse
→ model training service
→ explanation service
→ business recommendation service
→ audit log + RBAC + monitoring + human approval layer
```

This keeps the implementation realistic while showing large-scale design competence.

---

## Evaluation workspace

The app includes a built-in evaluation section for the dissertation.

It supports the intended comparison:

| Condition | Output shown to participant |
|---|---|
| Condition A | Prediction-only output |
| Condition B | Prediction + explanation + Copilot recommendation + safety warning |

The questionnaire captures:

- understanding;
- trust;
- usefulness;
- usability;
- decision confidence;
- safety awareness;
- optional comments.

Responses can be downloaded as CSV for analysis.

---

## Folder structure

```text
ai_copilot_v12_2_visual_builder_fixed/
├── app.py
├── README.md
├── requirements.txt
├── run_app.bat
├── .gitignore
├── src/
│   ├── cleaning.py
│   ├── copilot.py
│   ├── business_insights.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── explainability.py
│   ├── modeling.py
│   ├── report.py
│   ├── visual_analytics.py
│   └── text_utils.py
├── tests/
├── docs/
│   ├── architecture/
│   ├── research/
│   ├── scalability/
│   ├── testing/
│   └── screenshots/
├── evaluation/
└── data_samples/
```

---

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Or on Windows, double-click:

```text
run_app.bat
```

Run tests:

```bash
pytest -q
```

---

## GitHub guidance

Do not commit large raw datasets. Keep only small samples in GitHub.

Recommended `.gitignore` behaviour:

```text
data_raw/
*.zip
*.xlsx
__pycache__/
.pytest_cache/
```

Use GitHub for:

- regular commits;
- code evidence;
- screenshots;
- architecture diagrams;
- testing evidence;
- evaluation materials;
- implementation decisions;
- literature matrix and research-gap evidence.

Suggested commit message:

```bash
git add .
git commit -m "Add v12.2 visual builder and feedback issue analysis"
git push
```

---

## Academic limitations

The dissertation should state these limitations clearly:

- The implemented system is a proof-of-concept, not a production enterprise platform.
- The main evaluated workflow is customer churn classification.
- Regression is optional robustness functionality.
- Reinforcement learning is future work because static public datasets lack real action-reward feedback.
- Large datasets may need sampling for responsive Streamlit training.
- HR-related analysis must use public/synthetic/anonymised datasets only.
- Business recommendations are decision-support suggestions, not automatic decisions.
- External market trends are not included unless the user uploads relevant external data.

---

## Recommended dissertation wording

> This project develops a dataset-grounded Explainable AI Copilot for operational business decision support. The contribution is not a new machine-learning algorithm or a direct commercial replacement for BI Copilot platforms. Instead, the project investigates a controlled and reproducible workflow that integrates uploaded-data preparation, exploratory analysis, predictive modelling, explainable AI, plain-English Copilot interpretation, business-action recommendation, safety warnings and user evaluation. Customer churn is used as the primary evaluated case study, while additional datasets demonstrate robustness across business scenarios and support a small-to-large-scale design discussion.

## v12.2 test result

```text
25 passed
```


## v13 Data-Integrated Dataset-Grounded Upgrade

This version keeps the previous Explainable AI Copilot features and adds the new supervisor-aligned architecture:

- Local upload support: CSV, TSV, Excel, JSON and ZIP.
- Public URL/API support: CSV, JSON and Excel links.
- Google Drive / Google Sheets shared-link import.
- Optional PostgreSQL and MongoDB connectors for live integrated data.
- Dataset readiness gates with PASS / WARNING / FAIL status.
- Data-quality score and readiness evidence shown before modelling and Copilot answers.
- Automatic classification modelling once after data load when a binary target is detected.
- Manual classification and regression modelling remain available.
- Modern visual analytics with automatically selected graphs and a manual custom visual builder.
- Dataset-grounded Copilot answers with question understanding, data source, readiness status, columns used, evidence, limitations and safety warning.
- Business improvement suggestions generated from uploaded/integrated dataset evidence.
- Session and SQLite audit-log export for dissertation evidence.

The main evaluated case study remains customer churn. Bank marketing, online retail/sales and customer feedback datasets are used as integration, robustness and scalability evidence.
