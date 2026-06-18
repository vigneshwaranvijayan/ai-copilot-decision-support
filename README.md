# Proof-of-Concept Explainable AI Copilot for Operational Business Decision Support

This repository contains the refined MSc dissertation prototype for:

**Design and Evaluation of a Proof-of-Concept Explainable AI Copilot for Operational Business Decision Support**

## Scope refinement

The original project idea was broad: a scalable AI Copilot for business and employee data analytics.  
Following supervisor feedback, the implementation is now focused as a manageable proof-of-concept.

The core topic is still the same: **an explainable AI Copilot for business analytics and decision support**.

The implementation now prioritises:

- Customer churn as the **primary case study**
- Prediction-only vs prediction + explanation + recommendation as the **main evaluation contribution**
- Controlled, data-grounded Copilot outputs instead of an unrestricted chatbot
- Large-scale enterprise architecture as a design/discussion contribution, not a full deployment

Employee attrition and wider operational collaboration scenarios are retained as discussion/future-extension cases only.

## Core workflow

```text
CSV upload
→ data validation
→ exploratory dashboard
→ churn prediction
→ SHAP explanation
→ plain-English Copilot translation
→ recommendation + safety warning
→ evaluation/report evidence
```

## Copilot approach

The Copilot component is a controlled hybrid layer. It uses:

- calculated dataset statistics
- prediction score
- SHAP top drivers
- predefined explanation templates
- recommendation rule matrix
- typo/fuzzy matching for simple user questions
- safety warnings

It does **not** rely on an unrestricted LLM to invent answers or make decisions.

## Install and run

### Windows

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
streamlit run app.py
```

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
streamlit run app.py
```

## Recommended dataset

Use the **Telco Customer Churn** dataset or another public churn dataset with a `Churn` target column.

Expected target column:

```text
Churn
```

## Supported Copilot questions

Examples:

```text
What is the churn rate?
Show churn by contract
Show churn by monthly charges
Check missing values
Check duplicates
What should we improve?
Why is this customer high risk?
Can this system make automatic decisions?
```

The Copilot includes basic typo correction, for example:

```text
shw chrun by contrct
what colums in file
check missng values
```

## Repository evidence plan

Use this repository as dissertation evidence:

- weekly commits
- issue tracking
- screenshots after milestones
- architecture notes
- testing notes
- evaluation questionnaire and anonymised results
- final prototype code

Suggested GitHub folders are already included:

```text
/docs
/docs/architecture
/docs/screenshots
/evaluation
/tests
/data_samples
/notebooks
/src
```

## Important warning

This is a proof-of-concept decision-support system. It should support human judgement, not replace it.


## Version 2 improvement

This version improves the Copilot and UI:

- better Streamlit header and scope chips
- quick question buttons
- grammar-cleaner Copilot answers
- stronger spelling/typo correction
- fuzzy matching for question words and dataset columns
- fallback suggestions when a question is unclear
- documentation of the controlled Copilot intent design in `docs/COPILOT_INTENT_DESIGN.md`

This keeps the project aligned with the refined dissertation scope: a proof-of-concept explainable AI Copilot for operational business decision support.


## Version 3 flexible upload update

This version allows users to upload **any CSV**.

For any dataset, the app can provide:

- data quality summary
- dashboard charts
- column list
- missing values
- duplicate checks
- distribution charts
- average and highest numeric values
- controlled Copilot answers based on uploaded data

For machine learning, prediction and SHAP explanation, the user must choose a **binary target column** in the sidebar. This keeps the prototype safe and realistic while preventing errors when a random dataset is uploaded.


## Final polish update

This version includes the final strengthening points before GitHub/supervisor review:

- Flexible CSV upload is clearly labelled as a robustness feature only.
- The dissertation evaluation remains focused on the customer churn case study.
- Human-readable SHAP explanations are improved so encoded feature names are translated into clearer business language.
- More tests are added for data preparation, model training, Copilot answers, recommendation generation and report building.
- Evaluation task design and RQ2 measurement mapping are documented in `docs/EVALUATION_TASK_AND_RQ_MAPPING.md`.
- The operational collaboration case study is documented in `docs/OPERATIONAL_COLLABORATION_CASE_STUDY.md`.
- `__pycache__` and `.pyc` files are removed before creating the GitHub-ready ZIP.


## Version 5 multi-source data ingestion

This version supports more realistic data input options:

- CSV upload
- Excel upload (`.xlsx` / `.xls`)
- JSON upload
- public CSV/JSON API or URL

This is a **robustness feature**, not a new dissertation scope. The main evaluated case study remains customer churn decision support. After the source is loaded into a table, the same dashboard, data quality, controlled Copilot, and optional binary-target ML/SHAP pipeline are used.
