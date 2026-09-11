# Explainable AI Copilot for Operational Business Decision Support

Proof-of-concept Streamlit application developed for the MSc dissertation **Design and Evaluation of a Proof-of-Concept Explainable AI Copilot for Operational Business Decision Support**.

## Core capabilities

- structured dataset upload and data-readiness assessment
- descriptive Business Intelligence-style analytics
- comparison of Logistic Regression, Random Forest, Gradient Boosting and MLP classifiers
- global and local SHAP explanations
- evidence-grounded conversational Copilot
- short-term session context and safe-refusal / limitation handling
- runtime, memory, audit and results export evidence
- structured 100-question Copilot evaluation

## Repository contents

```text
app.py                 Streamlit application
src/                   application modules
tests/                 automated regression tests
evaluation/            100-question instrument and questionnaire template
data_samples/          small sample churn dataset for quick testing
requirements.txt       core runtime dependencies
requirements_dev.txt   test dependency
requirements_optional.txt optional/future architecture dependencies
TEST_RESULTS.txt       packaged final regression-test summary
```

The repository intentionally excludes generated runtime files, exported result packages, local databases, caches, virtual environments and historical development notes. These are created locally when required and are covered by `.gitignore`.

## Run locally

### Windows quick start

Double-click `run_app.bat`, or run:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload a structured CSV dataset in the application. A small sample file is provided at `data_samples/sample_churn.csv` for a quick functional check.

## Run automated tests

Install the development dependency and run:

```bash
python -m pip install -r requirements_dev.txt
python -m pytest -q
```

The final verified code version completes **86 / 86 automated tests**.

## Optional dependencies

The assessed dissertation workflow uses the core dependencies only. Optional/future document-retrieval and enterprise-architecture packages are listed in `requirements_optional.txt` and are not required to run the core evaluated workflow.

## Runtime-generated files

The application may create local runtime artefacts such as `audit_store.db` and `.copilot_memory/`. Exported CSV/ZIP result packages are generated through the application when requested. These files are intentionally not stored in the GitHub repository.

## Research scope

The evaluated contribution is an integrated, evidence-controlled workflow combining readiness, descriptive analytics, prediction, explanation, conversational grounding, responsible refusal, traceability and export. The project does not claim a state-of-the-art churn model or autonomous business decision making.
