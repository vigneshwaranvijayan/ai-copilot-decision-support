# Explainable AI Copilot for Business Decision Support

## Design and Evaluation of a Proof-of-Concept Explainable AI Copilot for Operational Business Decision Support

This repository contains a Streamlit-based MSc dissertation prototype for an **Explainable AI Copilot** that supports operational business decision-making using structured datasets. The prototype allows users to upload business data, clean and explore the dataset, train binary classification models, generate explainable AI outputs, and receive controlled, data-grounded business recommendations.

The main evaluated case study is **customer churn decision support**. Additional datasets such as sales, retail, bank marketing, employee attrition, customer feedback and operational service data are used only as robustness demonstrations to show that the ingestion, EDA, Copilot and business-insight layers can work across different structured-data scenarios.

---

## Project Aim

The aim of this project is to design, implement and evaluate a proof-of-concept Explainable AI Copilot that helps non-technical business users understand data, model predictions and possible operational actions. The system is not designed to replace human judgement. It provides decision support by combining:

- automatic data ingestion and cleaning;
- exploratory data analysis;
- machine-learning model comparison;
- explainable AI using SHAP or fallback feature importance;
- controlled Copilot-style question answering;
- data-grounded business insight and recommendation generation;
- safety warnings and human-review reminders.

---

## Research Scope

### Main dissertation scope

The core evaluation remains focused on **customer churn prediction and explanation**. The main workflow is:

```text
Upload customer churn dataset
→ clean and validate data
→ perform exploratory analysis
→ train and compare classification models
→ select the best model
→ generate SHAP or feature-importance explanation
→ translate results into plain-English Copilot output
→ produce recommendation and safety warning
→ evaluate prediction-only vs explanation-supported outputs
```

### Robustness scope

The prototype can also be tested with other structured business datasets, for example:

- sales and retail transaction data;
- bank marketing campaign data;
- customer feedback data;
- employee attrition or HR data;
- operational service or complaint data.

These additional datasets are used to test ingestion, cleaning, EDA, business-insight generation and data-grounded Copilot responses. They should not replace the main churn case study unless the project scope is formally changed.

---

## Key Features

### Dataset Hub

- Upload one or more datasets.
- Supported file types:
  - CSV
  - TSV
  - Excel
  - JSON
  - ZIP files containing supported structured data files
- Load public CSV, JSON, Excel or API URL data.
- Select an active dataset for analysis.
- Optional merge by shared key, such as `customerID`, `CustomerID`, `id` or another common column.
- Dataset preview and basic metadata summary.

### Automatic Data Cleaning

The cleaning pipeline supports:

- column-name normalisation;
- duplicate-row detection;
- missing-value profiling;
- text trimming;
- numeric conversion where appropriate;
- handling of common invalid values;
- cleaned-data export.

### Exploratory Data Analysis

The EDA module generates:

- dataset shape summary;
- missing-value table;
- numeric summary statistics;
- categorical summaries;
- target distribution where a binary target exists;
- group comparisons;
- interactive charts;
- large-dataset-friendly summaries.

### Model Training and Comparison

The modelling module is designed for **binary classification** datasets. It can train and compare:

- Logistic Regression;
- Random Forest;
- Extra Trees;
- Gradient Boosting;
- optional XGBoost, if installed.

The app selects the best model using a controlled comparison based on:

```text
F1-score → ROC-AUC → Recall → Precision
```

This avoids selecting a model only because it has high recall while producing too many false positives.

### Explainability

The explainability module provides:

- SHAP explanation where possible;
- fallback feature importance when SHAP is unavailable or unsuitable;
- local/customer-level explanation;
- global model driver explanation;
- plain-English interpretation of technical outputs.

### Controlled Copilot Chat

The Copilot is not an unrestricted chatbot. It uses a controlled, data-grounded intent system. It supports common business-analysis question types such as:

- missing values;
- dataset summary;
- highest and lowest values;
- category distributions;
- group comparisons;
- target-rate analysis;
- model performance;
- feature importance;
- customer-level explanation;
- worst or best feedback ranking;
- business improvement suggestions;
- follow-up questions using recent context.

If the question cannot be safely mapped to the uploaded dataset, the system should ask for clarification or suggest available columns instead of inventing unsupported answers.

### Business Insight Engine

The Business Insight Engine generates recommendations based only on the active uploaded dataset.

Supported insight areas include:

#### Customer churn and feedback

Examples:

- identify churn-risk patterns;
- find customers or rows with negative feedback;
- summarise feedback themes;
- suggest improvements for pricing, onboarding, service quality, support, package fit or retention actions.

#### Sales and retail

Examples:

- identify strong-selling products;
- detect declining products;
- compare recent and previous sales periods;
- suggest stock or purchasing priorities;
- identify products that performed well previously but are currently weaker.

#### Bank marketing

Examples:

- identify customer groups with higher positive response rates;
- support campaign targeting;
- highlight useful segment patterns.

#### Employee or HR data

Examples:

- identify groups with higher attrition risk;
- highlight workload or satisfaction concerns;
- suggest manager review or support actions.

Important: HR outputs must never be treated as final employment decisions. They are review signals only.

#### Operations and service data

Examples:

- identify high-volume complaint categories;
- identify service hotspots;
- suggest operational review areas.

#### General business data

If the dataset type is unclear, the app provides general improvement suggestions based on:

- missing values;
- duplicates;
- dominant categories;
- unusual numeric patterns;
- possible target columns;
- columns suitable for deeper analysis.

---

## Important Safety Boundary

This prototype provides **decision support only**.

It must not be used to make automatic decisions about customers, employees, pricing, finance, marketing or operations. All outputs require human review.

The system should avoid wording such as:

```text
Fire this employee.
This person is useless.
Automatically reject this customer.
Automatically remove this product.
```

Instead, it should use safer decision-support wording such as:

```text
This record may require management review.
This customer segment may need a retention action.
This product may need stock or promotion review.
This employee-related pattern should be reviewed by HR and the relevant manager.
```

---

## Repository Structure

```text
.
├── app.py                         # Main Streamlit app
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Files and folders excluded from Git
│
├── src/
│   ├── data_loader.py             # File, ZIP and URL/API loading
│   ├── cleaning.py                # Data cleaning and validation
│   ├── eda.py                     # Exploratory data analysis
│   ├── modeling.py                # ML model training and comparison
│   ├── explainability.py          # SHAP and fallback explanations
│   ├── copilot.py                 # Controlled Copilot intent engine
│   ├── business_insights.py       # Dataset-grounded business suggestions
│   ├── report.py                  # Markdown report export
│   └── __init__.py
│
├── tests/
│   ├── test_cleaning.py
│   ├── test_data_loader.py
│   ├── test_modeling.py
│   └── test_copilot.py
│
├── docs/
│   ├── architecture/
│   │   └── architecture_overview.md
│   └── screenshots/
│       └── add_app_screenshots_here.png
│
├── evaluation/
│   ├── README.md
│   └── questionnaire_template.csv
│
├── data_samples/
│   ├── tiny_churn_sample.csv
│   └── tiny_sales_sample.csv
│
└── data_raw/
    └── keep_full_datasets_locally_only.txt
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-copilot-decision-support.git
cd ai-copilot-decision-support
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app should open locally at:

```text
http://localhost:8501
```

---

## Running Tests

Run:

```bash
pytest -q
```

Expected result for the current version:

```text
22 passed
```

---

## Suggested Datasets

### Main evaluation dataset

Use a customer churn dataset with a clear binary target, for example:

- Telco Customer Churn dataset
- Target column: `Churn`
- Positive class: usually `Yes` or `1`, depending on the dataset version

This dataset is used for:

```text
model training
SHAP explanation
Copilot interpretation
recommendation generation
user evaluation
```

### Robustness datasets

These are optional and should be used only to test robustness:

| Dataset type | Example use | Model support |
|---|---|---|
| Sales / retail | product trend, stock suggestion, sales EDA | only if a binary target is available or created |
| Bank marketing | campaign response classification | yes, if target is `y` yes/no |
| Employee attrition | HR support and retention review | yes, if target is `Attrition` yes/no |
| Customer feedback | feedback themes and improvement actions | text analysis and optional churn model |
| Operations/service | complaint/service hotspot analysis | depends on target availability |

Do not upload large raw datasets to GitHub. Keep them locally in `data_raw/` and commit only small samples to `data_samples/`.

---

## Example Questions for the Copilot

### General data questions

```text
any missing values?
show dataset summary
which columns are numeric?
show duplicate rows
```

### Bank marketing examples

```text
show y distribution
show y by job
unemployed marital status
which job group has highest yes rate?
model performance
```

### Customer churn examples

```text
what are the top churn drivers?
which customers are high risk?
why is this customer high risk?
what action should be considered?
```

### Customer feedback examples

```text
which customerfeedback is worst?
who gave worst customerfeedback?
show best customerfeedback
show customerfeedback themes
how to improve customerfeedback?
```

### Sales and retail examples

```text
which product sold the most?
which product is declining?
what should we purchase more next month?
which product performed well last year but weak now?
```

### HR examples

```text
which department has high attrition?
who may need support review?
which group has high overtime?
what retention action should be considered?
```

---

## Evaluation Plan

The user evaluation should compare two conditions:

### Condition A: prediction-only output

Participants see only a risk score or category.

### Condition B: explanation-supported output

Participants see:

- prediction score;
- SHAP or feature drivers;
- plain-English explanation;
- recommendation;
- safety warning.

Suggested measures:

- understanding;
- trust;
- usefulness;
- usability;
- decision confidence;
- safety awareness.

Analysis can include descriptive statistics, comparison charts and thematic review of short comments.

---

## GitHub Evidence Checklist

For dissertation evidence, update the repository regularly with:

- meaningful commits;
- screenshots in `docs/screenshots/`;
- architecture diagrams in `docs/architecture/`;
- testing results;
- evaluation materials;
- implementation notes;
- README updates;
- issue/task tracking if possible.

Suggested screenshot names:

```text
01_dataset_hub.png
02_data_cleaning_report.png
03_eda_dashboard.png
04_model_leaderboard.png
05_shap_explanation.png
06_copilot_chat.png
07_business_insight_engine.png
08_report_export.png
```

---

## GitHub Upload Notes

Before pushing to GitHub, remove Python cache and test-cache folders:

Windows PowerShell:

```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue
```

macOS/Linux:

```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf .pytest_cache
```

Then commit:

```bash
git add .
git commit -m "Add explainable AI Copilot business insight prototype"
git push
```

---

## Limitations

This is a proof-of-concept prototype, not a production enterprise system.

Current limitations:

- predictive modelling focuses on binary classification;
- sales forecasting is rule/trend based unless a suitable target is created;
- market-wide recommendations require external market data, which is not included unless uploaded by the user;
- HR outputs are review signals only and must not be used for automatic employment decisions;
- LLM-style open-ended generation is intentionally avoided to keep the prototype controlled and reproducible;
- very large datasets may require sampling for modelling to keep Streamlit responsive.

---

## Large-Scale Architecture Extension

A future enterprise version could include:

- FastAPI backend;
- database storage such as PostgreSQL;
- role-based access control;
- audit logs;
- model monitoring;
- dataset versioning;
- explainability trace storage;
- secure authentication;
- controlled LLM integration with retrieval and governance;
- deployment on cloud infrastructure.

These are discussed as future architecture extensions and are not required for the proof-of-concept implementation.

---

## Academic Positioning

This project connects to the following areas:

- explainable artificial intelligence;
- decision-support systems;
- human-AI interaction;
- conversational analytics;
- business intelligence;
- responsible AI governance;
- human oversight in AI-supported decisions.

The prototype is designed to demonstrate how predictive analytics and explainability can be translated into understandable, data-grounded recommendations for business users.

---

## Version Notes

### v9.1

- Added Business Insight Engine.
- Added domain-aware recommendations for churn, feedback, sales, bank marketing, HR and operations datasets.
- Improved controlled Copilot intent handling.
- Fixed Python f-string syntax issue in `src/copilot.py`.
- Current expected test status: `22 passed`.

---

## Disclaimer

This project is for academic research and prototype demonstration only. Outputs are not professional, financial, legal, HR or operational advice. Human review is required before any real-world action.
