# Architecture Overview

```text
Data sources
  - CSV / TSV / Excel / JSON upload
  - ZIP containing supported files
  - Public CSV / JSON / Excel URL or API endpoint
        ↓
Dataset Hub
  - Stores multiple datasets in session
  - Cleans each dataset
  - Allows active dataset selection
  - Optional merge by common key
        ↓
Cleaning + validation
  - Column normalisation
  - Empty row/column removal
  - Duplicate removal
  - Numeric/date conversion
  - Missing-value and schema profile
        ↓
EDA Layer
  - Dataset overview
  - Missing values
  - Numeric/categorical/date/text summaries
  - Target distribution and group comparisons
        ↓
Model Layer, when binary target exists
  - Logistic Regression
  - Random Forest
  - Extra Trees
  - Gradient Boosting
  - Assessed four-model classification comparison: Logistic Regression, Random Forest, Gradient Boosting and MLP baseline
  - Best model selected by F1, Recall and ROC-AUC
        ↓
Explainability Layer
  - SHAP where available
  - Fallback model importance if SHAP fails
        ↓
Controlled Copilot Layer
  - Spelling/grammar tolerance
  - Maps question to controlled operations
  - Answers only from active uploaded dataset and model artefacts
  - Shows tables/charts and safety warnings
```

## Scope control

The main evaluated case study remains customer churn. Multi-dataset and large-dataset support is used as robustness evidence and does not change the core dissertation scope.
