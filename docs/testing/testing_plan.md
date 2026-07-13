# Testing Plan

## Unit tests

Run:

```bash
pytest -q
```

Expected coverage areas:

- data loading;
- cleaning;
- target detection;
- classification/regression training helpers;
- Copilot intent handling;
- feedback/business insight handling.

## Manual tests

| Test | Dataset | Expected result |
|---|---|---|
| Upload Telco churn | Telco CSV | Cleaning, EDA and classification enabled |
| Ask churn recommendation | Telco CSV | Dataset-grounded recommendation and warning |
| Upload bank marketing | bank-full.csv | Target `y` works without matching random letters in words |
| Upload sales dataset | retail CSV | Business insight and optional regression if numeric target selected |
| Upload large retail data | Online Retail / Online Retail II | Scale Readiness tab records row/memory evidence |
| Use evaluation workspace | Churn model trained | Condition A/B outputs and questionnaire export |
