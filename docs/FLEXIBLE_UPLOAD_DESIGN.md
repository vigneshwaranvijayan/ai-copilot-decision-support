# Flexible Upload Design

## Why this version was added

The dissertation proof-of-concept remains focused on customer churn decision support. However, the app now accepts any CSV file for general data exploration so the prototype does not fail when a user uploads a different dataset.

## What works for any CSV

- data preview
- missing values
- duplicate rows
- data types
- column list
- categorical distributions
- numeric histograms
- highest/lowest numeric values
- average numeric values
- controlled Copilot questions about the uploaded file

## What requires a binary target

- model training
- prediction
- SHAP explanation
- prediction-only vs explanation-supported evaluation

The selected target must contain exactly two classes, for example:

- Yes / No
- 1 / 0
- True / False

## Dissertation position

Flexible upload is a robustness and usability improvement. The main evaluated case study should still be customer churn to keep scope manageable.


## Important scope control statement

Flexible CSV upload is a robustness and usability feature only. It prevents the prototype from failing when a user uploads a non-churn CSV. The main dissertation evaluation remains focused on the customer churn case study.

This should be explained clearly in the dissertation and README so that flexible upload does not look like new scope creep.
