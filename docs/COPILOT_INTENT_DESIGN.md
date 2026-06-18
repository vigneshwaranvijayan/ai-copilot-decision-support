# Controlled Copilot Intent Design

## Purpose

The Copilot is not an unrestricted chatbot. It is a controlled, data-grounded layer for a proof-of-concept dissertation prototype.

## Why this is used

This approach is safer for an MSc dissertation because it avoids hallucinated answers and keeps outputs reproducible.

## New improvements

The improved version includes:

- common typo correction
- fuzzy word matching using Python `difflib`
- column-name matching
- question interpretation message
- suggested fallback questions
- quick question buttons in the Streamlit UI
- clearer grammar in Copilot answers

## Example typo handling

| User typed | System interpretation |
|---|---|
| shw chrun by contrct | show churn by contract |
| what colums in file | what columns are in this file |
| check missng values | check missing values |
| what should improv | what should improve |
| why high risk | why is this customer high risk |

## Supported intents

- data summary
- missing values
- duplicate rows
- columns in the file
- churn rate
- churn rate by group
- highest/lowest numeric value
- average numeric value
- SHAP drivers after prediction
- recommendation generation
- safety/governance warning
