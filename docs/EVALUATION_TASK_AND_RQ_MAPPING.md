# Evaluation Task Design and RQ2 Mapping

## Purpose

This file strengthens the evaluation design for the dissertation.

## Research Question 2

**RQ2:** To what extent do explanation and recommendation outputs improve users' understanding, trust, usability and decision confidence compared with prediction-only outputs?

## A/B conditions

| Condition | User sees | User task |
|---|---|---|
| A: Prediction only | Customer churn risk score/category only | Choose a possible retention action using only the prediction output. |
| B: Copilot supported | Prediction + SHAP drivers + plain-English explanation + recommendation + safety warning | Choose a possible retention action using the full decision-support output. |

## Counterbalancing

If possible, participants should be split into two groups:

- Group 1 sees Condition A first, then Condition B.
- Group 2 sees Condition B first, then Condition A.

This helps reduce order bias.

## Measurement mapping

| RQ2 measure | Example questionnaire item |
|---|---|
| Understanding | I understood why the prediction was made. |
| Trust | I trusted the output appropriately. |
| Usefulness | The recommendation was useful for decision support. |
| Usability | The system was easy to use. |
| Decision confidence | I felt confident choosing a possible action. |
| Safety awareness | The system made clear that human review is required. |

## Analysis

The dissertation should compare average Likert-scale ratings between Condition A and Condition B. Short comments can be analysed thematically to identify what users found useful or confusing.
