# Operational Collaboration Case Study

## Purpose

This case study supports the dissertation architecture and discussion chapter. It should not expand the implementation scope.

## Scenario

A customer success manager, business analyst, operations manager and governance/audit user work in a shared operational environment. They need to understand customer churn risk, identify weak customer segments, interpret predictive analytics, and decide which actions should be reviewed by humans.

## Stakeholders

| Stakeholder | Information need | Copilot support |
|---|---|---|
| Business analyst | Data quality, model performance, feature drivers | Dashboard, model metrics, SHAP evidence |
| Customer success manager | Customer risk reasons and possible retention actions | Plain-English explanation and recommendation |
| Operations manager | Segment-level problems and improvement priorities | Aggregated churn insights and charts |
| Governance/audit user | Logs, model versions, explanation trace and safety controls | Audit trail and governance layer in large-scale design |

## Large-scale architecture relevance

The large-scale architecture should discuss:

- role-based access control
- audit logs
- model versioning
- explanation traceability
- human review requirements
- monitoring and governance
- responsible AI safeguards

## Scope note

The implemented proof-of-concept remains focused on customer churn. This collaboration scenario is used for architecture and discussion only.
