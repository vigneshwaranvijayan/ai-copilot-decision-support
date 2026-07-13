# Visual Analytics Dashboard Strategy

## Purpose

The Visual Analytics dashboard strengthens the prototype because commercial BI tools such as Power BI and Tableau are already strong at visualisation. The project does not claim to outperform those tools. Instead, the dashboard demonstrates how visual evidence can be integrated with explainable AI, controlled Copilot responses, business recommendations and human-review safety warnings.

## Design principles

1. **Dataset-grounded visuals**: every chart is generated from the active uploaded dataset only.
2. **Domain-aware selection**: chart recommendations depend on the detected dataset type, such as churn, feedback, sales, bank marketing, HR or operations.
3. **Scale-aware rendering**: large datasets use aggregation, top-N summaries or sampling rather than plotting every row.
4. **Decision-support framing**: charts are connected to evidence and possible business actions rather than shown as decorative graphs.
5. **Human-review safety**: charts and recommendations support decisions but do not automate final actions.

## Dashboard outputs

- KPI cards: rows, columns, missing cells, duplicate rows, data quality score and detected domain.
- Data quality gauge: quick view of dataset readiness.
- Missing-value chart: identifies fields needing cleaning or caution.
- Target distribution chart: shows class balance for classification tasks.
- Domain charts: churn by contract/payment/service, sales trends/top products, campaign response by job/education, HR attrition by department/overtime, operations volume by status/category.
- Evidence tables: allow users and examiners to verify chart values.

## Dissertation use

The dashboard should be used as evidence that the prototype is not only a text chatbot. It provides a visual analytics layer similar in spirit to BI systems, while the novelty remains the combination of visual evidence with explainable modelling, Copilot interpretation, recommendation and safety warnings.
