# v15 Research Contribution and Change Log

## Previous work retained

- Multi-source data loading: upload, URL/API, Google Drive and PostgreSQL-ready connector.
- Automatic cleaning, EDA, visual analytics and manual visual builder.
- Automatic model training once after data upload, plus manual training.
- Classification-first model comparison and optional regression.
- SHAP/fallback explainability.
- Dataset-grounded Copilot answer format.
- Business improvement suggestions, safety warnings, audit log and exports.
- Customer churn as the primary proof-of-concept case study.

## Final v15 upgrades

1. **Research contribution made explicit** through five contribution areas: readiness, grounding, explainability-to-action, memory and evaluation.
2. **PASS/WARNING/FAIL thresholds defined** for missing data, duplicates, row count, target readiness, class balance, privacy and question evidence.
3. **MongoDB removed from the core architecture** because the supervisor advised that ChromaDB is more appropriate for retrieval-based AI memory.
4. **ChromaDB added** as long-term semantic memory for dataset summaries, explanations, reports, Copilot answers and generated insights.
5. **Short-term memory clarified** as session context for current dataset, selected target, last question, last answer and follow-up context.
6. **Evaluation framework strengthened** with model performance, explanation quality, grounding accuracy, trustworthiness, usability and decision-support metrics.
7. **Project report rebuilt** as a full aligned PDF with separate architecture images.
8. **Code checked** with py_compile and pytest.

## Final research statement

This project proposes and implements a dataset-grounded Explainable AI Copilot framework for operational business decision support. The system validates data readiness, retrieves evidence, explains model outputs, translates insights into business recommendations, records memory/audit evidence and evaluates whether explanation-supported outputs improve user trust, understanding and decision confidence compared with prediction-only outputs.
