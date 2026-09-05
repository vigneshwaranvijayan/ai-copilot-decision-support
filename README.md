# Final Working MSc Package - Explainable AI Copilot for Business Decision Support

This package is based on the last working v15.5 repository and keeps the complete UI, architecture diagrams, research notes, testing evidence, memory design, visual analytics and Copilot grounding files. It has been cleaned for final use by removing the `.git` folder and aligning the assessed classification workflow with the dissertation: Logistic Regression, Random Forest, Gradient Boosting and MLP Neural Network Baseline.

Current local test check: `pytest -q` passes after packaging.

## Final dissertation scope

- Main case study: customer churn decision support.
- Supporting capability: other suitable structured datasets can be tested for data upload, validation, visualisation, modelling and Copilot answers.
- Not included as a main case study: any specific external organisational case or employee-specific decision-making.
- Main contribution: data analysis, visualisation, modelling, SHAP/fallback explanation, Copilot chat, evidence grounding, memory support, recommendation, safety warning and exportable results.

---

# Dataset-Grounded AI Copilot v15.5 Remaining Reviewer Fix

v15.5 fixes remaining reviewer/demo questions found from the latest exported chat file: research-gap, ChromaDB, export/evidence, competitor-pricing safe refusal, and false FAIL grounding for relationship evidence.

Test result: `pytest -q` -> 46 passed.

# Dataset-Grounded AI Copilot v15.4 Presentation Mode Final

This is the final presentation/reviewer-ready version. It keeps the v15.3 stability fixes and adds Reviewer Mode buttons, screenshot-ready chat display, auto explanation after modelling, and confidence labels in Copilot answers.

## v15.4 highlights
- Reviewer Mode buttons for safe high-standard demo questions.
- Screenshot mode toggle for shorter, professional Copilot answers.
- Auto explanation after auto-modelling, so top-driver questions work without manual SHAP/fallback generation.
- Every answer now shows answer grounding and confidence.
- Safe refusal confidence for legal/unsupported questions.
- Model evidence appears in Copilot answers when a trained model exists.

## Correct project claim
This is a controlled dataset-grounded AI Copilot. It is not a perfect general chatbot. It aims to be reliable for supported operational decision-support questions and safe for unsupported questions using readiness checks, evidence grounding, model/explanation artefacts, memory, confidence labels, refusal logic and audit logs.



# Dataset-Grounded AI Copilot v15.3 Reviewer-Stability Edition

This version patches the reviewer challenge issues found in the uploaded chat export.

## v15.3 fixes
- Target relationship questions now compare requested columns against churn instead of returning only the overall churn rate.
- Example fixed: `How does tenure relate to churn?`
- Example fixed: `Which payment method is linked with higher churn?`
- Driver questions are prioritised before generic model-performance answers.
- Retention priority/action questions now use churn segment, feedback and model evidence where available.
- System/research questions now have controlled answers:
  - short-term memory
  - MongoDB removal / ChromaDB design
  - audit trail and answer logging
  - graph grounding
  - dataset RAM/memory use
- Legal-advice questions are explicitly refused with a safety warning.
- Prediction-evidence questions now show target, model metrics and explanation status.
- Tests increased to 40 passed.



# Dataset-Grounded AI Copilot v15.2 Final UI and Model-Grounded Edition

This version includes the final UI/navigation improvements and makes the role of modelling visible in Copilot answers.

## v15.2 highlights
- Modern top bar: AI Solution brand, dummy account, dark mode toggle and export button.
- Sidebar button navigation for Upload, Dashboard, Model, Graphs, Business, Chat, Evaluation, Export and Research.
- Central Upload Data page for local files, URL/API, Google Drive and PostgreSQL.
- Copilot answers now include answer-grounding status and model/prediction evidence when available.
- Model results are used for model-performance questions, top-driver questions, prediction/explanation questions and decision-support suitability.
- Dataset-only questions still use EDA/statistical evidence.
- ChromaDB remains the semantic memory layer; PostgreSQL remains the structured data layer.



# Dataset-Grounded AI Copilot for Operational Decision Support - v15.1 Reviewer-Ready Edition

This repository contains the final research-ready proof-of-concept prototype for an Explainable AI Copilot that supports operational business decision-making from integrated datasets.

## Main research contribution

The contribution is not just a dashboard. The final system demonstrates a dataset-grounded Explainable AI Copilot framework with five linked contributions:

1. **Measurable PASS/WARNING/FAIL readiness framework** before analysis, modelling or answer generation.
2. **Evidence-grounding mechanism** that packages source, columns, statistics, model metrics, explanations and limitations before answering.
3. **Explainability-to-action pipeline** that translates model drivers into business recommendations and safety warnings.
4. **Memory-supported Copilot architecture** with short-term session memory and ChromaDB long-term semantic memory.
5. **Evaluation framework** covering model performance, explanation quality, grounding accuracy, trust, usability and decision support.

## What changed from the previous implementation

| Area | Previous work | v15 final change |
|---|---|---|
| Data ingestion | Local upload, URL/API, Google Drive and database-ready ideas | Kept and documented as an integrated data-source layer |
| MongoDB | Considered for unstructured/chat storage | Removed from the core architecture |
| ChromaDB | Not central | Added as the semantic long-term memory layer for retrieved evidence |
| Readiness | General data quality score | Added measurable PASS/WARNING/FAIL thresholds |
| Copilot | Dataset-grounded response format | Added evidence package, answer readiness and memory retrieval |
| Evaluation | Prediction-only vs explanation-supported evaluation | Extended with model, explanation, grounding, trust, usability and performance metrics |
| UI | Modern dashboard and button navigation | Kept and aligned to final research flow |
| Copilot routing | Some ambiguous questions could map to the wrong column | Added safer intent priority, ID-column blocking, technical-issue routing and dataset-use-case answers |
| Case study | Customer churn primary case study | Kept as the main proof-of-concept; other datasets are robustness tests |

## Data sources supported

- Local upload: CSV, TSV, Excel, JSON and ZIP.
- Public URL/API: CSV, JSON and Excel links.
- Google Drive / Google Sheets shared links.
- PostgreSQL structured data connector.
- ChromaDB semantic memory for summaries, explanations, Copilot answers and generated insights.

## Primary case study

The main evaluated proof-of-concept is **customer churn decision support**. The workflow is:

`data integration -> readiness gates -> EDA -> classification model -> SHAP/fallback explanation -> Copilot answer -> business recommendation -> safety warning -> evaluation/export evidence`

Additional datasets such as bank marketing, online retail/sales and customer feedback are used only for robustness and integration evidence.

## Models used

Classification models used in the assessed workflow:

- Logistic Regression
- Random Forest
- Gradient Boosting
- MLP Neural Network Baseline

Regression is optional for numeric business targets:

- Linear Regression
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- Optional XGBoost Regressor

Model evidence includes accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix and training-row count.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Or on Windows, double-click:

```text
run_app.bat
```

## Final project report and diagrams

- `docs/proposal/Dataset_Grounded_AI_Copilot_Final_Project_Report_v15.pdf`
- `docs/architecture/v15_figures/`
- `docs/research/v15_research_contribution_and_changes.md`
- `docs/testing/final_test_results_v15.txt`
- `docs/testing/final_test_results_v15_1.txt`

## Safety boundary

The app is decision support only. It must not make automatic business, HR, legal or customer decisions. If evidence is missing, the Copilot should refuse or warn rather than inventing an unsupported answer.


## v15.1 reviewer-ready reliability fixes

Manual testing showed that some unclear questions could be routed incorrectly. For example, technical-issue questions could accidentally produce customer ID frequency tables, and dataset-use questions could be treated as feedback value counts. v15.1 fixes these problems using safer intent priority, required-evidence checks, ID-column blocking and feedback/theme routing.

The project should be presented as a proof-of-concept that aims for reliable answers inside supported scope, not as an always-correct general chatbot. Unsupported or low-confidence questions should return clarification, limitation or refusal instead of unsupported claims.
