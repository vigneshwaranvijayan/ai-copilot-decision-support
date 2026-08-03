# v15.2 Final UI and Model-Grounded Copilot Change Log

## Main reason for this version
The previous version worked technically, but the user interface was still crowded and the role of modelling was not visible enough in Copilot answers. This version improves both.

## Previous state
- Dashboard, Cleaning/EDA, Model, Business, Chat, Evaluation, Export, Visualisation and Research/Memory pages existed.
- Dataset upload worked through the sidebar.
- Auto-modelling ran once after upload when a binary target was detected.
- Copilot answers were dataset-grounded and included readiness, evidence, limitations and warnings.
- Some question-routing fixes were already added in v15.1.

## What changed in v15.2
1. Added a modern top navigation bar with:
   - left brand name: AI Solution
   - dummy account indicator: Research User
   - dark mode toggle
   - quick export button

2. Added modern sidebar navigation buttons:
   - Upload
   - Dashboard
   - Cleaning & EDA
   - Model
   - Graphs
   - Business
   - Chat
   - Evaluation
   - Export
   - Research

3. Added a central Upload Data page:
   - Local files
   - URL/API
   - Google Drive
   - PostgreSQL

4. Improved dashboard explanation:
   - shows how modelling is used
   - explains that the model is not only displayed as a metric
   - separates EDA answers from prediction/explanation answers

5. Improved Copilot answer frame:
   - added Answer grounding PASS/WARNING/FAIL
   - added Model / prediction evidence section
   - model name, target, F1, ROC-AUC, recall and precision are now shown when available
   - explanation availability is shown when SHAP/fallback has been generated

6. Added tests for:
   - model evidence appearing in Copilot answers
   - technical/customer issue question not using customer ID incorrectly

## Why modelling is used
Modelling is used when the user asks prediction/explanation questions, for example:
- What model was trained?
- Is this model good enough for decision support?
- What factors increased churn risk?
- Why is this customer high risk?
- What features are most important?
- What prediction evidence supports this recommendation?

Dataset summary questions still use EDA/statistics. Business suggestions use data evidence and, when available, model/explanation evidence.

## Test results
- py_compile app.py and src/*.py: passed
- pytest -q: 34 passed
