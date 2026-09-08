# Explainable AI Copilot for Business Decision Support

Advanced supervisor-aligned working prototype for the CS5500 MSc dissertation.

## Main project focus

The implemented proof-of-concept focuses on customer churn decision support and wider structured business dataset analysis. The system supports:

- structured dataset upload and preview
- data validation and PASS/WARNING/FAIL readiness checking
- visual analytics and exploratory charts
- four-model classification comparison
- SHAP / feature-importance explanation
- Copilot-style question answering using available evidence
- typo-tolerant question interpretation for common spelling/grammar mistakes
- recommendation, limitation and safety-warning outputs
- short-term session memory
- exportable evidence for dissertation reporting
- optional document-memory upload with overlapping chunking
- advanced retrieval design: metadata filtering, hybrid TF-IDF/BM25 search, reranking and retrieval metrics
- architecture notes for future PostgreSQL + ChromaDB enterprise extension

## Assessed four-model workflow

The classification workflow uses exactly these four models by default:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. MLP Neural Network Baseline

XGBoost is not part of the final assessed workflow.

## Advanced Copilot/memory additions

This version adds a local retrieval pipeline for the document knowledge and long-term memory design:

1. **Better chunking with overlap**: uploaded text/PDF/DOCX/Markdown documents can be split into chunks while preserving surrounding context.
2. **Metadata filtering**: chunks can be filtered by company, department, topic and document type before search.
3. **Hybrid search**: TF-IDF semantic-style search and BM25 keyword search run in parallel and combine scores.
4. **Reranking**: candidate chunks are reranked using a transparent relevance score before being passed to the Copilot evidence stage.
5. **Retrieval evaluation**: precision@k, recall@k, MRR and nDCG@k are available for document-retrieval testing.

These features are useful for Chapter 3 memory architecture and Chapter 6 future enterprise extension. The main evaluated case study remains customer churn.

## Supervisor feedback coverage

| Supervisor point | Included in project |
|---|---|
| Data integration | Upload, URL/API, Google Drive shared link and PostgreSQL connector modules |
| Readiness validation | PASS/WARNING/FAIL gates with measurable criteria |
| Analytics | EDA and visual analytics pages |
| Explainability | SHAP and fallback feature-importance support |
| Copilot answer generation | Evidence-grounded Copilot module |
| Evaluation | Model metrics, Copilot answer review, user questionnaire template, software tests and retrieval metrics |
| Research contribution | Dedicated research contribution page and notes |
| Measurable readiness criteria | Readiness criteria table and validation module |
| Model performance metrics | Accuracy, precision, recall, F1-score, ROC-AUC and confusion matrix |
| Explanation quality | SHAP driver outputs and explanation summaries |
| Trust/usability | Evaluation questionnaire and metrics table |
| Grounding accuracy | Evidence-package and answer-grounding checks |
| Short-term memory | Session memory stores active dataset, target, model, question and answer context |
| Long-term memory architecture | ChromaDB/vector-memory design and PostgreSQL structured-storage design |
| Advanced retrieval | Chunking, metadata filtering, hybrid search, reranking and retrieval metrics |
| MongoDB reconsidered | MongoDB removed from core design; PostgreSQL + ChromaDB used in architecture |

## Run locally

```powershell
cd C:\Users\Vignesh\Downloads\ai_copilot_decision_support_advanced_v1

py -3.13 -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python -m streamlit run app.py
```

Use `python -m streamlit run app.py` rather than only `streamlit run app.py` on Windows.

## Test

```powershell
python -m pytest -q
```

The package was checked after the advanced update: **50 tests passed**.

## Dissertation wording

In the abstract, keep the description simple: "data validation and readiness checks". In methodology and implementation, explain the measurable PASS/WARNING/FAIL readiness framework because this was specifically requested in supervisor feedback.

Do not claim the advanced document retrieval pipeline is the main evaluated contribution unless you fully evaluate it. Position it as an implemented supporting feature and future enterprise memory direction.
