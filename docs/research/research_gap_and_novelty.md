# Research Gap, Novelty and Contribution

## Why the project still matters when BI Copilot products already exist

Commercial platforms such as Power BI Copilot, Tableau Pulse and ThoughtSpot already provide AI-assisted analytics, natural-language interaction and business insight support. The dissertation should therefore **not** claim that an AI data Copilot is completely new.

The research contribution is different: this project builds and evaluates a **controlled, explainable and dataset-grounded academic prototype** that makes each stage visible and reproducible:

```text
uploaded data → cleaning → EDA → model comparison → SHAP/fallback explanation → Copilot translation → business recommendation → safety warning → user evaluation
```

Commercial tools are powerful, but they are normally enterprise products that depend on prepared semantic models, cloud services, licences, role configuration and organisational data governance. This project focuses on a transparent proof-of-concept that can be inspected, tested and evaluated in an MSc dissertation.

## Main research gap

Existing research and systems often focus on one of the following separately:

1. **BI Copilots and conversational analytics**: support natural-language data exploration and dashboard/metric explanation.
2. **XAI tools**: explain model outputs using methods such as SHAP or LIME.
3. **AutoML tools**: automate model selection and performance comparison.
4. **Decision-support systems**: help users make decisions, but may not integrate modern XAI and Copilot interaction.
5. **Responsible AI frameworks**: define governance and risk controls, but do not implement an end-user analytics workflow.

The gap is the **controlled integration** of these components into one data-grounded workflow for non-technical business users.

## Gap statements

### Gap 1: Explanation-to-action gap

SHAP/LIME explain why a model predicted something, but business users often need to know what action should be considered next. This project maps explanation drivers to business recommendations and warnings.

### Gap 2: Dataset-grounding gap

Open AI chat systems may hallucinate or answer outside the dataset. This project restricts answers to the active uploaded dataset, cleaning outputs, EDA summaries, model metrics, explanation artefacts and transparent recommendation rules.

### Gap 3: Human-evaluation gap

Many technical prototypes report model metrics but do not evaluate whether explanations actually improve understanding, trust, usefulness, decision confidence and safety awareness. This project includes a prediction-only vs explanation-supported evaluation workspace.

### Gap 4: Small-to-large-scale gap

Many student prototypes work only on small sample datasets. This project keeps a focused small-scale churn evaluation but adds a large-dataset robustness and scale-readiness pathway using sampling, aggregation and an enterprise architecture design.

### Gap 5: Safe business recommendation gap

Business recommendation systems can become risky if they appear to make automatic decisions. This project frames outputs as decision support only and includes safety warnings, especially for HR and customer treatment contexts.

## Novelty claim

The novelty is not a new algorithm. The novelty is the **design, integration and evaluation** of a controlled Explainable AI Copilot framework that:

- accepts uploaded structured business data;
- automatically cleans and profiles it;
- supports classification-first modelling and optional regression robustness;
- uses SHAP/fallback explanations;
- translates technical outputs into plain-English evidence;
- links evidence to business improvement suggestions;
- includes human-review safety warnings;
- evaluates user understanding and trust;
- demonstrates small-scale and large-scale readiness.

## Suggested dissertation contribution wording

> This dissertation contributes a controlled, dataset-grounded Explainable AI Copilot framework for operational business decision support. The contribution is not a new ML algorithm or a direct commercial alternative to existing BI Copilot products. Instead, it is the integration and evaluation of uploaded-data analysis, predictive modelling, explainable AI, plain-English Copilot translation, business-action recommendation and safety warnings within a reproducible proof-of-concept that can be evaluated with users and extended conceptually to enterprise scale.
