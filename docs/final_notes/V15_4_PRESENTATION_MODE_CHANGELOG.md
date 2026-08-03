# v15.4 Presentation Mode Final Change Log

## Purpose
This version improves the final reviewer/demo experience so the user does not need to manually correct questions repeatedly. It adds a controlled Reviewer Mode, auto explanation after modelling, confidence status, and screenshot-ready answer display.

## Main improvements
1. Reviewer Mode buttons in the Chat page for high-standard supervisor/reviewer questions.
2. Screenshot mode toggle in Chat page to show shorter presentation-ready answers while keeping full export evidence.
3. Auto explanation generation after auto-modelling so top-driver questions work without a manual extra step.
4. Confidence line added to every Copilot answer:
   - HIGH: supported by dataset/model/explanation evidence
   - MEDIUM: partial evidence; human review required
   - SAFE REFUSAL: unsupported/regulated request refused without guessing
5. Copilot framing now shows readiness, answer grounding, confidence, model/prediction evidence, integrated-data evidence, decision-support use, limitations and evidence format.
6. Added tests for confidence and legal-refusal presentation.

## Recommended final demo flow
1. Open app.
2. Use Upload Data page to load the Telco churn dataset.
3. Wait for auto-modelling and auto-explanation.
4. Open Chat page.
5. Use Reviewer Mode buttons for screenshots.
6. Export chat answers and audit evidence.

## Testing
- `python -m py_compile app.py src/*.py`: passed
- `pytest -q`: 42 passed
