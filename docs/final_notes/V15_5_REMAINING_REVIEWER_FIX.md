# v15.5 Remaining Reviewer Fix

This version fixes issues found in the latest exported Telco Copilot chat evidence.

## Problems seen in the export
- Research-gap question was not answered and fell back to "could not safely map".
- ChromaDB question was not answered and fell back to "could not safely map".
- Export chat/evidence question was not answered and fell back to "could not safely map".
- Competitor-pricing fail-safe question was too generic.
- Tenure/monthly-charges relationship answers had real evidence but were marked Answer grounding FAIL because the safety checker treated "automatic" wording too strictly.

## Fixes
1. Added direct reviewer/research answer route for:
   - research gap
   - contribution
   - dashboard comparison
2. Added direct ChromaDB route.
3. Added direct export/evidence route.
4. Added competitor-pricing safe refusal route.
5. Improved grounding safety check so correlation answers with "not automatic decision rule" are not falsely marked FAIL.
6. Added tests for these reviewer cases.

## Test result
pytest -q: 46 passed
py_compile app.py and src/*.py: passed
