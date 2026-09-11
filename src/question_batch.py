"""Utilities for batch reviewer-question input in the Copilot UI."""
from __future__ import annotations

import re
from typing import List


def split_reviewer_questions(raw: str) -> List[str]:
    """Split numbered or line-separated reviewer questions into separate turns.

    Supports 1..999 numbering and also plain copied text where each question is
    on its own line. A single ordinary question is returned unchanged.
    """
    text = (raw or "").strip()
    if not text:
        return []
    parts = re.split(r"(?=\b\d{1,3}\.\s+)", text)
    questions: List[str] = []
    for part in parts:
        cleaned = re.sub(r"^\s*\d{1,3}\.\s*", "", part).strip(" ;\n\t")
        if cleaned:
            questions.append(cleaned)
    if len(questions) > 1:
        return questions

    line_questions: List[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*[-*•]\s*", "", line).strip()
        if cleaned:
            line_questions.append(cleaned)
    likely_questions = [
        x for x in line_questions
        if x.endswith("?") or re.match(r"^(what|which|how|why|is|are|do|does|can|should|explain|show)\b", x, re.I)
    ]
    if len(line_questions) >= 2 and len(likely_questions) >= max(2, int(len(line_questions) * 0.8)):
        return line_questions
    return [text]
