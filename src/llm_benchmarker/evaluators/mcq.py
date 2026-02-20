"""MCQ answer extraction for 4-option and 10-option multiple choice."""

from __future__ import annotations

import re

# Letters for 4-option and 10-option MCQs
LETTERS_4 = "ABCD"
LETTERS_10 = "ABCDEFGHIJ"


def extract_mcq_answer(response: str, num_options: int = 4) -> str | None:
    """Extract a single letter answer from an MCQ response.

    Tries multiple heuristics:
    1. Looks for explicit patterns like "The answer is (B)" or "Answer: B"
    2. Looks for a standalone letter at the start/end
    3. Falls back to the first valid letter found
    """
    letters = LETTERS_10[:num_options]
    text = response.strip()

    # Pattern 1: "the answer is X", "answer: X", "(X)"
    patterns = [
        rf"(?:the\s+)?answer\s*(?:is|:)\s*\(?([{letters}])\)?",
        rf"\b([{letters}])\s*[\.\)]\s*$",
        rf"^\s*\(?([{letters}])\)?\s*$",
        rf"\(([{letters}])\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    # Pattern 2: last standalone letter
    matches = re.findall(rf"\b([{letters}])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    return None


def evaluate_mcq(response: str, expected: str, num_options: int = 4) -> bool:
    """Return True if the extracted MCQ answer matches the expected letter."""
    predicted = extract_mcq_answer(response, num_options)
    if predicted is None:
        return False
    return predicted.upper() == expected.upper()
