"""Normalized string matching evaluator."""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace, remove punctuation."""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(predicted: str, expected: str) -> bool:
    """Return True if normalized predicted matches normalized expected."""
    return normalize(predicted) == normalize(expected)


def contains_match(predicted: str, expected: str) -> bool:
    """Return True if normalized expected is found within normalized predicted."""
    return normalize(expected) in normalize(predicted)
