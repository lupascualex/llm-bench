"""Set match and Jaccard similarity for GraphWalks benchmark."""

from __future__ import annotations

import re


def extract_set(text: str) -> set[str]:
    """Extract a set of items from model output.

    Handles:
    - Comma-separated items
    - Newline-separated items
    - Items in braces {a, b, c}
    - Items in brackets [a, b, c]
    """
    text = text.strip()

    # Remove surrounding braces/brackets
    match = re.search(r'[\[{](.*?)[\]}]', text, re.DOTALL)
    if match:
        text = match.group(1)

    # Split by comma or newline
    items = re.split(r'[,\n]', text)
    return {item.strip().strip("'\"").strip() for item in items if item.strip()}


def set_exact_match(predicted: str, expected: str) -> bool:
    """Return True if the extracted sets are identical."""
    pred_set = extract_set(predicted)
    exp_set = extract_set(expected)
    return pred_set == exp_set


def jaccard_similarity(predicted: str, expected: str) -> float:
    """Compute Jaccard similarity between extracted sets."""
    pred_set = extract_set(predicted)
    exp_set = extract_set(expected)

    if not pred_set and not exp_set:
        return 1.0
    if not pred_set or not exp_set:
        return 0.0

    intersection = pred_set & exp_set
    union = pred_set | exp_set
    return len(intersection) / len(union)
