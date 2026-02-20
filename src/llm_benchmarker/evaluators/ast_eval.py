"""AST-based function call comparison for BFCL benchmark."""

from __future__ import annotations

import ast
import json
from typing import Any


def normalize_value(value: Any) -> Any:
    """Normalize a value for comparison (strings, numbers, lists, dicts)."""
    if isinstance(value, str):
        # Try to parse as JSON
        try:
            return normalize_value(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            return value.strip()
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in sorted(value.items())}
    return value


def parse_function_call(text: str) -> list[dict[str, Any]]:
    """Parse function call(s) from model output.

    Supports:
    - JSON array of {"name": ..., "arguments": {...}}
    - Single JSON object {"name": ..., "arguments": {...}}
    - Python-style function calls: func_name(arg1=val1, arg2=val2)
    """
    text = text.strip()

    # Try JSON parsing first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting JSON from code blocks
    import re
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1).strip())
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except (json.JSONDecodeError, TypeError):
            pass

    # Try Python AST parsing for function calls
    calls = []
    try:
        tree = ast.parse(text, mode="eval")
        if isinstance(tree.body, ast.Call):
            calls.append(_ast_call_to_dict(tree.body))
    except SyntaxError:
        # Try wrapping in a list
        try:
            tree = ast.parse(f"[{text}]", mode="eval")
            if isinstance(tree.body, ast.List):
                for elt in tree.body.elts:
                    if isinstance(elt, ast.Call):
                        calls.append(_ast_call_to_dict(elt))
        except SyntaxError:
            pass

    return calls


def _ast_call_to_dict(node: ast.Call) -> dict[str, Any]:
    """Convert an AST Call node to {"name": ..., "arguments": {...}}."""
    # Get function name
    if isinstance(node.func, ast.Name):
        name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        name = node.func.attr
    else:
        name = ""

    arguments = {}
    for kw in node.keywords:
        if kw.arg:
            arguments[kw.arg] = ast.literal_eval(kw.value)

    return {"name": name, "arguments": arguments}


def compare_function_calls(
    predicted: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Compare predicted function calls against expected ones.

    Returns (match, details) where match is True if all expected calls
    are satisfied by predicted calls (order-independent).
    """
    if len(predicted) != len(expected):
        return False, {
            "reason": f"count mismatch: predicted {len(predicted)} vs expected {len(expected)}"
        }

    matched = [False] * len(expected)
    details: dict[str, Any] = {"per_call": []}

    for pred in predicted:
        pred_name = pred.get("name", "")
        pred_args = normalize_value(pred.get("arguments", {}))

        for i, exp in enumerate(expected):
            if matched[i]:
                continue
            exp_name = exp.get("name", "")
            exp_args = normalize_value(exp.get("arguments", {}))

            if pred_name == exp_name and pred_args == exp_args:
                matched[i] = True
                details["per_call"].append({"expected": exp_name, "match": True})
                break
        else:
            details["per_call"].append({
                "predicted_name": pred_name,
                "match": False,
                "reason": "no matching expected call",
            })

    all_match = all(matched)
    return all_match, details
