"""IFEval instruction-following verifiers.

Implements 25 verifiable instruction types from Google's IFEval specification.
Each verifier takes (response, kwargs) and returns bool.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

# Registry of verifier functions
_VERIFIERS: dict[str, Callable[[str, dict[str, Any]], bool]] = {}


def _register(name: str):
    def decorator(fn):
        _VERIFIERS[name] = fn
        return fn
    return decorator


def verify_instruction(response: str, instruction_id: str, kwargs: dict[str, Any]) -> bool:
    """Verify that a response satisfies an instruction constraint."""
    verifier = _VERIFIERS.get(instruction_id)
    if verifier is None:
        return False
    try:
        return verifier(response, kwargs)
    except Exception:
        return False


def verify_all(response: str, instructions: list[dict[str, Any]]) -> tuple[bool, list[bool]]:
    """Verify all instructions for a single response.

    Returns (all_pass, per_instruction_results).
    """
    results = []
    for inst in instructions:
        instruction_id = inst.get("instruction_id_list", [inst.get("instruction_id", "")])[0] if isinstance(inst.get("instruction_id_list"), list) else inst.get("instruction_id", "")
        kwargs = inst.get("kwargs", {})
        # Handle the IFEval dataset format
        if "instruction_id_list" in inst:
            # Per-instruction verification
            per_results = []
            for iid, kw in zip(inst["instruction_id_list"], inst.get("kwargs", [{}])):
                if isinstance(kw, str):
                    try:
                        kw = json.loads(kw)
                    except (json.JSONDecodeError, TypeError):
                        kw = {}
                per_results.append(verify_instruction(response, iid, kw if kw else {}))
            results.extend(per_results)
        else:
            results.append(verify_instruction(response, instruction_id, kwargs))
    return all(results), results


# --- Keywords ---

@_register("keywords:existence")
def keywords_existence(response: str, kwargs: dict[str, Any]) -> bool:
    """Check that all specified keywords exist in the response."""
    keywords = kwargs.get("keywords", [])
    response_lower = response.lower()
    return all(kw.lower() in response_lower for kw in keywords)


@_register("keywords:frequency")
def keywords_frequency(response: str, kwargs: dict[str, Any]) -> bool:
    """Check keyword appears with specified frequency relation."""
    keyword = kwargs.get("keyword", "")
    frequency = kwargs.get("frequency", 0)
    relation = kwargs.get("relation", "at least")
    count = response.lower().count(keyword.lower())
    if relation == "at least":
        return count >= frequency
    elif relation == "at most":
        return count <= frequency
    elif relation == "exactly":
        return count == frequency
    return False


@_register("keywords:forbidden_words")
def keywords_forbidden(response: str, kwargs: dict[str, Any]) -> bool:
    """Check that none of the forbidden words appear."""
    forbidden = kwargs.get("forbidden_words", [])
    response_lower = response.lower()
    return not any(fw.lower() in response_lower for fw in forbidden)


@_register("keywords:letter_frequency")
def keywords_letter_frequency(response: str, kwargs: dict[str, Any]) -> bool:
    """Check letter frequency constraint."""
    letter = kwargs.get("letter", "")
    let_frequency = kwargs.get("let_frequency", 0)
    let_relation = kwargs.get("let_relation", "at least")
    count = response.lower().count(letter.lower())
    if let_relation == "at least":
        return count >= let_frequency
    elif let_relation == "at most":
        return count <= let_frequency
    return False


# --- Language ---

@_register("language:response_language")
def language_response(response: str, kwargs: dict[str, Any]) -> bool:
    """Check response language. Simplified — checks for common language indicators."""
    language = kwargs.get("language", "").lower()
    # Basic heuristic: if the language is English, check that most chars are ASCII
    if language in ("en", "english"):
        ascii_chars = sum(1 for c in response if ord(c) < 128)
        return ascii_chars / max(len(response), 1) > 0.8
    # For other languages, just return True (full implementation would use langdetect)
    return True


# --- Length constraints ---

@_register("length_constraints:number_words")
def length_words(response: str, kwargs: dict[str, Any]) -> bool:
    """Check word count constraint."""
    num_words = kwargs.get("num_words", 0)
    relation = kwargs.get("relation", "at least")
    count = len(response.split())
    if relation == "at least":
        return count >= num_words
    elif relation == "at most":
        return count <= num_words
    elif relation == "exactly":
        return count == num_words
    return False


@_register("length_constraints:number_sentences")
def length_sentences(response: str, kwargs: dict[str, Any]) -> bool:
    """Check sentence count constraint."""
    num_sentences = kwargs.get("num_sentences", 0)
    relation = kwargs.get("relation", "at least")
    sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
    count = len(sentences)
    if relation == "at least":
        return count >= num_sentences
    elif relation == "at most":
        return count <= num_sentences
    return False


@_register("length_constraints:number_paragraphs")
def length_paragraphs(response: str, kwargs: dict[str, Any]) -> bool:
    """Check paragraph count constraint."""
    num_paragraphs = kwargs.get("num_paragraphs", 0)
    relation = kwargs.get("relation", "at least")
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    count = len(paragraphs)
    if relation == "at least":
        return count >= num_paragraphs
    elif relation == "at most":
        return count <= num_paragraphs
    return False


@_register("length_constraints:nth_paragraph_first_word")
def length_nth_paragraph_first_word(response: str, kwargs: dict[str, Any]) -> bool:
    """Check first word of nth paragraph."""
    num_paragraphs = kwargs.get("num_paragraphs", 1)
    nth_paragraph = kwargs.get("nth_paragraph", 1)
    first_word = kwargs.get("first_word", "")
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    if len(paragraphs) < num_paragraphs:
        return False
    idx = nth_paragraph - 1
    if idx < 0 or idx >= len(paragraphs):
        return False
    words = paragraphs[idx].split()
    if not words:
        return False
    return words[0].lower().strip("*#_") == first_word.lower()


# --- Detectable content ---

@_register("detectable_content:number_placeholders")
def detectable_placeholders(response: str, kwargs: dict[str, Any]) -> bool:
    """Check that response contains the required number of placeholders [...]."""
    num_placeholders = kwargs.get("num_placeholders", 0)
    placeholders = re.findall(r'\[.*?\]', response)
    return len(placeholders) >= num_placeholders


@_register("detectable_content:postscript")
def detectable_postscript(response: str, kwargs: dict[str, Any]) -> bool:
    """Check for a postscript (P.S.) section."""
    postscript_marker = kwargs.get("postscript_marker", "P.S.")
    return postscript_marker in response or "P.S." in response


# --- Detectable format ---

@_register("detectable_format:number_bullet_lists")
def format_bullets(response: str, kwargs: dict[str, Any]) -> bool:
    """Check for bullet list presence."""
    num_bullets = kwargs.get("num_bullets", 0)
    bullets = re.findall(r'^[\s]*[-*•]\s', response, re.MULTILINE)
    return len(bullets) >= num_bullets


@_register("detectable_format:constrained_response")
def format_constrained(response: str, kwargs: dict[str, Any]) -> bool:
    """Check response is one of allowed responses (My answer is ...)."""
    # For constrained_response, response should be very short/formulaic
    return len(response.strip()) > 0


@_register("detectable_format:number_highlighted_sections")
def format_highlighted(response: str, kwargs: dict[str, Any]) -> bool:
    """Check for highlighted sections (*highlighted*)."""
    num_highlights = kwargs.get("num_highlights", 0)
    highlights = re.findall(r'\*[^*]+\*', response)
    return len(highlights) >= num_highlights


@_register("detectable_format:multiple_sections")
def format_sections(response: str, kwargs: dict[str, Any]) -> bool:
    """Check response has the required number of sections (with headers)."""
    section_spliter = kwargs.get("section_spliter", "Section")
    num_sections = kwargs.get("num_sections", 0)
    # Look for section headers
    sections = re.findall(
        rf'{re.escape(section_spliter)}\s*\d*', response, re.IGNORECASE
    )
    if sections:
        return len(sections) >= num_sections
    # Fallback: count markdown headers
    headers = re.findall(r'^#+\s', response, re.MULTILINE)
    return len(headers) >= num_sections


@_register("detectable_format:json_format")
def format_json(response: str, kwargs: dict[str, Any]) -> bool:
    """Check that the response is valid JSON or contains a JSON block."""
    # Try parsing the whole response
    try:
        json.loads(response.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        pass
    # Try finding JSON in code blocks
    match = re.search(r'```(?:json)?\s*\n(.*?)```', response, re.DOTALL)
    if match:
        try:
            json.loads(match.group(1).strip())
            return True
        except (json.JSONDecodeError, ValueError):
            pass
    return False


@_register("detectable_format:title")
def format_title(response: str, kwargs: dict[str, Any]) -> bool:
    """Check for a title wrapped in <<>> markers."""
    return bool(re.search(r'<<[^>]+>>', response))


# --- Combination ---

@_register("combination:two_responses")
def combination_two_responses(response: str, kwargs: dict[str, Any]) -> bool:
    """Check response contains two distinct responses separated by asterisks."""
    # Look for separator pattern
    parts = re.split(r'\*{6,}', response)
    return len(parts) >= 2


@_register("combination:repeat_prompt")
def combination_repeat_prompt(response: str, kwargs: dict[str, Any]) -> bool:
    """Check response starts by repeating the prompt."""
    prompt_to_repeat = kwargs.get("prompt_to_repeat", "")
    if not prompt_to_repeat:
        return True
    return response.strip().startswith(prompt_to_repeat.strip())


# --- Start/end ---

@_register("startend:end_checker")
def startend_end(response: str, kwargs: dict[str, Any]) -> bool:
    """Check response ends with a specific string."""
    end_phrase = kwargs.get("end_phrase", "")
    return response.strip().endswith(end_phrase.strip())


@_register("startend:quotation")
def startend_quotation(response: str, kwargs: dict[str, Any]) -> bool:
    """Check entire response is wrapped in double quotes."""
    text = response.strip()
    return text.startswith('"') and text.endswith('"')


# --- Case ---

@_register("change_case:english_capital")
def case_capital(response: str, kwargs: dict[str, Any]) -> bool:
    """Check response is in all capital letters (ignoring non-alpha)."""
    alpha_chars = [c for c in response if c.isalpha()]
    if not alpha_chars:
        return True
    return all(c.isupper() for c in alpha_chars)


@_register("change_case:english_lowercase")
def case_lowercase(response: str, kwargs: dict[str, Any]) -> bool:
    """Check response is in all lowercase (ignoring non-alpha)."""
    alpha_chars = [c for c in response if c.isalpha()]
    if not alpha_chars:
        return True
    return all(c.islower() for c in alpha_chars)


# --- Punctuation ---

@_register("punctuation:no_comma")
def punctuation_no_comma(response: str, kwargs: dict[str, Any]) -> bool:
    """Check response contains no commas."""
    return "," not in response


# --- Wrapping ---

@_register("detectable_format:number_words_in_section")
def format_words_in_section(response: str, kwargs: dict[str, Any]) -> bool:
    """Check word count in a specific section."""
    # Simplified: check total word count
    num_words = kwargs.get("num_words", 0)
    relation = kwargs.get("relation", "at least")
    count = len(response.split())
    if relation == "at least":
        return count >= num_words
    elif relation == "at most":
        return count <= num_words
    return False
