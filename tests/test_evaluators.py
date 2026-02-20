"""Tests for evaluation functions."""

import pytest


class TestExactMatch:
    def test_exact(self):
        from llm_benchmarker.evaluators.exact_match import exact_match
        assert exact_match("hello world", "Hello World")
        assert exact_match("  hello  world  ", "hello world")
        assert not exact_match("hello", "world")

    def test_contains(self):
        from llm_benchmarker.evaluators.exact_match import contains_match
        assert contains_match("the answer is 42", "42")
        assert contains_match("The Answer Is 42!", "answer is 42")
        assert not contains_match("hello", "world")

    def test_normalize_punctuation(self):
        from llm_benchmarker.evaluators.exact_match import normalize
        assert normalize("hello, world!") == "hello world"
        assert normalize("it's a test.") == "its a test"


class TestMCQ:
    def test_extract_letter(self):
        from llm_benchmarker.evaluators.mcq import extract_mcq_answer
        assert extract_mcq_answer("The answer is B") == "B"
        assert extract_mcq_answer("A") == "A"
        assert extract_mcq_answer("(C)") == "C"
        assert extract_mcq_answer("Answer: D") == "D"

    def test_extract_10_option(self):
        from llm_benchmarker.evaluators.mcq import extract_mcq_answer
        assert extract_mcq_answer("The answer is H", num_options=10) == "H"
        assert extract_mcq_answer("J", num_options=10) == "J"

    def test_evaluate(self):
        from llm_benchmarker.evaluators.mcq import evaluate_mcq
        assert evaluate_mcq("The answer is B", "B")
        assert not evaluate_mcq("The answer is A", "B")
        assert evaluate_mcq("B.", "B")


class TestSetComparison:
    def test_exact_match(self):
        from llm_benchmarker.evaluators.set_comparison import set_exact_match
        assert set_exact_match("{a, b, c}", "{c, b, a}")
        assert not set_exact_match("{a, b}", "{a, b, c}")

    def test_jaccard(self):
        from llm_benchmarker.evaluators.set_comparison import jaccard_similarity
        assert jaccard_similarity("{a, b, c}", "{a, b, c}") == 1.0
        assert jaccard_similarity("{a, b}", "{a, b, c}") == pytest.approx(2/3)
        assert jaccard_similarity("{a}", "{b}") == 0.0

    def test_extract_set(self):
        from llm_benchmarker.evaluators.set_comparison import extract_set
        assert extract_set("[a, b, c]") == {"a", "b", "c"}
        assert extract_set("a\nb\nc") == {"a", "b", "c"}


class TestIFEvalRules:
    def test_keywords_existence(self):
        from llm_benchmarker.evaluators.ifeval_rules import verify_instruction
        assert verify_instruction("hello world foo", "keywords:existence", {"keywords": ["hello", "world"]})
        assert not verify_instruction("hello bar", "keywords:existence", {"keywords": ["hello", "world"]})

    def test_keywords_forbidden(self):
        from llm_benchmarker.evaluators.ifeval_rules import verify_instruction
        assert verify_instruction("hello world", "keywords:forbidden_words", {"forbidden_words": ["foo"]})
        assert not verify_instruction("hello foo world", "keywords:forbidden_words", {"forbidden_words": ["foo"]})

    def test_length_words(self):
        from llm_benchmarker.evaluators.ifeval_rules import verify_instruction
        assert verify_instruction("one two three four five", "length_constraints:number_words",
                                  {"num_words": 5, "relation": "at least"})
        assert not verify_instruction("one two", "length_constraints:number_words",
                                      {"num_words": 5, "relation": "at least"})

    def test_no_comma(self):
        from llm_benchmarker.evaluators.ifeval_rules import verify_instruction
        assert verify_instruction("hello world", "punctuation:no_comma", {})
        assert not verify_instruction("hello, world", "punctuation:no_comma", {})

    def test_case_capital(self):
        from llm_benchmarker.evaluators.ifeval_rules import verify_instruction
        assert verify_instruction("HELLO WORLD", "change_case:english_capital", {})
        assert not verify_instruction("Hello World", "change_case:english_capital", {})

    def test_json_format(self):
        from llm_benchmarker.evaluators.ifeval_rules import verify_instruction
        assert verify_instruction('{"key": "value"}', "detectable_format:json_format", {})
        assert not verify_instruction("not json at all", "detectable_format:json_format", {})

    def test_end_checker(self):
        from llm_benchmarker.evaluators.ifeval_rules import verify_instruction
        assert verify_instruction("some text THE END", "startend:end_checker", {"end_phrase": "THE END"})
        assert not verify_instruction("THE END is not here", "startend:end_checker", {"end_phrase": "THE END"})


class TestASTEval:
    def test_parse_json(self):
        from llm_benchmarker.evaluators.ast_eval import parse_function_call
        calls = parse_function_call('[{"name": "get_weather", "arguments": {"city": "SF"}}]')
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"]["city"] == "SF"

    def test_compare_match(self):
        from llm_benchmarker.evaluators.ast_eval import compare_function_calls
        pred = [{"name": "f", "arguments": {"x": 1}}]
        exp = [{"name": "f", "arguments": {"x": 1}}]
        match, _ = compare_function_calls(pred, exp)
        assert match

    def test_compare_mismatch(self):
        from llm_benchmarker.evaluators.ast_eval import compare_function_calls
        pred = [{"name": "f", "arguments": {"x": 1}}]
        exp = [{"name": "f", "arguments": {"x": 2}}]
        match, _ = compare_function_calls(pred, exp)
        assert not match


class TestCodeExecution:
    @pytest.mark.asyncio
    async def test_execute_success(self):
        from llm_benchmarker.evaluators.code_execution import execute_python
        success, output = await execute_python("print('hello')")
        assert success
        assert "hello" in output

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        from llm_benchmarker.evaluators.code_execution import execute_python
        success, output = await execute_python("raise ValueError('oops')")
        assert not success
        assert "oops" in output

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        from llm_benchmarker.evaluators.code_execution import execute_python
        success, output = await execute_python("import time; time.sleep(10)", timeout=0.5)
        assert not success
        assert "timed out" in output.lower()
