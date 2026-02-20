"""BFCL benchmark — Berkeley Function Calling Leaderboard with AST evaluation."""

from __future__ import annotations

import json
from typing import Any

from . import register
from .base import BaseBenchmark, BenchmarkSample, SampleResult
from ..client import InstrumentedClient
from ..config import BenchmarkConfig


@register
class BFCLBenchmark(BaseBenchmark):
    name = "bfcl"
    description = "BFCL — Berkeley Function Calling Leaderboard with AST-based evaluation"
    tier = 3
    requires_tools = True
    is_multi_turn = True  # Need raw response for tool_calls

    async def load_dataset(self, data_dir: str, max_samples: int | None = None) -> list[BenchmarkSample]:
        from datasets import load_dataset

        ds = load_dataset(
            "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
            split="test",
            trust_remote_code=True,
        )
        samples = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            question = row.get("question", "")
            functions = row.get("function", [])
            expected = row.get("answer", row.get("ground_truth", ""))

            # Convert function definitions to OpenAI tools format
            tools = []
            if isinstance(functions, str):
                try:
                    functions = json.loads(functions)
                except (json.JSONDecodeError, TypeError):
                    functions = []
            if isinstance(functions, list):
                for fn in functions:
                    if isinstance(fn, dict):
                        tools.append({
                            "type": "function",
                            "function": fn,
                        })

            # Parse expected answer
            if isinstance(expected, str):
                try:
                    expected_parsed = json.loads(expected)
                except (json.JSONDecodeError, TypeError):
                    expected_parsed = expected
            else:
                expected_parsed = expected

            if isinstance(question, list):
                messages = question
            elif isinstance(question, str):
                messages = [{"role": "user", "content": question}]
            else:
                messages = [{"role": "user", "content": str(question)}]

            samples.append(BenchmarkSample(
                id=f"bfcl_{i}",
                messages=messages,
                metadata={
                    "tools": tools,
                    "expected": expected_parsed,
                    "expected_raw": str(expected),
                },
            ))
        return samples

    async def evaluate(self, sample: BenchmarkSample, response: str) -> SampleResult:
        raise NotImplementedError("Use multi_turn for BFCL")

    async def multi_turn(
        self,
        sample: BenchmarkSample,
        client: InstrumentedClient,
        config: BenchmarkConfig,
    ) -> SampleResult:
        from ..evaluators.ast_eval import compare_function_calls, parse_function_call

        tools = sample.metadata.get("tools", [])
        expected = sample.metadata["expected"]

        kwargs: dict[str, Any] = {
            "max_tokens": 512,
            "temperature": config.run.temperature,
        }
        if config.run.seed is not None:
            kwargs["seed"] = config.run.seed
        if tools:
            kwargs["tools"] = tools

        response, metrics = await client.chat_completion_raw(
            sample.messages,
            **kwargs,
        )

        # Extract tool calls from the response
        predicted_calls: list[dict[str, Any]] = []
        if response.choices:
            msg = response.choices[0].message
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = tc.function.arguments
                    predicted_calls.append({
                        "name": tc.function.name,
                        "arguments": args,
                    })
            elif msg.content:
                # Try parsing function calls from text
                predicted_calls = parse_function_call(msg.content)

        # Parse expected calls
        expected_calls: list[dict[str, Any]] = []
        if isinstance(expected, list):
            expected_calls = expected
        elif isinstance(expected, dict):
            expected_calls = [expected]
        elif isinstance(expected, str):
            expected_calls = parse_function_call(expected)

        match, details = compare_function_calls(predicted_calls, expected_calls)

        return SampleResult(
            id=sample.id,
            correct=match,
            score=1.0 if match else 0.0,
            predicted=str(predicted_calls)[:300],
            expected=str(expected_calls)[:300],
            details=details,
            metrics=metrics,
        )
