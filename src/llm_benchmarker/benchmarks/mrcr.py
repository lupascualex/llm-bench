"""MRCR benchmark — 2,400 multi-round long-context needle retrieval (4k-1M tokens)."""

from __future__ import annotations

from typing import Any

from . import register
from .base import BaseBenchmark, BenchmarkSample, SampleResult


@register
class MRCRBenchmark(BaseBenchmark):
    name = "mrcr"
    description = "MRCR — 2,400 multi-round long-context needle retrieval (4k to 1M tokens)"
    tier = 2

    async def load_dataset(self, data_dir: str, max_samples: int | None = None) -> list[BenchmarkSample]:
        from datasets import load_dataset

        ds = load_dataset("openai/mrcr", split="train", trust_remote_code=True)
        samples = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            # MRCR provides multi-round context as conversation messages
            messages = row.get("messages", [])
            expected = row.get("answer", row.get("expected", ""))

            if not messages:
                # Fallback: single prompt format
                prompt = row.get("prompt", row.get("question", ""))
                if prompt:
                    messages = [{"role": "user", "content": prompt}]

            samples.append(BenchmarkSample(
                id=f"mrcr_{i}",
                messages=messages,
                metadata={
                    "expected": str(expected),
                    "context_length": row.get("context_length", 0),
                },
            ))
        return samples

    async def evaluate(self, sample: BenchmarkSample, response: str) -> SampleResult:
        from ..evaluators.exact_match import contains_match, exact_match

        expected = sample.metadata["expected"]
        is_exact = exact_match(response, expected)
        is_contains = contains_match(response, expected)
        correct = is_exact or is_contains

        return SampleResult(
            id=sample.id,
            correct=correct,
            score=1.0 if correct else 0.0,
            predicted=response[:200],
            expected=expected[:200],
            details={
                "exact_match": is_exact,
                "contains_match": is_contains,
                "context_length": sample.metadata.get("context_length", 0),
            },
        )

    def get_request_kwargs(self, sample: BenchmarkSample) -> dict[str, Any]:
        return {"max_tokens": 256}
