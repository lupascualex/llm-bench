"""GraphWalks benchmark — 1,150 graph reasoning tasks."""

from __future__ import annotations

from typing import Any

from . import register
from .base import BaseBenchmark, BenchmarkSample, SampleResult


@register
class GraphWalksBenchmark(BaseBenchmark):
    name = "graphwalks"
    description = "GraphWalks — 1,150 graph reasoning tasks with set comparison"
    tier = 1

    async def load_dataset(self, data_dir: str, max_samples: int | None = None) -> list[BenchmarkSample]:
        from datasets import load_dataset

        ds = load_dataset("openai/graphwalks", split="train", trust_remote_code=True)
        samples = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            prompt = row.get("prompt", row.get("question", ""))
            expected = row.get("answer", row.get("expected", ""))

            samples.append(BenchmarkSample(
                id=f"graphwalks_{i}",
                messages=[{"role": "user", "content": prompt}],
                metadata={"expected": str(expected)},
            ))
        return samples

    async def evaluate(self, sample: BenchmarkSample, response: str) -> SampleResult:
        from ..evaluators.set_comparison import jaccard_similarity, set_exact_match

        expected = sample.metadata["expected"]
        exact = set_exact_match(response, expected)
        jaccard = jaccard_similarity(response, expected)

        return SampleResult(
            id=sample.id,
            correct=exact,
            score=jaccard,
            predicted=response[:200],
            expected=expected[:200],
            details={"jaccard": jaccard, "exact_match": exact},
        )

    def get_request_kwargs(self, sample: BenchmarkSample) -> dict[str, Any]:
        return {"max_tokens": 512}
