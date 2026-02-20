"""LongBench v2 — 503 long-context MCQ (8k-2M words)."""

from __future__ import annotations

from typing import Any

from . import register
from .base import BaseBenchmark, BenchmarkSample, SampleResult

CHOICES = ["A", "B", "C", "D"]


@register
class LongBenchV2Benchmark(BaseBenchmark):
    name = "longbench_v2"
    description = "LongBench v2 — 503 long-context MCQ spanning 8k to 2M words"
    tier = 1

    async def load_dataset(self, data_dir: str, max_samples: int | None = None) -> list[BenchmarkSample]:
        from datasets import load_dataset

        ds = load_dataset("THUDM/LongBench-v2", split="test", trust_remote_code=True)
        samples = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            context = row.get("context", "")
            question = row.get("question", "")
            choices = row.get("choices", [])
            answer = row.get("answer", "")

            # Build prompt with context + MCQ
            if choices:
                choice_text = "\n".join(f"{CHOICES[j]}. {c}" for j, c in enumerate(choices))
                prompt = f"{context}\n\nQuestion: {question}\n\n{choice_text}\n\nAnswer with the letter only."
            else:
                prompt = f"{context}\n\nQuestion: {question}\n\nAnswer with the letter only."

            samples.append(BenchmarkSample(
                id=f"longbench_v2_{i}",
                messages=[{"role": "user", "content": prompt}],
                metadata={"expected": answer},
            ))
        return samples

    async def evaluate(self, sample: BenchmarkSample, response: str) -> SampleResult:
        from ..evaluators.mcq import evaluate_mcq, extract_mcq_answer

        expected = sample.metadata["expected"]
        correct = evaluate_mcq(response, expected, num_options=4)
        predicted = extract_mcq_answer(response, num_options=4) or ""

        return SampleResult(
            id=sample.id,
            correct=correct,
            score=1.0 if correct else 0.0,
            predicted=predicted,
            expected=expected,
        )

    def get_request_kwargs(self, sample: BenchmarkSample) -> dict[str, Any]:
        return {"max_tokens": 32}
