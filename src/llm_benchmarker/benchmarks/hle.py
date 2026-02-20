"""HLE (Humanity's Last Exam) benchmark — 2,500 MCQ + short answer."""

from __future__ import annotations

from typing import Any

from . import register
from .base import BaseBenchmark, BenchmarkSample, SampleResult

CHOICES = ["A", "B", "C", "D"]


@register
class HLEBenchmark(BaseBenchmark):
    name = "hle"
    description = "Humanity's Last Exam — 2,500 expert-level MCQ and short-answer questions"
    tier = 1

    async def load_dataset(self, data_dir: str, max_samples: int | None = None) -> list[BenchmarkSample]:
        from datasets import load_dataset

        ds = load_dataset("cais/hle", split="test", trust_remote_code=True)
        samples = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            question = row["question"]
            answer = row.get("answer", "")
            question_type = row.get("question_type", "")
            image = row.get("image")

            # Build prompt based on question type
            if question_type == "multiple_choice":
                prompt = question + "\n\nAnswer with the letter only."
            else:
                prompt = question + "\n\nProvide a concise answer."

            # Skip image-based questions for now (text-only)
            if image is not None:
                continue

            samples.append(BenchmarkSample(
                id=f"hle_{i}",
                messages=[{"role": "user", "content": prompt}],
                metadata={
                    "expected": answer,
                    "question_type": question_type,
                },
            ))
        return samples

    async def evaluate(self, sample: BenchmarkSample, response: str) -> SampleResult:
        expected = sample.metadata["expected"]
        question_type = sample.metadata["question_type"]

        if question_type == "multiple_choice":
            from ..evaluators.mcq import evaluate_mcq, extract_mcq_answer
            correct = evaluate_mcq(response, expected, num_options=4)
            predicted = extract_mcq_answer(response, num_options=4) or ""
        else:
            from ..evaluators.exact_match import exact_match, contains_match
            predicted = response.strip()
            correct = exact_match(predicted, expected) or contains_match(predicted, expected)

        return SampleResult(
            id=sample.id,
            correct=correct,
            score=1.0 if correct else 0.0,
            predicted=predicted[:200],
            expected=expected,
            details={"question_type": question_type},
        )

    def get_request_kwargs(self, sample: BenchmarkSample) -> dict[str, Any]:
        return {"max_tokens": 256}
