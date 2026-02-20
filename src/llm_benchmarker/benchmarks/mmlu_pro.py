"""MMLU-Pro benchmark — 12,000+ MCQ with 10 options."""

from __future__ import annotations

from typing import Any

from . import register
from .base import BaseBenchmark, BenchmarkSample, SampleResult

CHOICES_10 = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def format_mcq_10(question: str, options: list[str]) -> str:
    lines = [question, ""]
    for letter, opt in zip(CHOICES_10, options):
        lines.append(f"{letter}. {opt}")
    lines.append("\nAnswer with the letter only.")
    return "\n".join(lines)


@register
class MMLUProBenchmark(BaseBenchmark):
    name = "mmlu_pro"
    description = "MMLU-Pro — 12,000+ MCQ with 10 options per question"
    tier = 1

    async def load_dataset(self, data_dir: str, max_samples: int | None = None) -> list[BenchmarkSample]:
        from datasets import load_dataset

        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
        samples = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            options = row["options"]
            question = row["question"]
            expected = row["answer"]
            category = row.get("category", "unknown")

            prompt = format_mcq_10(question, options)
            samples.append(BenchmarkSample(
                id=f"mmlu_pro_{i}",
                messages=[{"role": "user", "content": prompt}],
                metadata={"expected": expected, "category": category},
            ))
        return samples

    async def evaluate(self, sample: BenchmarkSample, response: str) -> SampleResult:
        from ..evaluators.mcq import evaluate_mcq, extract_mcq_answer

        expected = sample.metadata["expected"]
        correct = evaluate_mcq(response, expected, num_options=10)
        predicted = extract_mcq_answer(response, num_options=10) or ""

        return SampleResult(
            id=sample.id,
            correct=correct,
            score=1.0 if correct else 0.0,
            predicted=predicted,
            expected=expected,
            details={"category": sample.metadata.get("category", "")},
        )

    def get_request_kwargs(self, sample: BenchmarkSample) -> dict[str, Any]:
        return {"max_tokens": 64}

    def system_prompt(self) -> str | None:
        return "You are a knowledgeable assistant. Answer multiple-choice questions with only the letter of the correct answer."
