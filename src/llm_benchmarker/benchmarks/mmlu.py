"""MMLU benchmark — 15,908 MCQ across 57 subjects."""

from __future__ import annotations

from typing import Any

from . import register
from .base import BaseBenchmark, BenchmarkSample, SampleResult


CHOICES = ["A", "B", "C", "D"]


def format_mcq(question: str, choices: list[str]) -> str:
    """Format a multiple-choice question."""
    lines = [question, ""]
    for letter, choice in zip(CHOICES, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("\nAnswer with the letter only.")
    return "\n".join(lines)


@register
class MMLUBenchmark(BaseBenchmark):
    name = "mmlu"
    description = "Massive Multitask Language Understanding — 15,908 MCQ, 57 subjects"
    tier = 1

    async def load_dataset(self, data_dir: str, max_samples: int | None = None) -> list[BenchmarkSample]:
        from datasets import load_dataset

        ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
        samples = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            choices = row["choices"]
            question = row["question"]
            answer_idx = row["answer"]
            expected = CHOICES[answer_idx]
            subject = row.get("subject", "unknown")

            prompt = format_mcq(question, choices)
            samples.append(BenchmarkSample(
                id=f"mmlu_{i}",
                messages=[{"role": "user", "content": prompt}],
                metadata={"expected": expected, "subject": subject},
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
            details={"subject": sample.metadata.get("subject", "")},
        )

    def get_request_kwargs(self, sample: BenchmarkSample) -> dict[str, Any]:
        return {"max_tokens": 32}

    def system_prompt(self) -> str | None:
        return "You are a knowledgeable assistant. Answer multiple-choice questions with only the letter of the correct answer."
