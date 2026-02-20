"""HumanEval benchmark — 164 code generation tasks with pass@k."""

from __future__ import annotations

from typing import Any

from . import register
from .base import BaseBenchmark, BenchmarkSample, SampleResult


@register
class HumanEvalBenchmark(BaseBenchmark):
    name = "humaneval"
    description = "HumanEval — 164 Python code generation tasks with execution-based evaluation"
    tier = 1

    async def load_dataset(self, data_dir: str, max_samples: int | None = None) -> list[BenchmarkSample]:
        from datasets import load_dataset

        ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
        samples = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            prompt = row["prompt"]
            test = row["test"]
            entry_point = row["entry_point"]
            task_id = row["task_id"]

            user_msg = (
                "Complete the following Python function. "
                "Return only the function body (no explanation, no markdown).\n\n"
                f"{prompt}"
            )
            samples.append(BenchmarkSample(
                id=task_id,
                messages=[{"role": "user", "content": user_msg}],
                metadata={
                    "prompt": prompt,
                    "test": test,
                    "entry_point": entry_point,
                },
            ))
        return samples

    async def evaluate(self, sample: BenchmarkSample, response: str) -> SampleResult:
        from ..evaluators.code_execution import run_humaneval_test

        prompt = sample.metadata["prompt"]
        test = sample.metadata["test"]
        entry_point = sample.metadata["entry_point"]

        # Clean up response — remove markdown fences if present
        completion = response.strip()
        if completion.startswith("```"):
            lines = completion.split("\n")
            # Remove first and last lines (fences)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            completion = "\n".join(lines)

        # If the model repeated the function signature, try to extract just the body
        if completion.startswith("def "):
            # Use the completion as-is (it includes the full function)
            full_completion = completion
            # Remove the original prompt's function signature
            success, output = await run_humaneval_test(
                "", full_completion, test, entry_point
            )
        else:
            success, output = await run_humaneval_test(
                prompt, completion, test, entry_point
            )

        return SampleResult(
            id=sample.id,
            correct=success,
            score=1.0 if success else 0.0,
            predicted=completion[:500],
            expected=f"passes {entry_point} tests",
            details={"output": output[:500], "entry_point": entry_point},
        )

    def get_request_kwargs(self, sample: BenchmarkSample) -> dict[str, Any]:
        return {"max_tokens": 512}

    def system_prompt(self) -> str | None:
        return "You are an expert Python programmer. Complete the function as requested. Return only code, no explanations."
