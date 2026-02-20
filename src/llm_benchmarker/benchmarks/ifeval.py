"""IFEval benchmark — ~500 instruction-following tasks with 25 verifiable constraints."""

from __future__ import annotations

import json
from typing import Any

from . import register
from .base import BaseBenchmark, BenchmarkSample, SampleResult


@register
class IFEvalBenchmark(BaseBenchmark):
    name = "ifeval"
    description = "IFEval — ~500 instruction-following tasks with verifiable constraints"
    tier = 1

    async def load_dataset(self, data_dir: str, max_samples: int | None = None) -> list[BenchmarkSample]:
        from datasets import load_dataset

        ds = load_dataset("google/IFEval", split="train", trust_remote_code=True)
        samples = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            prompt = row["prompt"]
            instruction_id_list = row.get("instruction_id_list", [])
            kwargs_list = row.get("kwargs", [])

            # Parse kwargs if they're strings
            parsed_kwargs = []
            for kw in kwargs_list:
                if isinstance(kw, str):
                    try:
                        parsed_kwargs.append(json.loads(kw))
                    except (json.JSONDecodeError, TypeError):
                        parsed_kwargs.append({})
                elif kw is None:
                    parsed_kwargs.append({})
                else:
                    parsed_kwargs.append(kw)

            samples.append(BenchmarkSample(
                id=f"ifeval_{i}",
                messages=[{"role": "user", "content": prompt}],
                metadata={
                    "instruction_id_list": instruction_id_list,
                    "kwargs": parsed_kwargs,
                },
            ))
        return samples

    async def evaluate(self, sample: BenchmarkSample, response: str) -> SampleResult:
        from ..evaluators.ifeval_rules import verify_instruction

        instruction_ids = sample.metadata["instruction_id_list"]
        kwargs_list = sample.metadata["kwargs"]

        per_instruction = []
        all_pass = True
        for iid, kw in zip(instruction_ids, kwargs_list):
            passed = verify_instruction(response, iid, kw)
            per_instruction.append({"instruction_id": iid, "pass": passed})
            if not passed:
                all_pass = False

        return SampleResult(
            id=sample.id,
            correct=all_pass,
            score=1.0 if all_pass else 0.0,
            predicted=response[:200],
            expected="all instructions followed",
            details={
                "per_instruction": per_instruction,
                "num_instructions": len(instruction_ids),
                "num_passed": sum(1 for p in per_instruction if p["pass"]),
            },
        )

    def get_request_kwargs(self, sample: BenchmarkSample) -> dict[str, Any]:
        return {"max_tokens": 2048}
