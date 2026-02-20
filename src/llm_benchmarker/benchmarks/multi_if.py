"""Multi-IF benchmark — 4,501 multi-turn multilingual instruction following."""

from __future__ import annotations

import json
from typing import Any

from . import register
from .base import BaseBenchmark, BenchmarkSample, SampleResult
from ..client import InstrumentedClient, RequestMetrics
from ..config import BenchmarkConfig


@register
class MultiIFBenchmark(BaseBenchmark):
    name = "multi_if"
    description = "Multi-IF — 4,501 multi-turn multilingual instruction following (3 turns)"
    tier = 2
    is_multi_turn = True

    async def load_dataset(self, data_dir: str, max_samples: int | None = None) -> list[BenchmarkSample]:
        from datasets import load_dataset

        ds = load_dataset("facebook/Multi-IF", split="test", trust_remote_code=True)
        samples = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            turns = row.get("turns", [])
            instruction_id_lists = row.get("instruction_id_list", [])
            kwargs_lists = row.get("kwargs", [])

            # Parse kwargs
            parsed_kwargs = []
            for kw_list in kwargs_lists:
                if isinstance(kw_list, str):
                    try:
                        parsed_kwargs.append(json.loads(kw_list))
                    except (json.JSONDecodeError, TypeError):
                        parsed_kwargs.append([{}])
                elif isinstance(kw_list, list):
                    inner = []
                    for kw in kw_list:
                        if isinstance(kw, str):
                            try:
                                inner.append(json.loads(kw))
                            except (json.JSONDecodeError, TypeError):
                                inner.append({})
                        elif kw is None:
                            inner.append({})
                        else:
                            inner.append(kw)
                    parsed_kwargs.append(inner)
                else:
                    parsed_kwargs.append([{}])

            samples.append(BenchmarkSample(
                id=f"multi_if_{i}",
                messages=[],  # Built during multi_turn
                metadata={
                    "turns": turns,
                    "instruction_id_lists": instruction_id_lists,
                    "kwargs_lists": parsed_kwargs,
                },
            ))
        return samples

    async def evaluate(self, sample: BenchmarkSample, response: str) -> SampleResult:
        raise NotImplementedError("Use multi_turn for Multi-IF")

    async def multi_turn(
        self,
        sample: BenchmarkSample,
        client: InstrumentedClient,
        config: BenchmarkConfig,
    ) -> SampleResult:
        from ..evaluators.ifeval_rules import verify_instruction

        turns = sample.metadata["turns"]
        instruction_id_lists = sample.metadata["instruction_id_lists"]
        kwargs_lists = sample.metadata["kwargs_lists"]

        conversation: list[dict[str, str]] = []
        all_metrics: list[RequestMetrics] = []
        per_turn_results: list[dict[str, Any]] = []
        all_pass = True

        for turn_idx, turn_prompt in enumerate(turns):
            conversation.append({"role": "user", "content": turn_prompt})

            response, metrics = await client.chat_completion(
                conversation,
                stream=config.run.streaming,
                max_tokens=config.run.max_tokens,
                temperature=config.run.temperature,
                **({"seed": config.run.seed} if config.run.seed is not None else {}),
            )
            all_metrics.append(metrics)
            conversation.append({"role": "assistant", "content": response})

            # Evaluate this turn's constraints
            turn_pass = True
            if turn_idx < len(instruction_id_lists):
                iids = instruction_id_lists[turn_idx] if isinstance(instruction_id_lists[turn_idx], list) else [instruction_id_lists[turn_idx]]
                kws = kwargs_lists[turn_idx] if turn_idx < len(kwargs_lists) else [{}] * len(iids)
                if not isinstance(kws, list):
                    kws = [kws]

                for iid, kw in zip(iids, kws):
                    if not verify_instruction(response, iid, kw if kw else {}):
                        turn_pass = False

            if not turn_pass:
                all_pass = False
            per_turn_results.append({"turn": turn_idx, "pass": turn_pass})

        # Use the last turn's metrics as primary
        primary_metrics = all_metrics[-1] if all_metrics else None

        return SampleResult(
            id=sample.id,
            correct=all_pass,
            score=1.0 if all_pass else 0.0,
            predicted=conversation[-1]["content"][:200] if conversation else "",
            expected="all turns pass",
            details={"per_turn": per_turn_results, "num_turns": len(turns)},
            metrics=primary_metrics,
        )
