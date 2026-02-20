"""Terminal-Bench — 89 CLI tasks in Docker containers."""

from __future__ import annotations

import json
from typing import Any

from . import register
from .base import BenchmarkSample, SampleResult
from .docker_base import DockerBenchmark
from ..client import InstrumentedClient, RequestMetrics
from ..config import BenchmarkConfig


MAX_TURNS = 15


@register
class TerminalBenchBenchmark(DockerBenchmark):
    name = "terminal_bench"
    description = "Terminal-Bench — 89 CLI tasks executed in Docker containers"
    docker_image = "ubuntu:22.04"

    async def load_dataset(self, data_dir: str, max_samples: int | None = None) -> list[BenchmarkSample]:
        from datasets import load_dataset

        try:
            ds = load_dataset("laude-institute/terminal-bench", split="test", trust_remote_code=True)
        except Exception:
            ds = load_dataset("laude-institute/terminal-bench", split="train", trust_remote_code=True)

        samples = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            task = row.get("task", row.get("instruction", ""))
            setup = row.get("setup", "")
            validation = row.get("validation", row.get("check", ""))

            samples.append(BenchmarkSample(
                id=f"terminal_bench_{i}",
                messages=[{"role": "user", "content": task}],
                metadata={
                    "setup": setup,
                    "validation": validation,
                },
            ))
        return samples

    async def multi_turn(
        self,
        sample: BenchmarkSample,
        client: InstrumentedClient,
        config: BenchmarkConfig,
    ) -> SampleResult:
        container_id = None
        all_metrics: list[RequestMetrics] = []

        try:
            container_id = await self._create_container(sample)

            # Run setup if provided
            setup = sample.metadata.get("setup", "")
            if setup:
                await self._exec_in_container(container_id, setup)

            task = sample.messages[0]["content"]
            system_msg = (
                "You are a Linux CLI expert. Execute commands to complete the task. "
                "Respond with a bash command to execute. Wrap commands in ```bash\\n...\\n```."
            )

            conversation = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Task: {task}\n\nYou have access to a bash shell. What command should I run?"},
            ]

            for turn in range(MAX_TURNS):
                response, metrics = await client.chat_completion(
                    conversation,
                    stream=config.run.streaming,
                    max_tokens=1024,
                    temperature=config.run.temperature,
                    **({"seed": config.run.seed} if config.run.seed is not None else {}),
                )
                all_metrics.append(metrics)
                conversation.append({"role": "assistant", "content": response})

                # Extract command from response
                command = self._extract_command(response)
                if not command:
                    break

                # Execute command in container
                exit_code, output = await self._exec_in_container(container_id, command)
                result_msg = f"Exit code: {exit_code}\nOutput:\n{output[:2000]}"
                conversation.append({"role": "user", "content": result_msg})

                if exit_code == 0 and "done" in response.lower():
                    break

            # Validate
            validation = sample.metadata.get("validation", "")
            if validation:
                exit_code, output = await self._exec_in_container(container_id, validation)
                correct = exit_code == 0
            else:
                correct = True

            primary_metrics = all_metrics[-1] if all_metrics else None
            return SampleResult(
                id=sample.id,
                correct=correct,
                score=1.0 if correct else 0.0,
                predicted=conversation[-1].get("content", "")[:200] if conversation else "",
                expected="task completed",
                details={"num_turns": len(all_metrics)},
                metrics=primary_metrics,
            )

        finally:
            if container_id:
                await self._remove_container(container_id)

    @staticmethod
    def _extract_command(response: str) -> str | None:
        """Extract a bash command from a model response."""
        import re
        match = re.search(r'```(?:bash|sh)?\s*\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: look for lines starting with $
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("$ "):
                return line[2:]
        return None
