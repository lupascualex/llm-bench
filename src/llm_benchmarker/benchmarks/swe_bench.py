"""SWE-bench Verified — 500 real-world software engineering tasks."""

from __future__ import annotations

import json
from typing import Any

from . import register
from .base import BenchmarkSample, SampleResult
from .docker_base import DockerBenchmark
from ..client import InstrumentedClient, RequestMetrics
from ..config import BenchmarkConfig


MAX_TURNS = 30


@register
class SWEBenchBenchmark(DockerBenchmark):
    name = "swe_bench"
    description = "SWE-bench Verified — 500 real-world software engineering tasks in Docker"
    docker_image = "python:3.11"

    async def load_dataset(self, data_dir: str, max_samples: int | None = None) -> list[BenchmarkSample]:
        from datasets import load_dataset

        ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test", trust_remote_code=True)
        samples = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            instance_id = row.get("instance_id", f"swe_{i}")
            problem_statement = row.get("problem_statement", "")
            repo = row.get("repo", "")
            base_commit = row.get("base_commit", "")
            patch = row.get("patch", "")
            test_patch = row.get("test_patch", "")
            fail_to_pass = row.get("FAIL_TO_PASS", "")
            pass_to_pass = row.get("PASS_TO_PASS", "")

            if isinstance(fail_to_pass, str):
                try:
                    fail_to_pass = json.loads(fail_to_pass)
                except (json.JSONDecodeError, TypeError):
                    fail_to_pass = []

            if isinstance(pass_to_pass, str):
                try:
                    pass_to_pass = json.loads(pass_to_pass)
                except (json.JSONDecodeError, TypeError):
                    pass_to_pass = []

            samples.append(BenchmarkSample(
                id=instance_id,
                messages=[{"role": "user", "content": problem_statement}],
                metadata={
                    "repo": repo,
                    "base_commit": base_commit,
                    "gold_patch": patch,
                    "test_patch": test_patch,
                    "fail_to_pass": fail_to_pass,
                    "pass_to_pass": pass_to_pass,
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

            repo = sample.metadata["repo"]
            base_commit = sample.metadata["base_commit"]

            # Setup: clone repo and checkout base commit
            setup_cmds = [
                f"git clone https://github.com/{repo}.git /workspace/repo",
                f"cd /workspace/repo && git checkout {base_commit}",
            ]
            for cmd in setup_cmds:
                exit_code, output = await self._exec_in_container(container_id, cmd, timeout=120)
                if exit_code != 0:
                    return SampleResult(
                        id=sample.id,
                        correct=False,
                        details={"error": f"Setup failed: {output[:500]}"},
                    )

            problem = sample.messages[0]["content"]
            system_msg = (
                "You are an expert software engineer fixing a bug in a repository at /workspace/repo. "
                "Generate a unified diff patch that fixes the issue. "
                "Wrap the patch in ```diff\\n...\\n```."
            )

            conversation = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Fix this issue:\n\n{problem}"},
            ]

            # Single-turn: get the patch
            response, metrics = await client.chat_completion(
                conversation,
                stream=config.run.streaming,
                max_tokens=4096,
                temperature=config.run.temperature,
                **({"seed": config.run.seed} if config.run.seed is not None else {}),
            )
            all_metrics.append(metrics)

            # Extract and apply patch
            patch = self._extract_patch(response)
            if not patch:
                return SampleResult(
                    id=sample.id,
                    correct=False,
                    predicted=response[:300],
                    details={"error": "No patch extracted from response"},
                    metrics=metrics,
                )

            # Apply patch
            # Write patch to file and apply
            import shlex
            escaped_patch = patch.replace("'", "'\\''")
            apply_cmd = f"cd /workspace/repo && echo '{escaped_patch}' | git apply -"
            exit_code, output = await self._exec_in_container(container_id, apply_cmd)

            if exit_code != 0:
                # Try with --3way
                apply_cmd_3way = f"cd /workspace/repo && echo '{escaped_patch}' | git apply --3way -"
                exit_code, output = await self._exec_in_container(container_id, apply_cmd_3way)

            # Apply test patch if provided
            test_patch = sample.metadata.get("test_patch", "")
            if test_patch:
                escaped_test = test_patch.replace("'", "'\\''")
                test_cmd = f"cd /workspace/repo && echo '{escaped_test}' | git apply -"
                await self._exec_in_container(container_id, test_cmd)

            # Run FAIL_TO_PASS tests
            fail_to_pass = sample.metadata.get("fail_to_pass", [])
            pass_results = []
            for test in fail_to_pass:
                test_cmd = f"cd /workspace/repo && python -m pytest {test} -x --tb=short 2>&1 | tail -20"
                exit_code, output = await self._exec_in_container(container_id, test_cmd, timeout=120)
                pass_results.append(exit_code == 0)

            # Run PASS_TO_PASS tests (should still pass)
            pass_to_pass = sample.metadata.get("pass_to_pass", [])
            regression_results = []
            for test in pass_to_pass[:10]:  # Limit regression test count
                test_cmd = f"cd /workspace/repo && python -m pytest {test} -x --tb=short 2>&1 | tail -20"
                exit_code, output = await self._exec_in_container(container_id, test_cmd, timeout=120)
                regression_results.append(exit_code == 0)

            f2p_pass = all(pass_results) if pass_results else False
            p2p_pass = all(regression_results) if regression_results else True
            correct = f2p_pass and p2p_pass

            return SampleResult(
                id=sample.id,
                correct=correct,
                score=1.0 if correct else 0.0,
                predicted=patch[:500],
                expected="passing tests",
                details={
                    "fail_to_pass_results": pass_results,
                    "pass_to_pass_results": regression_results,
                    "f2p_pass": f2p_pass,
                    "p2p_pass": p2p_pass,
                },
                metrics=all_metrics[-1] if all_metrics else None,
            )

        finally:
            if container_id:
                await self._remove_container(container_id)

    @staticmethod
    def _extract_patch(response: str) -> str | None:
        """Extract a diff/patch from model response."""
        import re
        match = re.search(r'```(?:diff|patch)?\s*\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Look for diff-like content
        lines = response.split("\n")
        diff_lines = [l for l in lines if l.startswith(("+", "-", "@@", "diff ", "---", "+++"))]
        if len(diff_lines) >= 3:
            return "\n".join(diff_lines)
        return None
