"""SWE-bench Pro — 1,865 extended SWE tasks."""

from __future__ import annotations

from . import register
from .swe_bench import SWEBenchBenchmark
from .base import BenchmarkSample


@register
class SWEBenchProBenchmark(SWEBenchBenchmark):
    name = "swe_bench_pro"
    description = "SWE-bench Pro — 1,865 extended real-world software engineering tasks"

    async def load_dataset(self, data_dir: str, max_samples: int | None = None) -> list[BenchmarkSample]:
        import json
        from datasets import load_dataset

        ds = load_dataset("SWE-bench/SWE-bench", split="test", trust_remote_code=True)
        samples = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            instance_id = row.get("instance_id", f"swe_pro_{i}")
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
