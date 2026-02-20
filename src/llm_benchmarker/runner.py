"""Async benchmark execution engine."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from .benchmarks.base import BaseBenchmark, BenchmarkResult, BenchmarkSample, SampleResult
from .client import InstrumentedClient, RequestMetrics
from .config import BenchmarkConfig
from .metrics import aggregate_metrics

console = Console()


class BenchmarkRunner:
    """Runs benchmarks with controlled concurrency using asyncio."""

    def __init__(self, client: InstrumentedClient, config: BenchmarkConfig) -> None:
        self._client = client
        self._config = config
        self._semaphore = asyncio.Semaphore(config.run.concurrency)

    async def run_benchmark(self, benchmark: BaseBenchmark) -> BenchmarkResult:
        """Load dataset and run all samples for a single benchmark."""
        console.print(f"\n[bold blue]Running benchmark:[/] {benchmark.name}")

        samples = await benchmark.load_dataset(
            str(self._config.run.data_dir),
            self._config.run.max_samples,
        )
        console.print(f"  Loaded {len(samples)} samples (concurrency={self._config.run.concurrency})")

        wall_start = time.perf_counter()

        results: list[SampleResult] = []
        all_metrics: list[RequestMetrics] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(benchmark.name, total=len(samples))

            async def _run_one(sample: BenchmarkSample) -> SampleResult:
                result = await self._run_sample(benchmark, sample)
                progress.advance(task)
                return result

            tasks = [_run_one(s) for s in samples]
            gathered = await asyncio.gather(*tasks, return_exceptions=True)

        for i, item in enumerate(gathered):
            if isinstance(item, Exception):
                sr = SampleResult(
                    id=samples[i].id,
                    correct=False,
                    predicted="",
                    expected="",
                    details={"error": str(item)},
                    metrics=RequestMetrics(error=str(item)),
                )
                results.append(sr)
                if sr.metrics:
                    all_metrics.append(sr.metrics)
            else:
                results.append(item)
                if item.metrics:
                    all_metrics.append(item.metrics)

        wall_time = time.perf_counter() - wall_start
        agg = aggregate_metrics(all_metrics, wall_time)

        correct = sum(1 for r in results if r.correct)
        accuracy = correct / len(results) if results else 0.0
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0

        result = BenchmarkResult(
            name=benchmark.name,
            num_samples=len(results),
            accuracy=accuracy,
            score=avg_score,
            sample_results=results,
            aggregated_metrics=agg,
        )

        console.print(f"  [green]Done:[/] accuracy={accuracy:.4f} ({correct}/{len(results)}), "
                       f"wall_time={wall_time:.1f}s")

        return result

    async def _run_sample(
        self, benchmark: BaseBenchmark, sample: BenchmarkSample
    ) -> SampleResult:
        """Run a single sample with semaphore-controlled concurrency."""
        async with self._semaphore:
            if benchmark.is_multi_turn:
                return await benchmark.multi_turn(sample, self._client, self._config)

            # Build messages
            messages = list(sample.messages)
            sys_prompt = benchmark.system_prompt()
            if sys_prompt and (not messages or messages[0].get("role") != "system"):
                messages.insert(0, {"role": "system", "content": sys_prompt})

            # Build request kwargs
            kwargs: dict[str, Any] = {}
            kwargs.update(benchmark.get_request_kwargs(sample))
            kwargs.setdefault("max_tokens", self._config.run.max_tokens)
            if self._config.run.temperature is not None:
                kwargs.setdefault("temperature", self._config.run.temperature)
            if self._config.run.seed is not None:
                kwargs.setdefault("seed", self._config.run.seed)

            response, metrics = await self._client.chat_completion(
                messages,
                stream=self._config.run.streaming,
                **kwargs,
            )

            if metrics.error:
                return SampleResult(
                    id=sample.id,
                    correct=False,
                    predicted=response,
                    expected=sample.metadata.get("expected", ""),
                    details={"error": metrics.error},
                    metrics=metrics,
                )

            result = await benchmark.evaluate(sample, response)
            result.metrics = metrics
            return result
