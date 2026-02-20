"""Tests for the benchmark runner with a dummy benchmark."""

import pytest
from unittest.mock import AsyncMock, patch

from llm_benchmarker.benchmarks.base import BaseBenchmark, BenchmarkSample, SampleResult
from llm_benchmarker.client import InstrumentedClient, RequestMetrics
from llm_benchmarker.config import BenchmarkConfig, RunConfig, VLLMConfig
from llm_benchmarker.runner import BenchmarkRunner


class DummyBenchmark(BaseBenchmark):
    name = "dummy"
    description = "A test benchmark"
    tier = 0

    async def load_dataset(self, data_dir, max_samples=None):
        samples = [
            BenchmarkSample(
                id=f"dummy_{i}",
                messages=[{"role": "user", "content": f"Question {i}"}],
                metadata={"expected": f"Answer {i}"},
            )
            for i in range(5)
        ]
        if max_samples:
            samples = samples[:max_samples]
        return samples

    async def evaluate(self, sample, response):
        expected = sample.metadata["expected"]
        correct = response.strip() == expected
        return SampleResult(
            id=sample.id,
            correct=correct,
            score=1.0 if correct else 0.0,
            predicted=response,
            expected=expected,
        )


@pytest.fixture
def config():
    return BenchmarkConfig(
        vllm=VLLMConfig(model="test-model"),
        run=RunConfig(concurrency=2, max_samples=3, streaming=False),
    )


class TestBenchmarkRunner:
    @pytest.mark.asyncio
    async def test_run_benchmark(self, config):
        client = InstrumentedClient(config.vllm)
        runner = BenchmarkRunner(client, config)
        benchmark = DummyBenchmark()

        async def mock_chat(*args, **kwargs):
            return "Answer 0", RequestMetrics(prompt_tokens=10, completion_tokens=5)

        with patch.object(client, "chat_completion", side_effect=mock_chat):
            result = await runner.run_benchmark(benchmark)

        assert result.name == "dummy"
        assert result.num_samples == 3
        assert result.aggregated_metrics is not None
        assert result.aggregated_metrics.total_requests == 3
        # Only the first sample matches "Answer 0"
        assert any(sr.correct for sr in result.sample_results)

    @pytest.mark.asyncio
    async def test_handles_errors(self, config):
        client = InstrumentedClient(config.vllm)
        runner = BenchmarkRunner(client, config)
        benchmark = DummyBenchmark()

        async def mock_chat(*args, **kwargs):
            return "", RequestMetrics(error="timeout")

        with patch.object(client, "chat_completion", side_effect=mock_chat):
            result = await runner.run_benchmark(benchmark)

        assert result.num_samples == 3
        assert all(not sr.correct for sr in result.sample_results)
