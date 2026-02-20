"""Tests for benchmark registration and base classes."""

import pytest

from llm_benchmarker.benchmarks.base import (
    BaseBenchmark,
    BenchmarkResult,
    BenchmarkSample,
    SampleResult,
)
from llm_benchmarker.client import RequestMetrics
from llm_benchmarker.metrics import aggregate_metrics, compute_pass_at_k, confidence_interval_95


class TestBenchmarkRegistry:
    def test_list_benchmarks(self):
        from llm_benchmarker.benchmarks import list_benchmarks
        benchmarks = list_benchmarks()
        names = [b["name"] for b in benchmarks]
        assert "mmlu" in names
        assert "humaneval" in names
        assert "ifeval" in names

    def test_get_benchmark(self):
        from llm_benchmarker.benchmarks import get_benchmark
        b = get_benchmark("mmlu")
        assert b.name == "mmlu"
        assert b.tier == 1

    def test_get_unknown(self):
        from llm_benchmarker.benchmarks import get_benchmark
        with pytest.raises(KeyError, match="Unknown benchmark"):
            get_benchmark("nonexistent_benchmark_xyz")


class TestDataClasses:
    def test_sample_result_to_dict(self):
        sr = SampleResult(
            id="test_1",
            correct=True,
            score=1.0,
            predicted="A",
            expected="A",
            metrics=RequestMetrics(ttft_seconds=0.1, total_latency_seconds=0.5),
        )
        d = sr.to_dict()
        assert d["id"] == "test_1"
        assert d["correct"] is True
        assert "metrics" in d
        assert d["metrics"]["ttft_seconds"] == 0.1

    def test_benchmark_result_to_dict(self):
        result = BenchmarkResult(
            name="test",
            num_samples=10,
            accuracy=0.9,
            score=0.85,
        )
        d = result.to_dict()
        assert d["name"] == "test"
        assert d["accuracy"] == 0.9
        assert d["score"] == 0.85


class TestMetrics:
    def test_aggregate_empty(self):
        agg = aggregate_metrics([])
        assert agg.total_requests == 0
        assert agg.failed_requests == 0

    def test_aggregate_basic(self):
        metrics = [
            RequestMetrics(ttft_seconds=0.1, total_latency_seconds=0.5, prompt_tokens=100, completion_tokens=50),
            RequestMetrics(ttft_seconds=0.2, total_latency_seconds=1.0, prompt_tokens=200, completion_tokens=100),
        ]
        agg = aggregate_metrics(metrics, wall_time=1.5)
        assert agg.total_requests == 2
        assert agg.failed_requests == 0
        assert agg.total_prompt_tokens == 300
        assert agg.total_completion_tokens == 150
        assert agg.ttft_mean == pytest.approx(0.15)
        assert agg.wall_time_seconds == 1.5
        assert agg.effective_throughput_rps == pytest.approx(2 / 1.5)

    def test_aggregate_with_failures(self):
        metrics = [
            RequestMetrics(total_latency_seconds=0.5, prompt_tokens=100, completion_tokens=50),
            RequestMetrics(error="timeout"),
        ]
        agg = aggregate_metrics(metrics)
        assert agg.total_requests == 2
        assert agg.failed_requests == 1

    def test_pass_at_k(self):
        assert compute_pass_at_k(10, 10, 1) == 1.0
        assert compute_pass_at_k(10, 0, 1) == 0.0
        assert 0 < compute_pass_at_k(10, 5, 1) < 1

    def test_confidence_interval(self):
        scores = [1.0, 1.0, 0.0, 1.0, 0.0]
        lo, hi = confidence_interval_95(scores)
        assert lo < 0.6
        assert hi > 0.6
        assert lo < hi
