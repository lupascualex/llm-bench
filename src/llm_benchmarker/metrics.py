"""Metrics collection, aggregation, and statistical utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .client import RequestMetrics


@dataclass
class AggregatedMetrics:
    """Summary statistics across all requests in a benchmark run."""

    ttft_p50: float | None = None
    ttft_p95: float | None = None
    ttft_p99: float | None = None
    ttft_mean: float | None = None

    generation_tps_mean: float | None = None
    generation_tps_p50: float | None = None

    prompt_tps_mean: float | None = None

    latency_p50: float | None = None
    latency_p95: float | None = None
    latency_p99: float | None = None
    latency_mean: float | None = None

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    wall_time_seconds: float = 0.0
    effective_throughput_rps: float | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def aggregate_metrics(
    request_metrics: list[RequestMetrics],
    wall_time: float = 0.0,
) -> AggregatedMetrics:
    """Compute summary statistics from a list of per-request metrics."""
    agg = AggregatedMetrics()
    agg.total_requests = len(request_metrics)
    agg.wall_time_seconds = wall_time

    successful = [m for m in request_metrics if m.error is None]
    agg.failed_requests = agg.total_requests - len(successful)

    if not successful:
        return agg

    agg.total_prompt_tokens = sum(m.prompt_tokens for m in successful)
    agg.total_completion_tokens = sum(m.completion_tokens for m in successful)

    # TTFT
    ttfts = [m.ttft_seconds for m in successful if m.ttft_seconds is not None]
    if ttfts:
        arr = np.array(ttfts)
        agg.ttft_p50 = float(np.percentile(arr, 50))
        agg.ttft_p95 = float(np.percentile(arr, 95))
        agg.ttft_p99 = float(np.percentile(arr, 99))
        agg.ttft_mean = float(np.mean(arr))

    # Generation tokens/sec
    gen_tps = [m.tokens_per_second_generation for m in successful if m.tokens_per_second_generation is not None]
    if gen_tps:
        arr = np.array(gen_tps)
        agg.generation_tps_mean = float(np.mean(arr))
        agg.generation_tps_p50 = float(np.percentile(arr, 50))

    # Prompt tokens/sec
    prompt_tps = [m.tokens_per_second_prompt for m in successful if m.tokens_per_second_prompt is not None]
    if prompt_tps:
        agg.prompt_tps_mean = float(np.mean(prompt_tps))

    # Total latency
    latencies = [m.total_latency_seconds for m in successful]
    if latencies:
        arr = np.array(latencies)
        agg.latency_p50 = float(np.percentile(arr, 50))
        agg.latency_p95 = float(np.percentile(arr, 95))
        agg.latency_p99 = float(np.percentile(arr, 99))
        agg.latency_mean = float(np.mean(arr))

    # Effective throughput
    if wall_time > 0:
        agg.effective_throughput_rps = len(successful) / wall_time

    return agg


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator for pass@k.

    Args:
        n: total number of samples generated per problem
        c: number of correct samples
        k: k in pass@k

    Returns:
        Estimated pass@k probability.
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def confidence_interval_95(scores: list[float]) -> tuple[float, float]:
    """Compute 95% confidence interval using normal approximation."""
    arr = np.array(scores)
    n = len(arr)
    if n < 2:
        mean = float(np.mean(arr))
        return (mean, mean)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    return (mean - 1.96 * se, mean + 1.96 * se)
