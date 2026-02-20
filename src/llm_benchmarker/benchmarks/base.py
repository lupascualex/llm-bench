"""Base benchmark ABC and shared data classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..client import InstrumentedClient, RequestMetrics
from ..config import BenchmarkConfig
from ..metrics import AggregatedMetrics


@dataclass
class BenchmarkSample:
    """A single evaluation sample."""

    id: str
    messages: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    """Result of evaluating one sample."""

    id: str
    correct: bool
    score: float = 0.0
    predicted: str = ""
    expected: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    metrics: RequestMetrics | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "correct": self.correct,
            "score": self.score,
            "predicted": self.predicted,
            "expected": self.expected,
        }
        if self.details:
            d["details"] = self.details
        if self.metrics:
            d["metrics"] = {
                "ttft_seconds": self.metrics.ttft_seconds,
                "total_latency_seconds": self.metrics.total_latency_seconds,
                "prompt_tokens": self.metrics.prompt_tokens,
                "completion_tokens": self.metrics.completion_tokens,
                "tokens_per_second_generation": self.metrics.tokens_per_second_generation,
            }
        return d


@dataclass
class BenchmarkResult:
    """Aggregated result of a full benchmark run."""

    name: str
    num_samples: int
    accuracy: float
    score: float | None = None
    sample_results: list[SampleResult] = field(default_factory=list)
    aggregated_metrics: AggregatedMetrics | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "name": self.name,
            "num_samples": self.num_samples,
            "accuracy": self.accuracy,
        }
        if self.score is not None:
            d["score"] = self.score
        if self.aggregated_metrics:
            d["aggregated_metrics"] = self.aggregated_metrics.to_dict()
        if self.extra:
            d["extra"] = self.extra
        d["sample_results"] = [sr.to_dict() for sr in self.sample_results]
        return d


class BaseBenchmark(ABC):
    """Abstract base class for all benchmarks."""

    name: str = ""
    description: str = ""
    tier: int = 1
    is_multi_turn: bool = False
    requires_tools: bool = False
    requires_docker: bool = False

    @abstractmethod
    async def load_dataset(
        self, data_dir: str, max_samples: int | None = None
    ) -> list[BenchmarkSample]:
        """Load and return evaluation samples."""

    @abstractmethod
    async def evaluate(self, sample: BenchmarkSample, response: str) -> SampleResult:
        """Evaluate a model response against the expected answer."""

    def get_request_kwargs(self, sample: BenchmarkSample) -> dict[str, Any]:
        """Return extra kwargs for the API request (e.g. max_tokens, tools)."""
        return {}

    def system_prompt(self) -> str | None:
        """Return optional system prompt prepended to every request."""
        return None

    async def multi_turn(
        self,
        sample: BenchmarkSample,
        client: InstrumentedClient,
        config: BenchmarkConfig,
    ) -> SampleResult:
        """Execute a multi-turn conversation. Override for multi-turn benchmarks."""
        raise NotImplementedError("multi_turn not implemented for single-turn benchmarks")
