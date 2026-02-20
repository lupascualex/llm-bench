"""Pydantic configuration models."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class VLLMConfig(BaseModel):
    """Connection settings for a vLLM-compatible OpenAI API."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str
    timeout: float = 300.0


class RunConfig(BaseModel):
    """Execution settings for a benchmark run."""

    concurrency: int = Field(default=8, ge=1)
    max_samples: int | None = None
    streaming: bool = True
    seed: int | None = 42
    temperature: float = 0.0
    max_tokens: int = 2048
    output_dir: Path = Path("results")
    data_dir: Path = Path("data")
    scrape_prometheus: bool = False
    prometheus_url: str = "http://localhost:8000/metrics"


class BenchmarkConfig(BaseModel):
    """Combined config passed to benchmark runners."""

    vllm: VLLMConfig
    run: RunConfig
