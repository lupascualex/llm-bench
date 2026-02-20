"""Benchmark registry with decorator-based registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseBenchmark

_REGISTRY: dict[str, type[BaseBenchmark]] = {}


def register(cls: type[BaseBenchmark]) -> type[BaseBenchmark]:
    """Class decorator that registers a benchmark by its ``name`` attribute."""
    _REGISTRY[cls.name] = cls
    return cls


def get_benchmark(name: str) -> BaseBenchmark:
    """Instantiate and return a registered benchmark by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown benchmark {name!r}. Available: {available}")
    return _REGISTRY[name]()


def list_benchmarks() -> list[dict[str, str | int]]:
    """Return metadata for all registered benchmarks."""
    results = []
    for name in sorted(_REGISTRY):
        cls = _REGISTRY[name]
        results.append({
            "name": cls.name,
            "description": cls.description,
            "tier": cls.tier,
        })
    return results


# Import benchmark modules to trigger registration.
from . import (  # noqa: E402, F401
    mmlu,
    mmlu_pro,
    humaneval,
    ifeval,
    hle,
    longbench_v2,
    graphwalks,
    multi_if,
    mrcr,
    bfcl,
    tau_bench,
    docker_base,
    terminal_bench,
    swe_bench,
    swe_bench_pro,
)
