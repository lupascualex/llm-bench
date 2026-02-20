"""JSON result persistence and comparison."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .benchmarks.base import BenchmarkResult
from .config import BenchmarkConfig
from .metrics import confidence_interval_95

console = Console()


def save_result(result: BenchmarkResult, config: BenchmarkConfig) -> Path:
    """Save a benchmark result to a timestamped JSON file."""
    output_dir = Path(config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{result.name}_{config.vllm.model}_{timestamp}.json"
    # Sanitize model name for filesystem
    filename = filename.replace("/", "_").replace("\\", "_")
    filepath = output_dir / filename

    data = {
        "benchmark": result.name,
        "model": config.vllm.model,
        "timestamp": timestamp,
        "config": {
            "base_url": config.vllm.base_url,
            "concurrency": config.run.concurrency,
            "streaming": config.run.streaming,
            "temperature": config.run.temperature,
            "max_tokens": config.run.max_tokens,
            "seed": config.run.seed,
        },
        **result.to_dict(),
    }

    filepath.write_text(json.dumps(data, indent=2, default=str))
    console.print(f"  Results saved to [cyan]{filepath}[/]")
    return filepath


def load_result(filepath: Path) -> dict:
    """Load a result JSON file."""
    return json.loads(filepath.read_text())


def compare_results(filepaths: list[Path]) -> None:
    """Display a side-by-side comparison table of multiple result files."""
    results = []
    for fp in filepaths:
        try:
            results.append(load_result(fp))
        except Exception as e:
            console.print(f"[red]Error loading {fp}:[/] {e}")

    if not results:
        console.print("[red]No valid result files to compare.[/]")
        return

    table = Table(title="Benchmark Comparison", show_lines=True)
    table.add_column("Metric", style="bold")

    for r in results:
        label = f"{r.get('benchmark', '?')}\n{r.get('model', '?')}"
        table.add_column(label, justify="right")

    # Accuracy
    table.add_row("Accuracy", *[f"{r.get('accuracy', 0):.4f}" for r in results])
    table.add_row("Samples", *[str(r.get("num_samples", 0)) for r in results])

    if any("score" in r for r in results):
        table.add_row("Score", *[f"{r.get('score', 0):.4f}" for r in results])

    # Confidence intervals
    for r in results:
        sample_results = r.get("sample_results", [])
        if sample_results:
            scores = [1.0 if sr.get("correct") else 0.0 for sr in sample_results]
            ci_lo, ci_hi = confidence_interval_95(scores)
            r["_ci"] = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
        else:
            r["_ci"] = "N/A"
    table.add_row("95% CI", *[r["_ci"] for r in results])

    # Performance metrics
    metric_keys = [
        ("ttft_mean", "TTFT Mean (s)"),
        ("ttft_p50", "TTFT P50 (s)"),
        ("ttft_p95", "TTFT P95 (s)"),
        ("generation_tps_mean", "Gen TPS Mean"),
        ("latency_mean", "Latency Mean (s)"),
        ("latency_p95", "Latency P95 (s)"),
        ("effective_throughput_rps", "Throughput (req/s)"),
    ]

    for key, label in metric_keys:
        values = []
        for r in results:
            agg = r.get("aggregated_metrics", {})
            v = agg.get(key)
            values.append(f"{v:.4f}" if v is not None else "N/A")
        table.add_row(label, *values)

    console.print(table)
