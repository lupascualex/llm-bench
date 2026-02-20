"""Click CLI entry point."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import BenchmarkConfig, RunConfig, VLLMConfig

console = Console()


@click.group()
@click.option("--base-url", default="http://localhost:8000/v1", help="vLLM API base URL.")
@click.option("--model", default=None, help="Model name served by vLLM.")
@click.option("--api-key", default="EMPTY", help="API key (default: EMPTY).")
@click.option("--output-dir", default="results", help="Directory for result files.")
@click.option("--data-dir", default="data", help="Directory for cached datasets.")
@click.pass_context
def cli(ctx: click.Context, base_url: str, model: str | None, api_key: str, output_dir: str, data_dir: str) -> None:
    """LLM Benchmarker — benchmark LLMs served via vLLM."""
    ctx.ensure_object(dict)
    ctx.obj["base_url"] = base_url
    ctx.obj["model"] = model
    ctx.obj["api_key"] = api_key
    ctx.obj["output_dir"] = output_dir
    ctx.obj["data_dir"] = data_dir


@cli.command()
@click.pass_context
def check(ctx: click.Context) -> None:
    """Check connectivity to the vLLM server."""
    model = ctx.obj.get("model")
    if not model:
        raise click.UsageError("--model is required for the check command.")

    async def _check() -> None:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=ctx.obj["base_url"],
            api_key=ctx.obj["api_key"],
            timeout=10.0,
        )
        try:
            models = await client.models.list()
            model_ids = [m.id for m in models.data]
            console.print(f"[green]Connected to {ctx.obj['base_url']}[/]")
            console.print(f"Available models: {', '.join(model_ids)}")
            if model in model_ids:
                console.print(f"[green]Model {model!r} is available.[/]")
            else:
                console.print(f"[yellow]Warning: Model {model!r} not found in available models.[/]")
        except Exception as e:
            console.print(f"[red]Connection failed:[/] {e}")
            raise SystemExit(1)

    asyncio.run(_check())


@cli.command("list")
def list_benchmarks_cmd() -> None:
    """List all available benchmarks."""
    from .benchmarks import list_benchmarks

    benchmarks = list_benchmarks()
    table = Table(title="Available Benchmarks")
    table.add_column("Name", style="bold cyan")
    table.add_column("Tier", justify="center")
    table.add_column("Description")

    for b in benchmarks:
        table.add_row(str(b["name"]), str(b["tier"]), str(b["description"]))

    console.print(table)


@cli.command()
@click.argument("benchmarks", nargs=-1, required=True)
@click.option("--concurrency", default=8, help="Max concurrent requests.")
@click.option("--max-samples", default=None, type=int, help="Limit samples per benchmark.")
@click.option("--temperature", default=0.0, type=float, help="Sampling temperature.")
@click.option("--max-tokens", default=2048, type=int, help="Max tokens to generate.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--no-stream", is_flag=True, help="Disable streaming (loses TTFT).")
@click.option("--scrape-prometheus", is_flag=True, help="Also scrape /metrics endpoint.")
@click.pass_context
def run(
    ctx: click.Context,
    benchmarks: tuple[str, ...],
    concurrency: int,
    max_samples: int | None,
    temperature: float,
    max_tokens: int,
    seed: int,
    no_stream: bool,
    scrape_prometheus: bool,
) -> None:
    """Run one or more benchmarks."""
    model = ctx.obj.get("model")
    if not model:
        raise click.UsageError("--model is required for the run command.")

    vllm_config = VLLMConfig(
        base_url=ctx.obj["base_url"],
        api_key=ctx.obj["api_key"],
        model=model,
    )
    run_config = RunConfig(
        concurrency=concurrency,
        max_samples=max_samples,
        streaming=not no_stream,
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
        output_dir=Path(ctx.obj["output_dir"]),
        data_dir=Path(ctx.obj["data_dir"]),
        scrape_prometheus=scrape_prometheus,
    )
    config = BenchmarkConfig(vllm=vllm_config, run=run_config)

    async def _run() -> None:
        from .benchmarks import get_benchmark
        from .client import InstrumentedClient
        from .results import save_result
        from .runner import BenchmarkRunner

        client = InstrumentedClient(vllm_config)
        runner = BenchmarkRunner(client, config)

        for bench_name in benchmarks:
            try:
                benchmark = get_benchmark(bench_name)
            except KeyError as e:
                console.print(f"[red]{e}[/]")
                continue

            result = await runner.run_benchmark(benchmark)
            save_result(result, config)

    asyncio.run(_run())


@cli.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
def compare(files: tuple[str, ...]) -> None:
    """Compare results from multiple benchmark runs."""
    from .results import compare_results

    compare_results([Path(f) for f in files])
