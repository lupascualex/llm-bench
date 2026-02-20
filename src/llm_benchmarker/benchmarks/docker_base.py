"""Docker-sandboxed benchmark base class for Tier 4 benchmarks."""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseBenchmark, BenchmarkSample, SampleResult


class DockerBenchmark(BaseBenchmark):
    """Base class for benchmarks that run inside Docker containers."""

    tier = 4
    requires_docker = True
    is_multi_turn = True

    # Override in subclasses
    docker_image: str = "python:3.11-slim"
    container_timeout: float = 300.0

    async def _create_container(self, sample: BenchmarkSample) -> str:
        """Create and start a Docker container. Returns container ID."""
        try:
            import docker
        except ImportError:
            raise RuntimeError(
                "Docker benchmarks require the 'docker' package. "
                "Install with: pip install llm-benchmarker[docker]"
            )

        client = docker.from_env()
        container = client.containers.run(
            self.docker_image,
            command="sleep infinity",
            detach=True,
            network_mode="none",
            mem_limit="2g",
            cpu_period=100000,
            cpu_quota=100000,
        )
        return container.id

    async def _exec_in_container(
        self, container_id: str, command: str, timeout: float | None = None
    ) -> tuple[int, str]:
        """Execute a command in a running container.

        Returns (exit_code, output).
        """
        try:
            import docker
        except ImportError:
            raise RuntimeError("Docker package required.")

        client = docker.from_env()
        container = client.containers.get(container_id)

        timeout = timeout or self.container_timeout
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: container.exec_run(["bash", "-c", command]),
                ),
                timeout=timeout,
            )
            return result.exit_code, result.output.decode("utf-8", errors="replace")
        except asyncio.TimeoutError:
            return -1, f"Command timed out after {timeout}s"

    async def _remove_container(self, container_id: str) -> None:
        """Stop and remove a container."""
        try:
            import docker
            client = docker.from_env()
            container = client.containers.get(container_id)
            container.remove(force=True)
        except Exception:
            pass

    async def evaluate(self, sample: BenchmarkSample, response: str) -> SampleResult:
        raise NotImplementedError("Docker benchmarks use multi_turn")
