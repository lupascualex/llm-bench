"""Instrumented vLLM client with streaming TTFT measurement."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from .config import VLLMConfig


@dataclass
class RequestMetrics:
    """Per-request performance metrics."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    ttft_seconds: float | None = None
    total_latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prefill_time_seconds: float | None = None
    decode_time_seconds: float | None = None
    tokens_per_second_generation: float | None = None
    tokens_per_second_prompt: float | None = None
    error: str | None = None


class InstrumentedClient:
    """Async OpenAI client wrapper that measures TTFT and throughput."""

    def __init__(self, config: VLLMConfig) -> None:
        self._config = config
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        )

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = True,
        **kwargs: Any,
    ) -> tuple[str, RequestMetrics]:
        """Send a chat completion and return (response_text, metrics)."""
        metrics = RequestMetrics()
        kwargs.setdefault("model", self._config.model)

        if stream:
            return await self._stream_chat(messages, metrics, **kwargs)
        return await self._non_stream_chat(messages, metrics, **kwargs)

    async def chat_completion_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, RequestMetrics]:
        """Send a chat completion and return the raw response object + metrics.

        Used by benchmarks that need tool_calls or other structured output.
        """
        metrics = RequestMetrics()
        kwargs.setdefault("model", self._config.model)

        start = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                messages=messages,
                stream=False,
                **kwargs,
            )
        except Exception as exc:
            metrics.error = str(exc)
            metrics.total_latency_seconds = time.perf_counter() - start
            raise
        end = time.perf_counter()

        metrics.total_latency_seconds = end - start
        usage = response.usage
        if usage:
            metrics.prompt_tokens = usage.prompt_tokens or 0
            metrics.completion_tokens = usage.completion_tokens or 0

        return response, metrics

    async def completion(
        self,
        prompt: str,
        *,
        stream: bool = True,
        **kwargs: Any,
    ) -> tuple[str, RequestMetrics]:
        """Send a legacy completion and return (response_text, metrics)."""
        metrics = RequestMetrics()
        kwargs.setdefault("model", self._config.model)

        if stream:
            return await self._stream_completion(prompt, metrics, **kwargs)
        return await self._non_stream_completion(prompt, metrics, **kwargs)

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    async def _stream_chat(
        self,
        messages: list[dict[str, Any]],
        metrics: RequestMetrics,
        **kwargs: Any,
    ) -> tuple[str, RequestMetrics]:
        start = time.perf_counter()
        chunks: list[str] = []
        first_token_seen = False

        try:
            stream = await self._client.chat.completions.create(
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )
            async for chunk in stream:
                if chunk.usage:
                    metrics.prompt_tokens = chunk.usage.prompt_tokens or 0
                    metrics.completion_tokens = chunk.usage.completion_tokens or 0

                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        if not first_token_seen:
                            metrics.ttft_seconds = time.perf_counter() - start
                            first_token_seen = True
                        chunks.append(delta.content)
        except Exception as exc:
            metrics.error = str(exc)
            metrics.total_latency_seconds = time.perf_counter() - start
            return "", metrics

        end = time.perf_counter()
        metrics.total_latency_seconds = end - start
        self._compute_derived(metrics)
        return "".join(chunks), metrics

    async def _stream_completion(
        self,
        prompt: str,
        metrics: RequestMetrics,
        **kwargs: Any,
    ) -> tuple[str, RequestMetrics]:
        start = time.perf_counter()
        chunks: list[str] = []
        first_token_seen = False

        try:
            stream = await self._client.completions.create(
                prompt=prompt,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )
            async for chunk in stream:
                if chunk.usage:
                    metrics.prompt_tokens = chunk.usage.prompt_tokens or 0
                    metrics.completion_tokens = chunk.usage.completion_tokens or 0

                if chunk.choices:
                    text = chunk.choices[0].text
                    if text:
                        if not first_token_seen:
                            metrics.ttft_seconds = time.perf_counter() - start
                            first_token_seen = True
                        chunks.append(text)
        except Exception as exc:
            metrics.error = str(exc)
            metrics.total_latency_seconds = time.perf_counter() - start
            return "", metrics

        end = time.perf_counter()
        metrics.total_latency_seconds = end - start
        self._compute_derived(metrics)
        return "".join(chunks), metrics

    # ------------------------------------------------------------------
    # Non-streaming helpers
    # ------------------------------------------------------------------

    async def _non_stream_chat(
        self,
        messages: list[dict[str, Any]],
        metrics: RequestMetrics,
        **kwargs: Any,
    ) -> tuple[str, RequestMetrics]:
        start = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                messages=messages,
                stream=False,
                **kwargs,
            )
        except Exception as exc:
            metrics.error = str(exc)
            metrics.total_latency_seconds = time.perf_counter() - start
            return "", metrics

        end = time.perf_counter()
        metrics.total_latency_seconds = end - start

        usage = response.usage
        if usage:
            metrics.prompt_tokens = usage.prompt_tokens or 0
            metrics.completion_tokens = usage.completion_tokens or 0

        text = response.choices[0].message.content or "" if response.choices else ""
        return text, metrics

    async def _non_stream_completion(
        self,
        prompt: str,
        metrics: RequestMetrics,
        **kwargs: Any,
    ) -> tuple[str, RequestMetrics]:
        start = time.perf_counter()
        try:
            response = await self._client.completions.create(
                prompt=prompt,
                stream=False,
                **kwargs,
            )
        except Exception as exc:
            metrics.error = str(exc)
            metrics.total_latency_seconds = time.perf_counter() - start
            return "", metrics

        end = time.perf_counter()
        metrics.total_latency_seconds = end - start

        usage = response.usage
        if usage:
            metrics.prompt_tokens = usage.prompt_tokens or 0
            metrics.completion_tokens = usage.completion_tokens or 0

        text = response.choices[0].text or "" if response.choices else ""
        return text, metrics

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_derived(metrics: RequestMetrics) -> None:
        """Fill in derived throughput fields."""
        metrics.prefill_time_seconds = metrics.ttft_seconds

        if metrics.ttft_seconds and metrics.total_latency_seconds > metrics.ttft_seconds:
            metrics.decode_time_seconds = metrics.total_latency_seconds - metrics.ttft_seconds
        if metrics.decode_time_seconds and metrics.decode_time_seconds > 0 and metrics.completion_tokens > 1:
            metrics.tokens_per_second_generation = (metrics.completion_tokens - 1) / metrics.decode_time_seconds
        if metrics.ttft_seconds and metrics.ttft_seconds > 0 and metrics.prompt_tokens > 0:
            metrics.tokens_per_second_prompt = metrics.prompt_tokens / metrics.ttft_seconds
