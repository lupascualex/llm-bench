"""Tests for the instrumented client (mocked)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llm_benchmarker.client import InstrumentedClient, RequestMetrics
from llm_benchmarker.config import VLLMConfig


@pytest.fixture
def config():
    return VLLMConfig(model="test-model", base_url="http://test:8000/v1")


class TestRequestMetrics:
    def test_defaults(self):
        m = RequestMetrics()
        assert m.ttft_seconds is None
        assert m.error is None
        assert m.total_latency_seconds == 0.0


class TestInstrumentedClient:
    def test_init(self, config):
        client = InstrumentedClient(config)
        assert client._config.model == "test-model"

    @pytest.mark.asyncio
    async def test_non_stream_chat(self, config):
        client = InstrumentedClient(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        with patch.object(client._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            text, metrics = await client.chat_completion(
                [{"role": "user", "content": "hi"}],
                stream=False,
            )

        assert text == "Hello!"
        assert metrics.prompt_tokens == 10
        assert metrics.completion_tokens == 5
        assert metrics.total_latency_seconds > 0
        assert metrics.error is None

    @pytest.mark.asyncio
    async def test_error_handling(self, config):
        client = InstrumentedClient(config)

        with patch.object(client._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("connection refused")
            text, metrics = await client.chat_completion(
                [{"role": "user", "content": "hi"}],
                stream=False,
            )

        assert text == ""
        assert metrics.error == "connection refused"

    def test_compute_derived(self):
        m = RequestMetrics(
            ttft_seconds=0.1,
            total_latency_seconds=1.0,
            prompt_tokens=100,
            completion_tokens=50,
        )
        InstrumentedClient._compute_derived(m)
        assert m.prefill_time_seconds == 0.1
        assert m.decode_time_seconds == pytest.approx(0.9)
        assert m.tokens_per_second_generation == pytest.approx(49 / 0.9)
        assert m.tokens_per_second_prompt == pytest.approx(100 / 0.1)
