"""TAU-Bench — Multi-turn agentic benchmark with simulated tool execution."""

from __future__ import annotations

import json
from typing import Any

from . import register
from .base import BaseBenchmark, BenchmarkSample, SampleResult
from ..client import InstrumentedClient, RequestMetrics
from ..config import BenchmarkConfig


MAX_TURNS = 20


@register
class TAUBenchBenchmark(BaseBenchmark):
    name = "tau_bench"
    description = "TAU-Bench — Multi-turn agentic tasks with simulated tool execution"
    tier = 3
    is_multi_turn = True
    requires_tools = True

    async def load_dataset(self, data_dir: str, max_samples: int | None = None) -> list[BenchmarkSample]:
        from datasets import load_dataset

        try:
            ds = load_dataset("sierra-research/tau-bench", split="test", trust_remote_code=True)
        except Exception:
            # TAU-bench may need a different loading strategy
            ds = load_dataset("sierra-research/tau-bench", split="train", trust_remote_code=True)

        samples = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            user_instruction = row.get("user_instruction", row.get("instruction", ""))
            tools = row.get("tools", [])
            expected_actions = row.get("expected_actions", [])
            env_state = row.get("initial_state", {})

            if isinstance(tools, str):
                try:
                    tools = json.loads(tools)
                except (json.JSONDecodeError, TypeError):
                    tools = []

            if isinstance(expected_actions, str):
                try:
                    expected_actions = json.loads(expected_actions)
                except (json.JSONDecodeError, TypeError):
                    expected_actions = []

            samples.append(BenchmarkSample(
                id=f"tau_bench_{i}",
                messages=[{"role": "user", "content": user_instruction}],
                metadata={
                    "tools": tools,
                    "expected_actions": expected_actions,
                    "initial_state": env_state,
                },
            ))
        return samples

    async def evaluate(self, sample: BenchmarkSample, response: str) -> SampleResult:
        raise NotImplementedError("Use multi_turn for TAU-Bench")

    async def multi_turn(
        self,
        sample: BenchmarkSample,
        client: InstrumentedClient,
        config: BenchmarkConfig,
    ) -> SampleResult:
        tools_defs = sample.metadata.get("tools", [])
        expected_actions = sample.metadata.get("expected_actions", [])
        env_state = dict(sample.metadata.get("initial_state", {}))

        # Convert tools to OpenAI format
        openai_tools = []
        for tool in tools_defs:
            if isinstance(tool, dict):
                if "type" not in tool:
                    openai_tools.append({"type": "function", "function": tool})
                else:
                    openai_tools.append(tool)

        conversation = list(sample.messages)
        all_metrics: list[RequestMetrics] = []
        executed_actions: list[dict[str, Any]] = []

        for turn in range(MAX_TURNS):
            kwargs: dict[str, Any] = {
                "max_tokens": 1024,
                "temperature": config.run.temperature,
            }
            if config.run.seed is not None:
                kwargs["seed"] = config.run.seed
            if openai_tools:
                kwargs["tools"] = openai_tools

            response, metrics = await client.chat_completion_raw(
                conversation,
                **kwargs,
            )
            all_metrics.append(metrics)

            if not response.choices:
                break

            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Add assistant message to conversation
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                assistant_msg["content"] = msg.content
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            conversation.append(assistant_msg)

            # Process tool calls
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                    executed_actions.append({
                        "name": tc.function.name,
                        "arguments": args,
                    })

                    # Simulate tool execution with a basic response
                    tool_result = self._simulate_tool(tc.function.name, args, env_state)
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result),
                    })
            elif finish_reason == "stop":
                break

        # Evaluate by comparing executed actions to expected
        correct = self._evaluate_actions(executed_actions, expected_actions)
        primary_metrics = all_metrics[-1] if all_metrics else None

        return SampleResult(
            id=sample.id,
            correct=correct,
            score=1.0 if correct else 0.0,
            predicted=str(executed_actions)[:300],
            expected=str(expected_actions)[:300],
            details={
                "num_turns": len(all_metrics),
                "num_actions": len(executed_actions),
            },
            metrics=primary_metrics,
        )

    @staticmethod
    def _simulate_tool(name: str, args: dict, state: dict) -> dict:
        """Simulate a tool call. Returns a basic success response."""
        return {"status": "success", "result": f"Executed {name}"}

    @staticmethod
    def _evaluate_actions(
        executed: list[dict[str, Any]],
        expected: list[dict[str, Any]],
    ) -> bool:
        """Check if executed actions match expected actions."""
        if not expected:
            return True
        if len(executed) != len(expected):
            return False
        for ex, exp in zip(executed, expected):
            if ex.get("name") != exp.get("name"):
                return False
        return True
