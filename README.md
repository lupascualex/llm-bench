# llm-benchmarker

A CLI tool for benchmarking LLMs served via vLLM's OpenAI-compatible API, with per-request latency instrumentation, streaming TTFT measurement, and a catalog of 14 standard benchmarks organized across four capability tiers.

---

## Table of Contents

1. [Overview](#overview)
2. [Feature Highlights](#feature-highlights)
3. [Architecture Overview](#architecture-overview)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [CLI Reference](#cli-reference)
   - [Global Options](#global-options)
   - [check](#check)
   - [list](#list)
   - [run](#run)
   - [compare](#compare)
8. [Tier System](#tier-system)
9. [Benchmark Catalog](#benchmark-catalog)
   - [Tier 1 — Core Capability](#tier-1--core-capability)
   - [Tier 2 — Long-Context and Multi-Turn](#tier-2--long-context-and-multi-turn)
   - [Tier 3 — Agentic and Tool Use](#tier-3--agentic-and-tool-use)
   - [Tier 4 — Docker-Sandboxed Execution](#tier-4--docker-sandboxed-execution)
10. [Evaluators](#evaluators)
11. [Performance Metrics](#performance-metrics)
12. [How Streaming TTFT Measurement Works](#how-streaming-ttft-measurement-works)
13. [Async Execution Engine](#async-execution-engine)
14. [Results Format](#results-format)
15. [Comparing Results](#comparing-results)
16. [Configuration Reference](#configuration-reference)
17. [Examples](#examples)
18. [Development](#development)

---

## Overview

`llm-benchmarker` is a Python CLI tool that connects to any OpenAI-compatible API endpoint — primarily vLLM — and runs a suite of standardized language model benchmarks. It collects both quality metrics (accuracy, pass rates, scores) and performance metrics (time to first token, generation throughput, end-to-end latency) in a single unified run.

Results are written as timestamped JSON files. Multiple result files can be compared side-by-side in a rich terminal table. The tool is designed to be deterministic (fixed seeds), async-first (controlled concurrency via a semaphore), and extensible (all benchmarks register themselves through a decorator-based registry).

---

## Feature Highlights

- **14 built-in benchmarks** spanning knowledge, reasoning, code generation, instruction following, long-context understanding, function calling, and agentic task completion
- **Streaming TTFT measurement** using the OpenAI streaming API with `include_usage` — measures wall-clock time from request dispatch to first content token
- **Derived throughput metrics** — generation tokens per second (decode phase), prompt tokens per second (prefill phase), and effective request throughput over wall time
- **Async concurrency engine** — `asyncio.Semaphore`-bounded task pool with a rich progress bar
- **Four benchmark tiers** from fast single-turn MCQ (Tier 1) to Docker-sandboxed software engineering tasks (Tier 4)
- **Pluggable evaluators** — MCQ extraction, IFEval rule checking, code execution, AST-based function call comparison, set comparison, and exact/contains matching
- **Side-by-side comparison** of multiple result files including 95% confidence intervals
- **Prometheus scraping** support for vLLM's `/metrics` endpoint
- **`--no-stream` mode** to disable streaming when TTFT is not needed
- **`--max-samples`** to run a quick sanity check on a subset of each benchmark

---

## Architecture Overview

```
llm-benchmarker/
├── pyproject.toml                   # Build system, dependencies, entry point
└── src/
    └── llm_benchmarker/
        ├── cli.py                   # Click command definitions (check, list, run, compare)
        ├── config.py                # Pydantic config models (VLLMConfig, RunConfig, BenchmarkConfig)
        ├── client.py                # InstrumentedClient — async OpenAI wrapper with TTFT timing
        ├── metrics.py               # AggregatedMetrics, aggregate_metrics(), pass@k, 95% CI
        ├── runner.py                # BenchmarkRunner — semaphore-based async execution engine
        ├── results.py               # JSON serialization, save_result(), compare_results()
        ├── benchmarks/
        │   ├── __init__.py          # Benchmark registry (register decorator, get_benchmark, list_benchmarks)
        │   ├── base.py              # BaseBenchmark ABC, BenchmarkSample, SampleResult, BenchmarkResult
        │   ├── mmlu.py              # Tier 1 — MMLU (15,908 MCQ, 57 subjects)
        │   ├── mmlu_pro.py          # Tier 1 — MMLU-Pro (12,000+ MCQ, 10 options)
        │   ├── humaneval.py         # Tier 1 — HumanEval (164 code tasks, pass@1)
        │   ├── ifeval.py            # Tier 1 — IFEval (~500 instruction-following tasks)
        │   ├── hle.py               # Tier 1 — HLE (2,500 expert-level MCQ + short answer)
        │   ├── longbench_v2.py      # Tier 1 — LongBench v2 (503 long-context MCQ)
        │   ├── graphwalks.py        # Tier 1 — GraphWalks (1,150 graph reasoning tasks)
        │   ├── multi_if.py          # Tier 2 — Multi-IF (4,501 multi-turn multilingual IF)
        │   ├── mrcr.py              # Tier 2 — MRCR (2,400 needle-retrieval, 4k–1M tokens)
        │   ├── bfcl.py              # Tier 3 — BFCL (Berkeley Function Calling Leaderboard)
        │   ├── tau_bench.py         # Tier 3 — TAU-Bench (multi-turn agentic with tool sim)
        │   ├── docker_base.py       # Tier 4 — DockerBenchmark base class
        │   ├── terminal_bench.py    # Tier 4 — Terminal-Bench (89 CLI tasks in Docker)
        │   ├── swe_bench.py         # Tier 4 — SWE-bench Verified (500 SE tasks in Docker)
        │   └── swe_bench_pro.py     # Tier 4 — SWE-bench Pro (1,865 SE tasks in Docker)
        └── evaluators/
            ├── exact_match.py       # Normalized exact match and substring match
            ├── mcq.py               # MCQ letter extraction (4-option and 10-option)
            ├── code_execution.py    # Async subprocess Python execution for HumanEval
            ├── ifeval_rules.py      # 25 IFEval constraint verifiers
            ├── ast_eval.py          # AST/JSON function call parsing and comparison
            └── set_comparison.py    # Set extraction, exact match, Jaccard similarity
```

### Key Design Patterns

**Registry pattern.** Each benchmark class applies the `@register` decorator, which inserts it into a module-level dictionary keyed by its `name` attribute. The `get_benchmark(name)` function instantiates on demand, and `list_benchmarks()` returns metadata without instantiation.

**Abstract base class.** `BaseBenchmark` defines the contract: `load_dataset()`, `evaluate()`, optional `get_request_kwargs()`, optional `system_prompt()`, and optional `multi_turn()`. The runner inspects `benchmark.is_multi_turn` to decide which path to take.

**Dataclass result chain.** `BenchmarkSample` carries input data, `SampleResult` carries per-sample output and raw `RequestMetrics`, and `BenchmarkResult` aggregates all samples and computes `AggregatedMetrics`.

---

## Prerequisites

- **Python 3.11 or later** (enforced in `pyproject.toml`)
- A running **vLLM server** (or any OpenAI-compatible API) accessible on the network. The default URL is `http://localhost:8000/v1`.
- For **Tier 4 Docker benchmarks**: Docker Engine installed and running on the host, plus the `docker` Python extra (`pip install llm-benchmarker[docker]`).
- For **HumanEval**: `python3` must be available in `PATH` (used as a subprocess for code execution).
- Internet access (or pre-cached data) to download datasets via Hugging Face `datasets`.

---

## Installation

### Basic installation

```bash
pip install llm-benchmarker
```

### With Docker support (required for Tier 4 benchmarks)

```bash
pip install "llm-benchmarker[docker]"
```

### Development installation

```bash
git clone <repo-url>
cd llm-benchmarker
pip install -e ".[dev]"
```

The `dev` extra installs `pytest>=8.0`, `pytest-asyncio>=0.23`, and `ruff>=0.4`.

### Runtime dependencies

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| click | 8.1 | CLI framework |
| openai | 1.30 | API client (chat completions, streaming) |
| httpx | 0.27 | HTTP transport for openai |
| datasets | 2.19 | Hugging Face dataset loading |
| pydantic | 2.7 | Configuration models |
| rich | 13.7 | Progress bars, formatted tables |
| numpy | 1.26 | Percentile/mean calculations |

---

## Quick Start

```bash
# 1. Start vLLM (example)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

# 2. Check connectivity
llm-bench --model meta-llama/Llama-3.1-8B-Instruct check

# 3. List available benchmarks
llm-bench list

# 4. Run a quick MMLU sanity check (50 samples)
llm-bench --model meta-llama/Llama-3.1-8B-Instruct run mmlu --max-samples 50

# 5. Run the full Tier 1 suite
llm-bench --model meta-llama/Llama-3.1-8B-Instruct run \
    mmlu mmlu_pro humaneval ifeval hle longbench_v2 graphwalks

# 6. Compare two runs
llm-bench compare \
    results/mmlu_meta-llama_Llama-3.1-8B-Instruct_20260220T120000Z.json \
    results/mmlu_meta-llama_Llama-3.1-70B-Instruct_20260220T130000Z.json
```

---

## CLI Reference

The tool is invoked as `llm-bench`. All subcommands share global connection options.

### Global Options

These options appear before the subcommand and are available to all subcommands.

| Option | Default | Description |
|--------|---------|-------------|
| `--base-url TEXT` | `http://localhost:8000/v1` | Base URL of the vLLM (or any OpenAI-compatible) API. Include the `/v1` path prefix. |
| `--model TEXT` | *(required for most commands)* | The model identifier as it is registered in the vLLM server. Must match exactly what `/v1/models` returns. |
| `--api-key TEXT` | `EMPTY` | API key. vLLM does not require authentication by default, so `EMPTY` is the conventional placeholder. Set this to a real key when pointing at a guarded endpoint. |
| `--output-dir TEXT` | `results` | Directory where result JSON files are written. Created automatically if it does not exist. |
| `--data-dir TEXT` | `data` | Directory where Hugging Face datasets are cached. Passed to `load_dataset` as the cache location hint. |

### check

```
llm-bench --model <model> check
```

Verifies connectivity to the vLLM server by listing models and confirming the requested model is available. Prints a green success message or a yellow warning if the model name is not found in the list. Exits with code 1 on connection failure.

**Requires:** `--model`

**Example:**

```bash
llm-bench --base-url http://gpu-server:8000/v1 --model Qwen/Qwen2.5-72B-Instruct check
```

### list

```
llm-bench list
```

Displays a rich table of all registered benchmarks showing their `name`, `tier`, and `description`. No server connection required.

**Example output:**

```
           Available Benchmarks
┌──────────────┬──────┬────────────────────────────────────────────────────────────────────┐
│ Name         │ Tier │ Description                                                        │
├──────────────┼──────┼────────────────────────────────────────────────────────────────────┤
│ bfcl         │  3   │ BFCL — Berkeley Function Calling Leaderboard with AST-based eval   │
│ graphwalks   │  1   │ GraphWalks — 1,150 graph reasoning tasks with set comparison       │
│ hle          │  1   │ Humanity's Last Exam — 2,500 expert-level MCQ and short-answer     │
│ humaneval    │  1   │ HumanEval — 164 Python code generation tasks                       │
│ ifeval       │  1   │ IFEval — ~500 instruction-following tasks with verifiable rules     │
│ longbench_v2 │  1   │ LongBench v2 — 503 long-context MCQ spanning 8k to 2M words       │
│ mmlu         │  1   │ MMLU — 15,908 MCQ, 57 subjects                                     │
│ mmlu_pro     │  1   │ MMLU-Pro — 12,000+ MCQ with 10 options per question                │
│ mrcr         │  2   │ MRCR — 2,400 multi-round long-context needle retrieval             │
│ multi_if     │  2   │ Multi-IF — 4,501 multi-turn multilingual instruction following      │
│ swe_bench    │  4   │ SWE-bench Verified — 500 real-world SE tasks in Docker             │
│ swe_bench_pro│  4   │ SWE-bench Pro — 1,865 extended real-world SE tasks                │
│ tau_bench    │  3   │ TAU-Bench — Multi-turn agentic tasks with simulated tool execution │
│ terminal_bench│ 4   │ Terminal-Bench — 89 CLI tasks executed in Docker containers        │
└──────────────┴──────┴────────────────────────────────────────────────────────────────────┘
```

### run

```
llm-bench --model <model> run [OPTIONS] BENCHMARKS...
```

Runs one or more benchmarks sequentially. For each benchmark, datasets are loaded, all samples are dispatched concurrently (up to `--concurrency` at a time), results are evaluated, and a JSON file is saved to `--output-dir`.

**Arguments:**

| Argument | Description |
|----------|-------------|
| `BENCHMARKS...` | One or more benchmark names (space-separated). Must match entries from `llm-bench list`. |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--concurrency INTEGER` | `8` | Maximum number of requests in flight simultaneously. Controls the `asyncio.Semaphore` used by the runner. Higher values increase throughput but may overwhelm the server or cause OOM on large models. |
| `--max-samples INTEGER` | *(all samples)* | Truncate each benchmark to at most this many samples. Useful for quick validation runs. When omitted, the full dataset is used. |
| `--temperature FLOAT` | `0.0` | Sampling temperature passed to every completion request. Use `0.0` for greedy (deterministic) decoding. |
| `--max-tokens INTEGER` | `2048` | Default maximum tokens to generate per request. Individual benchmarks override this with their own tighter limits (e.g., MCQ benchmarks use 32–64 tokens). The benchmark-level value takes precedence when it is smaller. |
| `--seed INTEGER` | `42` | Random seed passed to the vLLM server for reproducible sampling. Set to a different value or use `--no-stream` to disable seed propagation. |
| `--no-stream` | *(streaming enabled)* | Disable streaming responses. TTFT will not be measured (all TTFT fields will be `null`). Use when the server does not support streaming or when benchmarking pure throughput. |
| `--scrape-prometheus` | *(disabled)* | After each benchmark run, scrape vLLM's Prometheus `/metrics` endpoint (default: `http://localhost:8000/metrics`) and attach the raw text to the result. Requires the `prometheus_url` config field if the metrics port differs. |

**Example — run two benchmarks with 16 concurrent requests and limit to 100 samples each:**

```bash
llm-bench \
    --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-72B-Instruct \
    --output-dir ./runs \
    run mmlu humaneval \
    --concurrency 16 \
    --max-samples 100 \
    --temperature 0.0 \
    --seed 1337
```

**Example — run Tier 4 SWE-bench without streaming (Docker required):**

```bash
llm-bench \
    --model claude-3-5-sonnet \
    run swe_bench \
    --concurrency 2 \
    --no-stream \
    --max-samples 10
```

### compare

```
llm-bench compare FILE [FILE ...]
```

Loads two or more result JSON files and prints a formatted side-by-side comparison table. Each column corresponds to one result file and shows the benchmark name and model on two lines in the column header.

Metrics displayed:

- Accuracy (proportion of `correct=True` samples)
- Number of samples
- Score (if present; some benchmarks emit a continuous score distinct from accuracy)
- 95% confidence interval on accuracy (normal approximation: mean ± 1.96 * SE)
- TTFT mean, P50, P95 (seconds)
- Generation TPS mean
- End-to-end latency mean and P95 (seconds)
- Effective throughput (requests per second over total wall time)

**Example:**

```bash
llm-bench compare \
    results/mmlu_modelA_20260101T000000Z.json \
    results/mmlu_modelB_20260101T010000Z.json
```

---

## Tier System

Benchmarks are assigned to one of four tiers reflecting their complexity, runtime, and infrastructure requirements.

| Tier | Focus | Typical Runtime | Requirements |
|------|-------|----------------|--------------|
| 1 | Core capability (knowledge, reasoning, code, instruction following) | Minutes to a few hours | vLLM server only |
| 2 | Long-context understanding and multi-turn conversations | Hours | vLLM server only |
| 3 | Tool use, function calling, and multi-turn agentic behavior | Hours | vLLM server with tool support |
| 4 | Real-world code and terminal task completion with execution verification | Hours to days | Docker Engine + `llm-benchmarker[docker]` |

Tier assignment is a class attribute on each benchmark (`tier: int`) and is displayed by `llm-bench list`.

---

## Benchmark Catalog

### Tier 1 — Core Capability

#### mmlu

| Property | Value |
|----------|-------|
| Dataset | `cais/mmlu`, split `test` |
| Items | 15,908 |
| What it tests | Broad world knowledge and academic reasoning across 57 subjects including mathematics, history, medicine, law, and more |
| Evaluator | MCQ extraction (`mcq.evaluate_mcq`) with 4 options (A–D) |
| `max_tokens` | 32 |
| System prompt | "You are a knowledgeable assistant. Answer multiple-choice questions with only the letter of the correct answer." |
| Special requirements | None |

Each question is formatted with the four choices labelled A–D, followed by "Answer with the letter only." The subject (e.g., `abstract_algebra`, `clinical_knowledge`) is stored in `details.subject` in each sample result for per-subject analysis.

---

#### mmlu_pro

| Property | Value |
|----------|-------|
| Dataset | `TIGER-Lab/MMLU-Pro`, split `test` |
| Items | 12,000+ |
| What it tests | Harder version of MMLU with 10-option questions (A–J), requiring stronger reasoning to discriminate among more distractors |
| Evaluator | MCQ extraction (`mcq.evaluate_mcq`) with 10 options (A–J) |
| `max_tokens` | 64 |
| System prompt | "You are a knowledgeable assistant. Answer multiple-choice questions with only the letter of the correct answer." |
| Special requirements | None |

The larger option set makes random-choice accuracy 10% and places greater demand on model knowledge. Category is stored in `details.category`.

---

#### humaneval

| Property | Value |
|----------|-------|
| Dataset | `openai/openai_humaneval`, split `test` |
| Items | 164 |
| What it tests | Python code generation; each problem is a function stub and docstring that the model must complete |
| Evaluator | Subprocess code execution (`code_execution.run_humaneval_test`) — compiles and runs HumanEval test harness |
| `max_tokens` | 512 |
| System prompt | "You are an expert Python programmer. Complete the function as requested. Return only code, no explanations." |
| Special requirements | `python3` in `PATH` |

The evaluator strips markdown fences if present and handles both "function body only" and "full function including signature" outputs. The assembled code (prompt + completion + test harness) is executed in a subprocess with a 10-second timeout. Results record `pass@1` by default.

The `metrics.compute_pass_at_k(n, c, k)` utility implements the unbiased estimator for `pass@k` if you generate multiple samples per problem and want `pass@5` or `pass@10`.

---

#### ifeval

| Property | Value |
|----------|-------|
| Dataset | `google/IFEval`, split `train` |
| Items | ~500 |
| What it tests | Instruction following: each prompt contains one or more verifiable constraints that the response must satisfy |
| Evaluator | IFEval rule engine (`ifeval_rules.verify_instruction`) — 25 constraint types |
| `max_tokens` | 2048 |
| System prompt | None |
| Special requirements | None |

A sample is marked `correct` only if **all** constraints in `instruction_id_list` pass. The `details` field records per-instruction pass/fail and counts. See the [IFEval evaluator section](#ifeval_rules) for the full list of 25 constraint types.

---

#### hle

| Property | Value |
|----------|-------|
| Dataset | `cais/hle`, split `test` |
| Items | 2,500 (text-only; image questions are automatically skipped) |
| What it tests | Expert-level questions from Humanity's Last Exam — graduate-level questions in mathematics, science, and humanities intended to challenge frontier models |
| Evaluator | MCQ extraction for multiple-choice questions; exact/contains match for short-answer questions |
| `max_tokens` | 256 |
| System prompt | None |
| Special requirements | None |

Mixed question format: `question_type == "multiple_choice"` uses the MCQ evaluator (4 options); all other types use `exact_match` or `contains_match` on the raw response.

---

#### longbench_v2

| Property | Value |
|----------|-------|
| Dataset | `THUDM/LongBench-v2`, split `test` |
| Items | 503 |
| What it tests | Long-context comprehension — each question includes a long document or document set ranging from 8,000 words to 2 million words, followed by a 4-option MCQ |
| Evaluator | MCQ extraction (`mcq.evaluate_mcq`) with 4 options |
| `max_tokens` | 32 |
| System prompt | None |
| Special requirements | A model with a context window large enough to handle the document. Many items require 32k+ context. |

The full document context is prepended to the question and choices in the user message. This is the most context-heavy Tier 1 benchmark and may require reducing `--concurrency` or using a model with 128k+ context.

---

#### graphwalks

| Property | Value |
|----------|-------|
| Dataset | `openai/graphwalks`, split `train` |
| Items | 1,150 |
| What it tests | Graph reasoning — questions about reachability, connected components, or traversal paths on explicitly described graphs |
| Evaluator | Set comparison (`set_comparison.set_exact_match` and `jaccard_similarity`) |
| `max_tokens` | 512 |
| System prompt | None |
| Special requirements | None |

`correct` is set by exact set equality; `score` (0–1) is the Jaccard similarity between the predicted and expected node/edge sets. The Jaccard score provides partial credit when the model identifies some but not all members of the set.

---

### Tier 2 — Long-Context and Multi-Turn

#### multi_if

| Property | Value |
|----------|-------|
| Dataset | `facebook/Multi-IF`, split `test` |
| Items | 4,501 |
| What it tests | Multi-turn multilingual instruction following — 3-turn conversations where each turn introduces new verifiable constraints that must be satisfied in that turn's response |
| Evaluator | IFEval rule engine applied per-turn; correct only if all turns pass all constraints |
| `max_tokens` | Inherits from `RunConfig` |
| System prompt | None |
| Multi-turn | Yes (`is_multi_turn = True`) |
| Special requirements | None |

The conversation history is maintained across all 3 turns. The runner calls `multi_turn()` directly because `is_multi_turn` is `True`. `details.per_turn` records pass/fail for each turn individually. Primary latency metrics are taken from the last turn.

---

#### mrcr

| Property | Value |
|----------|-------|
| Dataset | `openai/mrcr`, split `train` |
| Items | 2,400 |
| What it tests | Multi-Round Coreference Resolution — retrieval of a specific answer buried in a very long multi-round conversation context (4,000 to 1,000,000 tokens) |
| Evaluator | Exact match and contains match on the predicted response |
| `max_tokens` | 256 |
| System prompt | None |
| Special requirements | Very large context window (up to 1M tokens for the hardest items) |

Dataset items provide pre-built `messages` arrays (multi-round conversation history). `details.context_length` records the token count of the context for analysis by length bucket.

---

### Tier 3 — Agentic and Tool Use

#### bfcl

| Property | Value |
|----------|-------|
| Dataset | `gorilla-llm/Berkeley-Function-Calling-Leaderboard`, split `test` |
| Items | Varies (full leaderboard dataset) |
| What it tests | Function calling accuracy — given a natural language request and a set of tool definitions in OpenAI tools format, the model must invoke the correct function with correct arguments |
| Evaluator | AST-based function call comparison (`ast_eval.compare_function_calls`) |
| `max_tokens` | 512 |
| Multi-turn | Yes (uses `multi_turn` path to access raw `tool_calls` in the response) |
| Requires tools | Yes |
| Special requirements | Model must support the OpenAI tool-calling API |

Tool definitions from the dataset are translated into OpenAI `tools` format and passed in the request. The response's `tool_calls` are extracted and compared against the expected calls. If the model returns plain-text function calls (no structured `tool_calls`), the AST/JSON parser attempts to extract them from the text.

Comparison is order-independent: each predicted call is matched against any unmatched expected call by name and normalized argument equality.

---

#### tau_bench

| Property | Value |
|----------|-------|
| Dataset | `sierra-research/tau-bench`, split `test` (falls back to `train`) |
| Items | Varies |
| What it tests | Multi-turn agentic task completion — the model must invoke a series of tools across up to 20 turns to accomplish a user goal, with simulated tool responses |
| Evaluator | Action sequence comparison: executed action names must match expected action names in order |
| `max_tokens` | 1024 per turn |
| Multi-turn | Yes (up to 20 turns) |
| Requires tools | Yes |
| Special requirements | Model must support tool calling |

Each turn, the runner sends the current conversation (including previous tool results) to the model and processes any `tool_calls`. Tool execution is simulated: every call returns `{"status": "success", "result": "Executed <name>"}`. The agent loop breaks when `finish_reason == "stop"` with no tool calls, or after `MAX_TURNS = 20` turns. Correct if the sequence of called tool names matches the expected action list.

---

### Tier 4 — Docker-Sandboxed Execution

All Tier 4 benchmarks inherit from `DockerBenchmark` and require Docker Engine and `pip install llm-benchmarker[docker]`.

Each benchmark creates a Docker container, runs setup commands inside it, conducts a multi-turn conversation with the model (showing the model command outputs), and validates the final state by running a validation script inside the container. Containers are isolated with no network access (`network_mode="none"`), 2 GB memory limit, and 1 CPU core (`cpu_period=100000, cpu_quota=100000`). Containers are always removed in a `finally` block after each sample.

---

#### terminal_bench

| Property | Value |
|----------|-------|
| Dataset | `laude-institute/terminal-bench`, split `test` (falls back to `train`) |
| Items | 89 |
| Docker image | `ubuntu:22.04` |
| What it tests | CLI task completion — the model is given a Linux task description and must issue bash commands interactively, seeing command output, until the task is done |
| Evaluator | Validation script exit code (0 = success) |
| `max_tokens` | 1024 per turn |
| Multi-turn | Yes (up to 15 turns) |
| Special requirements | Docker Engine, `llm-benchmarker[docker]` |

The model responds with bash commands wrapped in ` ```bash ``` ` fences (or prefixed with `$ `). Commands are executed inside the Ubuntu container and output is fed back as the next user message. A dataset-provided setup script initializes the environment before the conversation begins. A dataset-provided validation script (run after the conversation) determines success.

---

#### swe_bench

| Property | Value |
|----------|-------|
| Dataset | `SWE-bench/SWE-bench_Verified`, split `test` |
| Items | 500 |
| Docker image | `python:3.11` |
| What it tests | Real-world bug fixing — each task is a GitHub issue from a real Python repository; the model generates a unified diff patch that must make failing tests pass without breaking passing tests |
| Evaluator | `FAIL_TO_PASS` test suite (must now pass) + `PASS_TO_PASS` test suite (must still pass), both run via `pytest` inside Docker |
| `max_tokens` | 4096 |
| Multi-turn | Yes (single model turn, then patch application and test execution) |
| Special requirements | Docker Engine, `llm-benchmarker[docker]`, internet access (clones from GitHub) |

Workflow:
1. Container starts with `python:3.11`
2. Repository is cloned from GitHub and checked out at `base_commit`
3. Model is prompted to produce a unified diff patch
4. Patch is applied via `git apply` (with `--3way` fallback)
5. Test patch is applied
6. `FAIL_TO_PASS` tests are run; all must pass
7. Up to 10 `PASS_TO_PASS` regression tests are run; all must still pass
8. `correct = f2p_pass and p2p_pass`

---

#### swe_bench_pro

| Property | Value |
|----------|-------|
| Dataset | `SWE-bench/SWE-bench`, split `test` |
| Items | 1,865 |
| Docker image | `python:3.11` |
| What it tests | Same as `swe_bench` but with the full (unverified) SWE-bench dataset, providing a larger and harder test set |
| Evaluator | Same FAIL_TO_PASS / PASS_TO_PASS methodology as `swe_bench` |
| `max_tokens` | 4096 (inherited) |
| Multi-turn | Yes (inherited from `SWEBenchBenchmark`) |
| Special requirements | Same as `swe_bench` |

`SWEBenchProBenchmark` inherits all multi-turn logic from `SWEBenchBenchmark` and only overrides `load_dataset` to pull from the full SWE-bench dataset.

---

## Evaluators

### exact_match

**File:** `evaluators/exact_match.py`

Two comparison functions, both operating on normalized text:

- `normalize(text)`: applies Unicode NFKD normalization, lowercases, strips leading/trailing whitespace, removes all non-word non-whitespace characters (punctuation), and collapses runs of whitespace to a single space.
- `exact_match(predicted, expected)`: returns `True` if `normalize(predicted) == normalize(expected)`.
- `contains_match(predicted, expected)`: returns `True` if `normalize(expected)` is a substring of `normalize(predicted)`.

Used by: `hle` (short-answer questions), `mrcr`.

---

### mcq

**File:** `evaluators/mcq.py`

Extracts a single letter answer from free-form model output using a cascade of regex heuristics:

1. Patterns like "the answer is B", "answer: B", "(B)", or "B." at end of line.
2. Standalone letter on its own line.
3. Last valid letter found anywhere in the response.

`extract_mcq_answer(response, num_options=4)` supports 4-option (A–D) and 10-option (A–J) MCQs by slicing `LETTERS_10`.

`evaluate_mcq(response, expected, num_options=4)` returns `True` if the extracted letter matches the expected letter (case-insensitive).

Used by: `mmlu`, `mmlu_pro`, `hle` (MCQ questions), `longbench_v2`.

---

### code_execution

**File:** `evaluators/code_execution.py`

Executes Python code in an async subprocess:

- `execute_python(code, timeout=10.0)`: creates a `python3 -c <code>` subprocess, pipes stdout and stderr, enforces a timeout via `asyncio.wait_for`. Returns `(True, stdout)` on exit code 0, `(False, stderr)` otherwise.
- `run_humaneval_test(prompt, completion, test_code, entry_point, timeout=10.0)`: concatenates `prompt + completion + test_code + "check(entry_point)"` and calls `execute_python`. This forms a complete, self-contained Python script that defines the function and then calls the HumanEval test harness.

Used by: `humaneval`.

---

### ifeval_rules

**File:** `evaluators/ifeval_rules.py`

Implements 25 instruction-following verifiers matching the Google IFEval specification. Each verifier is registered by ID using the `@_register("id:subtype")` decorator.

**Keyword constraints:**
- `keywords:existence` — all listed keywords present in response
- `keywords:frequency` — keyword appears "at least"/"at most"/"exactly" N times
- `keywords:forbidden_words` — none of the listed words appear
- `keywords:letter_frequency` — a specific letter appears with a given frequency relation

**Language:**
- `language:response_language` — checks that the response is in the specified language (simplified: checks ASCII ratio for English; returns `True` for other languages)

**Length constraints:**
- `length_constraints:number_words` — word count satisfies a relation
- `length_constraints:number_sentences` — sentence count satisfies a relation (split on `.!?`)
- `length_constraints:number_paragraphs` — paragraph count satisfies a relation (split on `\n\n`)
- `length_constraints:nth_paragraph_first_word` — first word of the Nth paragraph matches expected

**Detectable content:**
- `detectable_content:number_placeholders` — response contains at least N `[...]` placeholders
- `detectable_content:postscript` — response contains a P.S. section

**Detectable format:**
- `detectable_format:number_bullet_lists` — at least N bullet points (`-`, `*`, or `•`)
- `detectable_format:constrained_response` — response is non-empty
- `detectable_format:number_highlighted_sections` — at least N `*highlighted*` spans
- `detectable_format:multiple_sections` — at least N section headers
- `detectable_format:json_format` — response is valid JSON or contains a JSON code block
- `detectable_format:title` — response contains a `<<title>>` marker
- `detectable_format:number_words_in_section` — word count in section satisfies a relation

**Combination:**
- `combination:two_responses` — response contains two parts separated by `******`
- `combination:repeat_prompt` — response starts with the original prompt text

**Start/end:**
- `startend:end_checker` — response ends with a specific phrase
- `startend:quotation` — entire response is wrapped in double quotes

**Case:**
- `change_case:english_capital` — all alphabetic characters are uppercase
- `change_case:english_lowercase` — all alphabetic characters are lowercase

**Punctuation:**
- `punctuation:no_comma` — response contains no commas

`verify_instruction(response, instruction_id, kwargs)` looks up the verifier by ID and returns a bool. Returns `False` if the ID is not registered (no crash). Used by: `ifeval`, `multi_if`.

---

### ast_eval

**File:** `evaluators/ast_eval.py`

Parses and compares function calls for the BFCL benchmark.

`parse_function_call(text)` attempts (in order):
1. JSON parsing of the entire text as a list or dict
2. JSON extraction from a ` ```json ``` ` code block
3. Python AST parsing of a function call expression (e.g., `get_weather(city="London", unit="celsius")`)
4. AST parsing wrapped in a list literal for multiple calls

`compare_function_calls(predicted, expected)`:
- Returns `(False, details)` immediately if the count of calls differs
- For each predicted call, finds an unmatched expected call with the same name and identical `normalize_value(arguments)`
- `normalize_value` recursively sorts dict keys and tries to parse string values as JSON, enabling robust comparison across serialization variations

Used by: `bfcl`.

---

### set_comparison

**File:** `evaluators/set_comparison.py`

Extracts a set of string items from free-form model output and compares it to the expected set.

`extract_set(text)`:
- Strips surrounding `{...}` or `[...]` brackets if present
- Splits on commas and newlines
- Strips whitespace and quote characters from each item

`set_exact_match(predicted, expected)`: returns `True` if extracted sets are identical.

`jaccard_similarity(predicted, expected)`: returns `|intersection| / |union|` as a float in [0, 1]. Returns 1.0 if both sets are empty, 0.0 if exactly one is empty.

Used by: `graphwalks` (score = Jaccard, correct = exact match).

---

## Performance Metrics

### Per-Request Metrics (`RequestMetrics`)

Collected by `InstrumentedClient` for every API call:

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Random 8-hex-character ID for log correlation |
| `ttft_seconds` | `float \| None` | Time to first token (seconds). `None` when streaming is disabled. |
| `total_latency_seconds` | `float` | Wall-clock time from request dispatch to full response |
| `prompt_tokens` | `int` | Number of prompt tokens (from usage data) |
| `completion_tokens` | `int` | Number of generated tokens (from usage data) |
| `prefill_time_seconds` | `float \| None` | Alias for `ttft_seconds` (set during derived metric computation) |
| `decode_time_seconds` | `float \| None` | `total_latency_seconds - ttft_seconds` |
| `tokens_per_second_generation` | `float \| None` | `(completion_tokens - 1) / decode_time_seconds` |
| `tokens_per_second_prompt` | `float \| None` | `prompt_tokens / ttft_seconds` |
| `error` | `str \| None` | Exception message if the request failed |

### Aggregated Metrics (`AggregatedMetrics`)

Computed by `aggregate_metrics()` over all successful requests in a benchmark run:

| Field | Description |
|-------|-------------|
| `ttft_p50` | Median TTFT across all streaming requests |
| `ttft_p95` | 95th percentile TTFT |
| `ttft_p99` | 99th percentile TTFT |
| `ttft_mean` | Mean TTFT |
| `generation_tps_mean` | Mean generation throughput (tokens/sec during decode phase) |
| `generation_tps_p50` | Median generation throughput |
| `prompt_tps_mean` | Mean prompt processing throughput (tokens/sec during prefill phase) |
| `latency_p50` | Median end-to-end latency |
| `latency_p95` | 95th percentile end-to-end latency |
| `latency_p99` | 99th percentile end-to-end latency |
| `latency_mean` | Mean end-to-end latency |
| `total_prompt_tokens` | Sum of prompt tokens across all successful requests |
| `total_completion_tokens` | Sum of completion tokens across all successful requests |
| `total_requests` | Total number of requests attempted |
| `failed_requests` | Number of requests that raised an exception |
| `wall_time_seconds` | Total elapsed time for the entire benchmark run |
| `effective_throughput_rps` | `successful_requests / wall_time_seconds` |

All TTFT and latency fields use `numpy.percentile` and `numpy.mean` for statistical correctness.

### Statistical Utilities

`compute_pass_at_k(n, c, k)` implements the unbiased pass@k estimator from the HumanEval paper:

```
pass@k = 1 - product((n-c choose j) / (n choose j)) for j in range(k)
```

Simplified: `1 - prod(1 - k / range(n-c+1, n+1))`.

`confidence_interval_95(scores)` computes a 95% CI using the normal approximation: `mean ± 1.96 * (std / sqrt(n))`.

---

## How Streaming TTFT Measurement Works

When streaming is enabled (the default), `InstrumentedClient._stream_chat()` works as follows:

1. `start = time.perf_counter()` is recorded immediately before `await self._client.chat.completions.create(stream=True, stream_options={"include_usage": True})`.
2. The response is consumed chunk by chunk with `async for chunk in stream`.
3. On the first chunk where `chunk.choices[0].delta.content` is non-empty (the first actual text token), `metrics.ttft_seconds = time.perf_counter() - start` is set and a `first_token_seen` flag is raised.
4. After the stream is exhausted, `end = time.perf_counter()` is recorded and `metrics.total_latency_seconds = end - start`.
5. Usage data (prompt and completion token counts) arrive on the final chunk via `chunk.usage` (requires `stream_options={"include_usage": True}`).
6. `_compute_derived()` then fills in `decode_time_seconds`, `tokens_per_second_generation`, and `tokens_per_second_prompt`.

The `stream_options={"include_usage": True}` parameter is a vLLM/OpenAI extension that causes token counts to be emitted on the final stream chunk. Without it, `prompt_tokens` and `completion_tokens` would be zero.

When `--no-stream` is used, `_non_stream_chat()` is called instead: `ttft_seconds` remains `None`, and `total_latency_seconds` covers the full round-trip. Usage is taken from `response.usage`.

`chat_completion_raw()` always uses non-streaming and returns the full response object (needed for `tool_calls` in BFCL and TAU-Bench).

---

## Async Execution Engine

`BenchmarkRunner` manages the async execution of all samples in a benchmark:

```python
self._semaphore = asyncio.Semaphore(config.run.concurrency)
```

The semaphore limits the number of requests in flight at any time. The full execution flow:

1. `load_dataset()` is called to get all `BenchmarkSample` objects synchronously within the async context.
2. A `_run_one(sample)` coroutine is created for each sample. All coroutines are gathered with `asyncio.gather(*tasks, return_exceptions=True)`.
3. Each `_run_one` calls `_run_sample`, which `async with self._semaphore` — blocking if `concurrency` slots are occupied.
4. Inside the semaphore, the runner checks `benchmark.is_multi_turn`: if `True`, it calls `benchmark.multi_turn(sample, client, config)` directly; otherwise it builds the message list, prepends the system prompt if one is defined, merges request kwargs (with benchmark-level values overriding run-level defaults), and calls `client.chat_completion()`.
5. After evaluation, `SampleResult.metrics` is set to the `RequestMetrics` from that request.
6. Exceptions caught by `asyncio.gather` produce a failed `SampleResult` with `correct=False` and the exception message in `details.error`.
7. After all samples complete, `aggregate_metrics()` is called on the collected `RequestMetrics` list with the total wall time.
8. The rich progress bar (`Progress` with `SpinnerColumn`, `BarColumn`, `MofNCompleteColumn`) advances one step per completed sample.

This design means all samples for a benchmark are dispatched as concurrent tasks immediately; the semaphore acts as a rate limiter rather than a sequential queue, maximizing server utilization.

---

## Results Format

Each benchmark run produces a JSON file at:

```
<output-dir>/<benchmark_name>_<model_name>_<timestamp>Z.json
```

Slashes in model names are replaced with underscores. Timestamp is UTC in `YYYYMMDDTHHMMSSz` format.

### Top-level structure

```json
{
  "benchmark": "mmlu",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "timestamp": "20260220T120000Z",
  "config": {
    "base_url": "http://localhost:8000/v1",
    "concurrency": 8,
    "streaming": true,
    "temperature": 0.0,
    "max_tokens": 2048,
    "seed": 42
  },
  "name": "mmlu",
  "num_samples": 15908,
  "accuracy": 0.6312,
  "score": 0.6312,
  "aggregated_metrics": { ... },
  "sample_results": [ ... ]
}
```

### `aggregated_metrics` object

```json
{
  "ttft_p50": 0.182,
  "ttft_p95": 0.451,
  "ttft_p99": 0.612,
  "ttft_mean": 0.201,
  "generation_tps_mean": 142.3,
  "generation_tps_p50": 138.1,
  "prompt_tps_mean": 4210.5,
  "latency_p50": 0.231,
  "latency_p95": 0.502,
  "latency_p99": 0.720,
  "latency_mean": 0.248,
  "total_prompt_tokens": 2450112,
  "total_completion_tokens": 508256,
  "total_requests": 15908,
  "failed_requests": 0,
  "wall_time_seconds": 1842.3,
  "effective_throughput_rps": 8.64
}
```

Fields with `None` values are omitted from the dict (see `AggregatedMetrics.to_dict()`).

### `sample_results` array

Each element:

```json
{
  "id": "mmlu_0",
  "correct": true,
  "score": 1.0,
  "predicted": "B",
  "expected": "B",
  "details": {
    "subject": "abstract_algebra"
  },
  "metrics": {
    "ttft_seconds": 0.174,
    "total_latency_seconds": 0.213,
    "prompt_tokens": 154,
    "completion_tokens": 2,
    "tokens_per_second_generation": 141.2
  }
}
```

The `details` dict is benchmark-specific. For IFEval it contains `per_instruction`; for GraphWalks it contains `jaccard`; for SWE-bench it contains `fail_to_pass_results` and `pass_to_pass_results`.

---

## Comparing Results

The `compare` command loads any number of result JSON files and renders a rich table:

```bash
llm-bench compare results/mmlu_model_a_*.json results/mmlu_model_b_*.json
```

The table includes:

| Row | Description |
|-----|-------------|
| Accuracy | Proportion of `correct=True` samples |
| Samples | Total sample count |
| Score | Average continuous score (if present) |
| 95% CI | `[lower, upper]` confidence interval on accuracy |
| TTFT Mean (s) | Mean time to first token |
| TTFT P50 (s) | Median TTFT |
| TTFT P95 (s) | 95th percentile TTFT |
| Gen TPS Mean | Mean generation tokens per second |
| Latency Mean (s) | Mean end-to-end latency |
| Latency P95 (s) | 95th percentile end-to-end latency |
| Throughput (req/s) | Effective requests per second over wall time |

Confidence intervals are computed from `sample_results` in the JSON file using `confidence_interval_95()`.

---

## Configuration Reference

### `VLLMConfig` (Pydantic model)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_url` | `str` | `"http://localhost:8000/v1"` | Full base URL of the OpenAI-compatible API including `/v1` path. |
| `api_key` | `str` | `"EMPTY"` | API key. Use `"EMPTY"` for unauthenticated vLLM servers. |
| `model` | `str` | *(required)* | Model identifier as returned by `/v1/models`. |
| `timeout` | `float` | `300.0` | HTTP request timeout in seconds. Increase for very long SWE-bench or long-context requests. |

### `RunConfig` (Pydantic model)

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `concurrency` | `int` | `8` | `>= 1` | Semaphore size; maximum simultaneous in-flight requests. |
| `max_samples` | `int \| None` | `None` | — | Per-benchmark sample limit. `None` = all samples. |
| `streaming` | `bool` | `True` | — | Enable streaming responses. Set to `False` with `--no-stream`. |
| `seed` | `int \| None` | `42` | — | Sampling seed. `None` = do not pass seed to the API. |
| `temperature` | `float` | `0.0` | — | Sampling temperature. `0.0` = greedy. |
| `max_tokens` | `int` | `2048` | — | Default max tokens per request (overridden per benchmark). |
| `output_dir` | `Path` | `Path("results")` | — | Directory for JSON output files. |
| `data_dir` | `Path` | `Path("data")` | — | Hugging Face dataset cache directory. |
| `scrape_prometheus` | `bool` | `False` | — | Whether to scrape vLLM's Prometheus endpoint after each run. |
| `prometheus_url` | `str` | `"http://localhost:8000/metrics"` | — | URL of the Prometheus metrics endpoint. |

### `BenchmarkConfig` (Pydantic model)

Combines `VLLMConfig` and `RunConfig` into a single object passed through the system:

```python
class BenchmarkConfig(BaseModel):
    vllm: VLLMConfig
    run: RunConfig
```

---

## Examples

### Run MMLU with a remote server and custom concurrency

```bash
llm-bench \
    --base-url http://192.168.1.100:8000/v1 \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --output-dir ./benchmark-results \
    run mmlu \
    --concurrency 32 \
    --temperature 0.0 \
    --seed 0
```

### Quick validation of all Tier 1 benchmarks (50 samples each)

```bash
llm-bench \
    --model Qwen/Qwen2.5-7B-Instruct \
    run mmlu mmlu_pro humaneval ifeval hle longbench_v2 graphwalks \
    --max-samples 50 \
    --concurrency 4
```

### Disable streaming to measure only throughput (not TTFT)

```bash
llm-bench \
    --model meta-llama/Llama-3.1-8B-Instruct \
    run mmlu mmlu_pro \
    --no-stream \
    --concurrency 64
```

### Run BFCL function-calling benchmark

```bash
llm-bench \
    --model Qwen/Qwen2.5-72B-Instruct \
    run bfcl \
    --concurrency 4 \
    --max-samples 200
```

### Run SWE-bench Verified (requires Docker)

```bash
pip install "llm-benchmarker[docker]"

llm-bench \
    --model meta-llama/Llama-3.1-70B-Instruct \
    run swe_bench \
    --concurrency 1 \
    --max-samples 20 \
    --no-stream
```

### Compare multiple models on MMLU

```bash
# Run both models
for model in Llama-3.1-8B-Instruct Llama-3.1-70B-Instruct; do
    llm-bench --model "meta-llama/$model" run mmlu --max-samples 500
done

# Compare results
llm-bench compare results/mmlu_meta-llama_Llama-3.1-8B-Instruct_*.json \
                  results/mmlu_meta-llama_Llama-3.1-70B-Instruct_*.json
```

### Scrape Prometheus metrics alongside benchmark results

```bash
llm-bench \
    --model Qwen/Qwen2.5-72B-Instruct \
    run mmlu \
    --scrape-prometheus
```

### Writing a custom benchmark

Create a new file in `src/llm_benchmarker/benchmarks/` and apply the `@register` decorator:

```python
from . import register
from .base import BaseBenchmark, BenchmarkSample, SampleResult

@register
class MyBenchmark(BaseBenchmark):
    name = "my_benchmark"
    description = "My custom benchmark"
    tier = 1

    async def load_dataset(self, data_dir, max_samples=None):
        # Return a list of BenchmarkSample
        return [
            BenchmarkSample(
                id="sample_0",
                messages=[{"role": "user", "content": "What is 2 + 2?"}],
                metadata={"expected": "4"},
            )
        ]

    async def evaluate(self, sample, response):
        correct = "4" in response
        return SampleResult(
            id=sample.id,
            correct=correct,
            score=1.0 if correct else 0.0,
            predicted=response,
            expected=sample.metadata["expected"],
        )
```

Import it in `benchmarks/__init__.py` to trigger registration:

```python
from . import my_benchmark  # noqa: F401
```

Then run it:

```bash
llm-bench --model mymodel run my_benchmark
```

---

## Development

### Install development dependencies

```bash
pip install -e ".[dev]"
```

This installs `pytest>=8.0`, `pytest-asyncio>=0.23`, and `ruff>=0.4` in addition to the base dependencies.

### Running tests

```bash
pytest
```

Asyncio mode is set to `auto` in `pyproject.toml`, so `async def test_*` functions are automatically handled without decorators.

### Linting and formatting

```bash
ruff check src/
ruff format src/
```

The `ruff` configuration targets Python 3.11 (`target-version = "py311"`).

### Project structure conventions

- All benchmarks live in `src/llm_benchmarker/benchmarks/` and register themselves with `@register`.
- All evaluators live in `src/llm_benchmarker/evaluators/` and expose pure functions (no classes).
- Benchmarks import evaluators lazily inside `evaluate()` or `multi_turn()` to avoid circular imports and keep startup fast.
- `BaseBenchmark.get_request_kwargs()` is the correct place to override per-benchmark API parameters (e.g., `max_tokens`, `tools`). The runner calls this method and merges the result with run-level defaults, with benchmark-level values taking precedence via `kwargs.setdefault(...)`.
- Multi-turn benchmarks set `is_multi_turn = True` on the class and implement `multi_turn()`. The `evaluate()` method should raise `NotImplementedError` for these.
- Docker benchmarks inherit from `DockerBenchmark` (not `BaseBenchmark` directly) and implement only `load_dataset()` and `multi_turn()`.
