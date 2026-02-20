# Installation Guide

Step-by-step instructions to go from `git clone` to a working `llm-bench` CLI.

---

## 1. Prerequisites

- **Python 3.11+** — verify with:
  ```bash
  python3 --version
  ```
  If below 3.11, upgrade before proceeding. The package will refuse to install on older versions.

- **Git** — to clone the repository.

- **A running vLLM server** (or any OpenAI-compatible API) — required for `check` and `run` commands. Not needed for `llm-bench list`.

- **Docker** (optional) — only needed for Tier 4 benchmarks (see step 7).

---

## 2. Clone the Repository

```bash
git clone https://github.com/lupascualex/llm-benchmarker.git
cd llm-benchmarker
```

---

## 3. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

Verify you're in the venv:
```bash
which python3
# Should show: /path/to/llm-benchmarker/.venv/bin/python3
```

---

## 4. Install the Package

```bash
pip install -e .
```

This installs `llm-benchmarker` and all its dependencies (click, openai, httpx, datasets, pydantic, rich, numpy) while keeping the source editable — any code changes take effect immediately.

---

## 5. Verify the Installation

Run this — it requires no server:

```bash
llm-bench list
```

You should see a table of 14 benchmarks:

```
                          Available Benchmarks
┏━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name           ┃ Tier ┃ Description                               ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ bfcl           │  3   │ BFCL — Berkeley Function Calling ...      │
│ graphwalks     │  1   │ GraphWalks — 1,150 graph reasoning ...    │
│ ...            │      │ (14 benchmarks total)                     │
└────────────────┴──────┴───────────────────────────────────────────┘
```

If you see this table, the CLI is installed and working.

---

## 6. First Benchmark Run

Start small to verify the full pipeline:

```bash
llm-bench --model <your-model-name> run mmlu --max-samples 10
```

This will:
1. Download the MMLU dataset from HuggingFace (first run only)
2. Send 10 questions to your vLLM server
3. Evaluate the responses
4. Save results as JSON in `./results/`

---

## 7. Docker Setup (Tier 4 Only)

Tier 4 benchmarks (`terminal_bench`, `swe_bench`, `swe_bench_pro`) execute code inside Docker containers. These benchmarks evaluate models on real-world tasks — CLI operations in a Linux shell, applying patches to Git repositories, running test suites — that require an isolated, sandboxed environment. Docker provides this sandbox so that model-generated code cannot affect the host system.

```bash
# Install Docker: https://docs.docker.com/get-docker/
docker --version

# Install the package with Docker extra
pip install -e ".[docker]"

# Verify Docker is accessible
docker run --rm hello-world
```

---

## Troubleshooting

**`llm-bench: command not found`**
- Make sure your virtual environment is activated: `source .venv/bin/activate`
- Reinstall: `pip install -e .`

**`pip install` fails with Python version error**
- You need Python 3.11+. Check with `python3 --version`.

**HumanEval `python3 not found` errors**
- The code execution evaluator calls `python3` via subprocess. Ensure `python3` is on your PATH.

**Docker permission errors (Tier 4)**
- Add your user to the docker group: `sudo usermod -aG docker $USER` then re-login.
