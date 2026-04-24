# Hawker

[English](./README.md) | [简体中文](./README.zh-CN.md)

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](./LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)
[![Mypy](https://img.shields.io/badge/types-mypy-blue.svg)](https://mypy-lang.org/)

Hawker is an autonomous web intelligence agent for turning browser interactions, network traffic, and LLM reasoning into auditable structured outputs.

It is designed for workflows where a plain scraper is not enough:

- Interactive pages and multi-step flows
- Browser-assisted extraction with API replay when available
- Reproducible run artifacts for debugging and review
- Structured final delivery with traceability

## Project Status

Hawker is under active development and should currently be considered **experimental but usable**.

Recommended use cases:

- Internal research and automation
- Agent engineering experiments
- Browser-first data collection pipelines

Not yet recommended without additional hardening:

- Unreviewed production deployments in regulated environments
- Security-sensitive workloads without dependency governance and secret isolation

## Why Hawker

Most web automation stacks stop at "the task ran" or "the page was clicked". Hawker focuses on what happens after that:

- **Structured delivery**: final outputs are written to `result.json` with items and delivery artifacts.
- **Operational traceability**: every run emits logs, notebook-style execution history, and LLM IO traces.
- **Browser + network strategy**: the agent can favor API replay over fragile DOM extraction when network signals are available.
- **Long-task ergonomics**: history compression, runtime checkpoints, and memory-assisted execution reduce prompt bloat and retry cost.

## Architecture

At a high level, Hawker is composed of:

- **Agent runtime**: step orchestration, execution, final-delivery gating, and stopping logic
- **Browser layer**: Playwright/browser-use based interaction, DOM snapshots, and network logging
- **LLM layer**: provider abstraction, token accounting, evaluator/healer sidecars
- **Storage layer**: run artifacts, result export, notebook export, and memory persistence

Core output directories per run:

- `hawker_file/<run_id>/result/result.json`: user-facing delivery artifact
- `hawker_file/<run_id>/run.ipynb`: executable notebook-style trace
- `hawker_file/<run_id>/llm_io.json`: serialized model interaction log
- `log/<run_id>/`: application and run logs

## Quick Start

### 1. Install dependencies

This project uses [`uv`](https://github.com/astral-sh/uv).

```bash
git clone <your-fork-or-repo-url>
cd hawker
uv sync
uv run playwright install chromium
```

For end users who only want the CLI, the intended install path is:

```bash
pipx install hawker-agent
# or
uv tool install hawker-agent
playwright install chromium
```

### 2. Configure environment

```bash
hawker config init
```

By default, `hawker config init` starts an interactive setup wizard and writes
global CLI configuration to:

- `~/hawker/config.env`

For scripts or CI, skip the wizard and write a template:

```bash
hawker config init --no-interactive
```

Change individual values without opening the file:

```bash
hawker config set api-key
hawker config set model openai/gpt-5.4
hawker config set max-steps 20
```

Open the active config in your editor:

```bash
hawker config edit
```

Typical settings:

```ini
OPENAI_API_KEY=...
MODEL_NAME=openai/gpt-5.4
HEADLESS=false
```

Browser modes:

```ini
# Local browser (default)
BROWSER_PROVIDER=local
HEADLESS=false

# Browser Use Cloud
BROWSER_PROVIDER=browser_use_cloud
BROWSER_USE_API_KEY=...
BROWSER_USE_PROFILE_ID=...
BROWSER_USE_PROXY_COUNTRY_CODE=us
```

Notes:

- `BROWSER_PROVIDER=browser_use_cloud` makes Hawker create a Browser Use Cloud session automatically and connect through its returned CDP URL.
- `BROWSER_USE_PROFILE_ID` is strongly recommended. It lets the cloud browser reuse login state, cookies, and other persisted browser data across sessions.
- `BROWSER_CDP_URL` is still supported when you already have an existing browser endpoint and do not want Hawker to create one for you.

Inspect the effective configuration:

```bash
hawker config show
hawker doctor
```

Configuration precedence:

1. CLI flags
2. OS environment variables
3. `~/hawker/config.env`
4. Built-in defaults

Important configuration groups:

| Group | Variables |
| --- | --- |
| LLM | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL_NAME`, `SMALL_MODEL_NAME`, `REASONING_EFFORT` |
| Runtime | `MAX_STEPS`, `MAX_TOTAL_TOKENS`, `MAX_NO_PROGRESS_STEPS`, `MESSAGE_COMPRESSION_TOKENS` |
| Sidecars | `HEALER_ENABLED`, `FINAL_EVALUATOR_ENABLED`, `OBSERVER_ENABLED` |
| Browser | `HEADLESS`, `BROWSER_PROVIDER`, `BROWSER_EXECUTABLE_PATH`, `BROWSER_USER_DATA_DIR`, `BROWSER_STORAGE_STATE`, `BROWSER_CDP_URL`, `BROWSER_USE_API_KEY`, `BROWSER_USE_PROFILE_ID`, `BROWSER_USE_PROXY_COUNTRY_CODE` |
| Storage & logs | `SCRAPE_DIR`, `KNOWLEDGE_DB_PATH`, `LOG_LEVEL` |

Default storage paths now follow a simple user-home layout rather than the current working directory:

- `SCRAPE_DIR`: `~/hawker`
- `KNOWLEDGE_DB_PATH`: `~/hawker/knowledge.db`

If your target site requires authentication, Hawker can reuse browser state via:

- local Chrome user data directory
- exported `storage_state.json`
- an existing browser CDP endpoint

See `.env.example` for the currently supported variables.

### 3. Run a task

Interactive task file:

```bash
uv run run.py
```

CLI mode:

```bash
uv run python -m hawker_agent.cli "Collect GitHub Trending repositories" --max-steps 15
```

Packaged CLI mode:

```bash
hawker "Collect GitHub Trending repositories" --max-steps 15
hawker run "Collect GitHub Trending repositories" --max-steps 15
```

### 4. Smoke-check a packaged install

These are the minimum checks worth running before publishing:

```bash
hawker --help
hawker run "Collect one GitHub Trending repository" --max-steps 1
```

## Usage Model

Hawker is optimized for tasks that require judgment, not just selectors.

Examples:

- "Collect the latest papers on Web Agents and download the PDFs"
- "Open a product listing page, identify the real JSON data API, then replay it"
- "Summarize the latest posts from a public profile and return structured items"

The normal execution loop is:

1. Explore the page and network surface
2. Prefer replayable APIs when possible
3. Fall back to DOM extraction when needed
4. Validate and normalize collected items
5. Submit a final answer that passes delivery checks

## Development

Install the development toolchain:

```bash
uv sync --extra dev
```

Common commands:

```bash
uv run ruff check hawker_agent tests run.py
uv run mypy hawker_agent
uv run pytest -q
uv run python -m build
```

The repository currently uses:

- `ruff` for linting
- `mypy` in strict mode
- `pytest` and `pytest-asyncio` for tests

## Packaging & Release

Recommended release flow:

1. Build artifacts locally
2. Validate the wheel in a clean environment
3. Publish to TestPyPI
4. Install from TestPyPI and run the CLI smoke checks
5. Publish to PyPI

Build locally:

```bash
uv sync --extra dev
uv run python -m build
```

Typical TestPyPI publish:

```bash
uv run python -m twine upload --repository testpypi dist/*
```

Typical PyPI publish:

```bash
uv run python -m twine upload dist/*
```

For a one-command local release check:

```bash
bash scripts/release_check.sh
```

See [document/release_process.md](./document/release_process.md) for the fuller TestPyPI and PyPI checklist.

## Security

Please do not open public issues for sensitive vulnerabilities.

For security reporting instructions, see [SECURITY.md](./SECURITY.md).

## Contributing

Contributions are welcome. Before opening a pull request, please:

1. Run lint and tests locally
2. Keep changes focused and reviewable
3. Update documentation when behavior changes

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the contributor workflow.

## License

This project is licensed under the [Apache License 2.0](./LICENSE).

Third-party dependencies remain under their respective licenses. See:

- [NOTICE](./NOTICE)
- [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)

## Acknowledgements

Hawker builds on top of several excellent open source projects. In particular:

- [browser-use](https://github.com/browser-use/browser-use) for browser automation primitives and agent-oriented browser workflows
- [Playwright for Python](https://github.com/microsoft/playwright-python) for browser control and automation foundations
- [LiteLLM](https://github.com/BerriAI/litellm) for multi-provider LLM access
- [Langfuse](https://github.com/langfuse/langfuse) for observability and tracing integration
- [HTTPX](https://github.com/encode/httpx), [Typer](https://github.com/fastapi/typer), [Rich](https://github.com/Textualize/rich), [Jinja](https://github.com/pallets/jinja), [Pydantic Settings](https://github.com/pydantic/pydantic-settings), [nbformat](https://github.com/jupyter/nbformat), and [Boto3](https://github.com/boto/boto3)

We are grateful to the maintainers and contributors of these projects.
