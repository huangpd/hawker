# Contributing

Thanks for considering a contribution to Hawker.

## Development Setup

```bash
uv sync --extra dev
uv run playwright install chromium
```

## Before Opening a Pull Request

Please run:

```bash
uv run ruff check hawker_agent tests run.py
uv run mypy hawker_agent
uv run pytest -q
```

## Contribution Expectations

- Keep pull requests focused.
- Add or update tests when behavior changes.
- Update documentation when user-facing behavior changes.
- Avoid unrelated refactors in the same PR.
- Do not commit secrets, tokens, or private browser profiles.

## Style

The repository uses:

- `ruff` for linting
- `mypy` in strict mode
- `pytest` for tests

When in doubt, prefer clarity over cleverness.
