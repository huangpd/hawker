from __future__ import annotations

from hawker_agent.models.cell import CellStatus, CodeCell
from hawker_agent.models.state import TokenStats
from hawker_agent.knowledge.observer import (
    build_data_access_summary,
    build_execution_log,
    classify_observer_evidence,
    extract_requested_fields,
    extract_golden_rule,
    infer_observer_domain,
    infer_page_pattern,
    infer_preferred_entry,
    infer_should_inspect_first,
    load_observer_examples,
    select_observer_examples,
    smart_merge_sop,
    validate_browser_harness_style_sop,
)


def _cell(
    *,
    step: int,
    status: CellStatus,
    source: str,
    output: str | None = None,
    error: str | None = None,
    url: str = "",
) -> CodeCell:
    return CodeCell(
        step=step,
        thought="检查页面并提取数据",
        source=source,
        output=output,
        error=error,
        status=status,
        duration=0.1,
        usage=TokenStats(),
        url=url,
        items_count=1 if status == CellStatus.SUCCESS else 0,
    )


def test_build_execution_log_prefers_success_and_key_failures() -> None:
    cells = [
        _cell(step=1, status=CellStatus.ERROR, source="bad()", error="[执行错误] HTTP 404"),
        _cell(step=2, status=CellStatus.ERROR, source="oops()", error="[执行错误] SyntaxError: invalid syntax"),
        _cell(step=3, status=CellStatus.SUCCESS, source="items = http_json('/api')", output="提取 3 条", url="https://arxiv.org"),
    ]
    text = build_execution_log(cells, [{"title": "A"}])
    assert "HTTP 404" in text
    assert "SyntaxError" not in text
    assert "http_json('/api')" in text
    assert "Confirmed data sample" in text


def test_build_data_access_summary_formats_explicit_tools() -> None:
    cells = [
        _cell(
            step=1,
            status=CellStatus.SUCCESS,
            source='data = await fetch("https://example.com/api/search?q=demo", parse="json")',
            output="提取 3 条",
        )
    ]
    summary = build_data_access_summary(cells, [{"id": 1}])
    assert "fetch" in summary
    assert "explicit_url=https://example.com/api/search?q=demo" in summary
    assert "confirmed_items=1" in summary


def test_extract_golden_rule_from_markdown() -> None:
    markdown = "# arxiv.org\n\n**Golden Rule:** 优先公开 API，必要时回退 Browser。"
    assert extract_golden_rule(markdown) == "优先公开 API，必要时回退 Browser。"


def test_load_observer_examples() -> None:
    examples = load_observer_examples()
    assert set(examples) == {"api_only", "hybrid", "browser_required"}
    assert "Scraping & Data Extraction" in examples["api_only"].content


def test_classify_observer_evidence_api_only() -> None:
    kind = classify_observer_evidence(
        "items = http_json('https://example.com/api')\nprint(items)",
        "- step=1 tools=http_json explicit_url=https://example.com/api output=application/json",
    )
    assert kind == "api_only"


def test_classify_observer_evidence_hybrid() -> None:
    kind = classify_observer_evidence(
        "await nav('https://example.com')\nawait click('button')\nitems = http_json('https://example.com/api')",
        "- step=2 tools=http_json explicit_url=https://example.com/api",
    )
    assert kind == "hybrid"


def test_select_observer_examples_prefers_primary_then_fallback() -> None:
    examples = select_observer_examples(
        "await nav('https://example.com/listings')\nawait js('return 1')",
        "No explicit data access evidence.",
    )
    assert examples[0].key == "browser_required"
    assert len(examples) == 2


def test_extract_requested_fields() -> None:
    task = "提取字段:\n- URL: 项目链接\n- start: start数\n- fork： fork数"
    assert extract_requested_fields(task) == ["URL", "start", "fork"]


def test_infer_page_pattern_and_execution_policy() -> None:
    assert infer_page_pattern("打开 https://github.com/trending", "") == "/trending"
    assert infer_preferred_entry("browser_required", "/trending") == "nav_summary_then_extract"
    assert infer_should_inspect_first("browser_required", "/trending", "await js('x')\nConfirmed data sample: ...") is False


def test_infer_observer_domain_prefers_task_then_cells() -> None:
    assert infer_observer_domain("打开 https://github.com/trending", []) == "github.com"
    cells = [_cell(step=1, status=CellStatus.SUCCESS, source="pass", output="ok", url="https://openreview.net/group?id=ICLR")]
    assert infer_observer_domain("任务里没域名", cells) == "openreview.net"


def test_infer_observer_domain_reads_url_from_success_code() -> None:
    cells = [
        _cell(
            step=1,
            status=CellStatus.SUCCESS,
            source='page = await nav("https://github.com/trending", mode="summary")',
            output="ok",
        )
    ]
    assert infer_observer_domain("抓取 GitHub Trending 前 10 个仓库", cells) == "github.com"


def test_validate_browser_harness_style_sop_accepts_valid_markdown() -> None:
    markdown = """# arxiv.org — Scraping & Data Extraction

`https://arxiv.org` — 开放论文站点。**Never use the browser for ArXiv.** 优先公开 API。

## Do this first

先确认是否已有可直接消费的 JSON 返回。

## Common workflows

```python
data = http_json("https://arxiv.org/api?q=llm")
# Confirmed output (2026-04-20): {"items":[{"id":"1"}]}
```

## Gotchas

- 不要先请求 full DOM。
"""
    result = validate_browser_harness_style_sop(markdown, "arxiv.org")
    assert result.ok is True


def test_validate_browser_harness_style_sop_rejects_missing_sections() -> None:
    markdown = """# arxiv.org — Scraping & Data Extraction

`https://arxiv.org` — 开放论文站点。

## Do this first

先看接口。
"""
    result = validate_browser_harness_style_sop(markdown, "arxiv.org")
    assert result.ok is False
    assert "missing required section" in result.reason


def test_validate_browser_harness_style_sop_rejects_secret_like_material() -> None:
    markdown = """# arxiv.org — Scraping & Data Extraction

`https://arxiv.org` — 开放论文站点。优先 API。

## Do this first

先看接口。

## Common workflows

```python
headers = {"Authorization": "Bearer secret-token-value"}
# Confirmed output (2026-04-20): {"ok": true}
```

## Gotchas

- 注意登录态。
"""
    result = validate_browser_harness_style_sop(markdown, "arxiv.org")
    assert result.ok is False
    assert "sensitive" in result.reason


def test_smart_merge_sop_keeps_old_api_reference_when_new_is_placeholder() -> None:
    existing = """# arxiv.org — Scraping & Data Extraction

`https://arxiv.org` — 开放论文站点。优先 API。

## Do this first

先看接口。

## Common workflows

```python
data = http_json("https://arxiv.org/api")
# Confirmed output (2026-04-20): {"items":[{"id":"1"}]}
```

## API reference

| Field | Meaning |
| --- | --- |
| `q` | query |

## Gotchas

- 不要先请求 full DOM。
"""
    candidate = """# arxiv.org — Scraping & Data Extraction

`https://arxiv.org` — 开放论文站点。优先 API，必要时回退 Browser。

## Do this first

先看接口。

## Common workflows

```python
data = http_json("https://arxiv.org/api/v2")
# Confirmed output (2026-04-21): {"items":[{"id":"2"}]}
```

## API reference

暂无稳定 API 词典。

## Gotchas

- 翻页参数 page 从 0 开始。
"""
    merged = smart_merge_sop(existing, candidate)
    assert "| `q` | query |" in merged
    assert "- 翻页参数 page 从 0 开始。" in merged
    assert "- 不要先请求 full DOM。" in merged
