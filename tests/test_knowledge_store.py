from __future__ import annotations

from hawker_agent.knowledge.store import (
    SiteSOP,
    SiteSOPStore,
    extract_site_keys,
    normalize_page_pattern,
    normalize_site_key,
)


def test_normalize_site_key() -> None:
    assert normalize_site_key("https://www.ArXiv.org/abs/123") == "arxiv.org"
    assert normalize_site_key("openreview.net") == "openreview.net"


def test_extract_site_keys() -> None:
    task = "打开 https://www.arxiv.org/search 并对 openreview.net 做结果对比"
    assert extract_site_keys(task) == ["arxiv.org", "openreview.net"]


def test_normalize_page_pattern() -> None:
    assert normalize_page_pattern("https://github.com/trending?since=daily") == "/trending"
    assert normalize_page_pattern("/trending/") == "/trending"


def test_site_sop_store_upsert_and_find(tmp_path) -> None:
    store = SiteSOPStore(tmp_path / "knowledge.db")
    stored = store.upsert_sop(
        SiteSOP(
            domain="www.arxiv.org",
            page_pattern="/search/cs",
            workflow_kind="api_only",
            should_inspect_first=False,
            preferred_entry="api_direct",
            field_contract=["title", "url"],
            confidence=0.95,
            sop_markdown="# arxiv.org — Scraping & Data Extraction\n\n## Do this first\n...",
            golden_rule="优先直接走公开 API。",
            update_reason="manual_seed",
            source_run_id="run-1",
            proof_summary="已验证搜索接口可返回论文列表",
        )
    )

    assert stored.domain == "arxiv.org"
    assert stored.version == 1
    assert stored.page_pattern == "/search/cs"
    assert stored.should_inspect_first is False

    hit = store.find_for_task("请抓取 https://arxiv.org/search/cs 的结果")
    assert hit is not None
    assert hit.domain == "arxiv.org"
    assert "Scraping & Data Extraction" in hit.sop_markdown


def test_site_sop_store_versions_increment(tmp_path) -> None:
    store = SiteSOPStore(tmp_path / "knowledge.db")
    store.upsert_sop(
        SiteSOP(
            domain="archive.org",
            page_pattern="/details/foo",
            sop_markdown="# archive.org\n\nfirst",
            golden_rule="先看详情页接口。",
            update_reason="seed",
            source_run_id="run-1",
        )
    )
    stored = store.upsert_sop(
        SiteSOP(
            domain="archive.org",
            page_pattern="/details/foo",
            sop_markdown="# archive.org\n\nsecond",
            golden_rule="优先 API，必要时回退 Browser。",
            update_reason="refresh",
            source_run_id="run-2",
        )
    )

    assert stored.version == 2
    assert stored.golden_rule == "优先 API，必要时回退 Browser。"


def test_find_for_task_respects_page_pattern(tmp_path) -> None:
    store = SiteSOPStore(tmp_path / "knowledge.db")
    store.upsert_sop(
        SiteSOP(
            domain="github.com",
            page_pattern="/trending",
            workflow_kind="browser_required",
            should_inspect_first=False,
            preferred_entry="nav_summary_then_extract",
            field_contract=["url", "star"],
            confidence=0.9,
            sop_markdown="# github.com — Scraping & Data Extraction\n\n## Do this first\n...\n\n## Common workflows\n```python\npass\n# Confirmed output (2026-04-20): ok\n```\n\n## Gotchas\n- x",
            golden_rule="直接提取 trending 卡片。",
            update_reason="seed",
            source_run_id="run-1",
        )
    )
    assert store.find_for_task("打开 https://github.com/trending 获取项目列表") is not None
    assert store.find_for_task("打开 https://github.com/browser-use/browser-use 仓库主页") is None
def test_site_sop_store_counts_recent_versions_for_cooldown(tmp_path) -> None:
    store = SiteSOPStore(tmp_path / "knowledge.db")
    store.upsert_sop(
        SiteSOP(
            domain="arxiv.org",
            sop_markdown="# arxiv.org — Scraping & Data Extraction\n\nv1",
            golden_rule="优先 API。",
            update_reason="seed",
            source_run_id="run-1",
        )
    )
    store.upsert_sop(
        SiteSOP(
            domain="arxiv.org",
            sop_markdown="# arxiv.org — Scraping & Data Extraction\n\nv2",
            golden_rule="优先 API。",
            update_reason="refresh",
            source_run_id="run-2",
        )
    )

    assert store.recent_accepted_update_count("arxiv.org", hours=24) == 2
