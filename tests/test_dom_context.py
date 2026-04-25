from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hawker_agent.browser.actions import DomActionResult
from hawker_agent.browser.dom_utils import (
    build_dom_snapshot,
    render_dom_diff,
    render_dom_summary,
)
from hawker_agent.models.state import CodeAgentState
from hawker_agent.tools.browser_tools import register_browser_tools
from hawker_agent.tools.registry import ToolRegistry


class TestDomSnapshot:
    def test_build_dom_snapshot_extracts_interactives_and_regions(self) -> None:
        dom_repr = """
        <main>
          [i_12]<button>搜索</button>
          [i_18]<a href="/next">下一页</a>
          <dialog>筛选</dialog>
        </main>
        """
        snapshot = build_dom_snapshot(
            title="列表页",
            url="https://example.com/list",
            dom_repr=dom_repr,
            pages_above=0.0,
            pages_below=1.5,
            pending_requests=2,
            tabs=1,
        )
        assert snapshot["title"] == "列表页"
        assert snapshot["interactive_count"] == 2
        assert any("[i_12]" in item for item in snapshot["interactive_preview"])
        assert "main" in snapshot["regions"]
        assert snapshot["pending_requests"] == 2

    def test_render_dom_summary_and_diff(self) -> None:
        previous = build_dom_snapshot(
            title="列表页",
            url="https://example.com/list",
            dom_repr="[i_12]<button>搜索</button>\n<main></main>",
        )
        current = build_dom_snapshot(
            title="列表页",
            url="https://example.com/list",
            dom_repr="[i_12]<button>搜索</button>\n[i_88]<button>确认</button>\n<main></main>\n<dialog></dialog>",
        )
        summary = render_dom_summary(current)
        diff = render_dom_diff(previous, current)
        assert "[DOM Summary]" in summary
        assert "交互元素" in summary
        assert "[DOM Diff]" in diff
        assert "新增区域" in diff or "新增交互示例" in diff


@pytest.mark.asyncio
class TestBrowserToolModes:
    async def test_auto_nav_uses_summary_and_updates_snapshot(self) -> None:
        registry = ToolRegistry()
        history = MagicMock()
        session = MagicMock()
        state = CodeAgentState()

        register_browser_tools(registry, session, history, state)
        nav = registry.as_namespace_dict()["nav"]

        async def fake_nav(*args, **kwargs):
            assert kwargs["mode"] == "summary"
            assert kwargs["previous_snapshot"] is None
            return DomActionResult(
                summary="[OK] 列表页 | 交互元素 2 | DOM=summary",
                dom=None,
                snapshot={"title": "列表页", "interactive_count": 2},
            )

        from hawker_agent.tools import browser_tools as browser_tools_module

        original = browser_tools_module.actions.nav
        browser_tools_module.actions.nav = fake_nav
        try:
            result = await nav("https://example.com")
        finally:
            browser_tools_module.actions.nav = original

        assert "DOM=summary" in result
        assert state.last_dom_snapshot == {"title": "列表页", "interactive_count": 2}
        history.inject_browser_context.assert_not_called()

    async def test_nav_search_defaults_to_summary(self) -> None:
        registry = ToolRegistry()
        history = MagicMock()
        session = MagicMock()
        state = CodeAgentState()

        register_browser_tools(registry, session, history, state)
        nav_search = registry.as_namespace_dict()["nav_search"]

        async def fake_nav_search(*args, **kwargs):
            assert kwargs["mode"] == "summary"
            return DomActionResult(
                summary="[OK] 搜索页 | 交互元素 6 | DOM=summary",
                dom=None,
                snapshot={"title": "搜索页", "interactive_count": 6},
            )

        from hawker_agent.tools import browser_tools as browser_tools_module

        original = browser_tools_module.actions.nav_search
        browser_tools_module.actions.nav_search = fake_nav_search
        try:
            result = await nav_search("web agent", engine="google")
        finally:
            browser_tools_module.actions.nav_search = original

        assert "DOM=summary" in result
        history.inject_browser_context.assert_not_called()

    async def test_dom_state_auto_uses_full_without_previous_snapshot(self) -> None:
        registry = ToolRegistry()
        history = MagicMock()
        session = MagicMock()
        state = CodeAgentState()

        register_browser_tools(registry, session, history, state)
        dom_state = registry.as_namespace_dict()["dom_state"]

        async def fake_dom_state(*args, **kwargs):
            assert kwargs["mode"] == "full"
            assert kwargs["previous_snapshot"] is None
            return DomActionResult(
                summary="[OK] 新页 | 交互元素 3 | DOM=full",
                dom="<html>...</html>",
                snapshot={"title": "新页", "interactive_count": 3},
            )

        from hawker_agent.tools import browser_tools as browser_tools_module

        original = browser_tools_module.actions.dom_state
        browser_tools_module.actions.dom_state = fake_dom_state
        try:
            result = await dom_state()
        finally:
            browser_tools_module.actions.dom_state = original

        assert "DOM=full" in result
        history.inject_browser_context.assert_called_once()
        assert state.last_dom_snapshot == {"title": "新页", "interactive_count": 3}

    async def test_auto_click_uses_diff_with_previous_snapshot(self) -> None:
        registry = ToolRegistry()
        history = MagicMock()
        session = MagicMock()
        state = CodeAgentState(last_dom_snapshot={"title": "旧页"})

        register_browser_tools(registry, session, history, state)
        click = registry.as_namespace_dict()["click"]

        async def fake_click(*args, **kwargs):
            assert kwargs["mode"] == "diff"
            assert kwargs["previous_snapshot"] == {"title": "旧页"}
            return DomActionResult(
                summary="[OK] 新页 | 交互元素 3 | DOM=diff",
                dom="[DOM Diff]\n- URL 未变化",
                snapshot={"title": "新页", "interactive_count": 3},
            )

        from hawker_agent.tools import browser_tools as browser_tools_module

        original = browser_tools_module.actions.click
        browser_tools_module.actions.click = fake_click
        try:
            result = await click(".next")
        finally:
            browser_tools_module.actions.click = original

        assert "DOM=diff" in result
        history.inject_browser_context.assert_called_once()
        assert state.last_dom_snapshot == {"title": "新页", "interactive_count": 3}

    async def test_auto_click_upgrades_to_full_after_no_progress(self) -> None:
        registry = ToolRegistry()
        history = MagicMock()
        session = MagicMock()
        state = CodeAgentState(last_dom_snapshot={"title": "旧页"}, no_progress_streak=2)

        register_browser_tools(registry, session, history, state)
        click_index = registry.as_namespace_dict()["click_index"]

        async def fake_click_index(*args, **kwargs):
            assert kwargs["mode"] == "full"
            return DomActionResult(
                summary="[OK] 新页 | 交互元素 3 | DOM=full",
                dom="<html>...</html>",
                snapshot={"title": "新页", "interactive_count": 3},
            )

        from hawker_agent.tools import browser_tools as browser_tools_module

        original = browser_tools_module.actions.click_index
        browser_tools_module.actions.click_index = fake_click_index
        try:
            result = await click_index(12)
        finally:
            browser_tools_module.actions.click_index = original

        assert "DOM=full" in result
        history.inject_browser_context.assert_called_once()

    async def test_auto_click_failure_triggers_diagnostic_dom(self) -> None:
        registry = ToolRegistry()
        history = MagicMock()
        session = MagicMock()
        state = CodeAgentState(last_dom_snapshot={"title": "旧页"})

        register_browser_tools(registry, session, history, state)
        click = registry.as_namespace_dict()["click"]

        async def fake_click(*args, **kwargs):
            return DomActionResult(summary="[失败] 未找到元素")

        async def fake_dom_state(*args, **kwargs):
            assert kwargs["mode"] == "diff"
            return DomActionResult(
                summary="[OK] 旧页 | 交互元素 2 | DOM=diff",
                dom="[DOM Diff]\n- 交互元素数不变: 2",
                snapshot={"title": "旧页", "interactive_count": 2},
            )

        from hawker_agent.tools import browser_tools as browser_tools_module

        original_click = browser_tools_module.actions.click
        original_dom_state = browser_tools_module.actions.dom_state
        browser_tools_module.actions.click = fake_click
        browser_tools_module.actions.dom_state = fake_dom_state
        try:
            result = await click(".missing")
        finally:
            browser_tools_module.actions.click = original_click
            browser_tools_module.actions.dom_state = original_dom_state

        assert "已自动补充 DOM=diff" in result
        assert history.inject_browser_context.call_count == 1
        assert state.last_dom_snapshot == {"title": "旧页", "interactive_count": 2}

    async def test_inspect_page_combines_dom_cookies_and_selector(self) -> None:
        registry = ToolRegistry()
        history = MagicMock()
        session = MagicMock()
        raw_session = MagicMock()
        session.raw = raw_session
        state = CodeAgentState()

        register_browser_tools(registry, session, history, state)
        inspect_page = registry.as_namespace_dict()["inspect_page"]

        async def fake_dom_state(*args, **kwargs):
            assert kwargs["mode"] == "summary"
            return DomActionResult(
                summary="[OK] 结果页 | 交互元素 4 | DOM=summary",
                dom=None,
                snapshot={"title": "结果页", "interactive_count": 4},
            )

        async def fake_get_cookies(*args, **kwargs):
            return [{"name": "sid", "value": "abc"}]

        async def fake_get_selector(*args, **kwargs):
            return {"selector": ".item", "shadow_path": []}

        from hawker_agent.tools import browser_tools as browser_tools_module

        original_dom_state = browser_tools_module.actions.dom_state
        original_get_cookies = browser_tools_module.actions.get_cookies
        browser_tools_module.actions.dom_state = fake_dom_state
        browser_tools_module.actions.get_cookies = fake_get_cookies

        from hawker_agent.browser import dom_utils as dom_utils_module
        original_get_selector = dom_utils_module.get_selector_from_index
        dom_utils_module.get_selector_from_index = fake_get_selector
        try:
            result = await inspect_page(dom=True, cookies=True, selector_index=12)
        finally:
            browser_tools_module.actions.dom_state = original_dom_state
            browser_tools_module.actions.get_cookies = original_get_cookies
            dom_utils_module.get_selector_from_index = original_get_selector

        assert result["dom"]["summary"].endswith("DOM=summary")
        assert "network" not in result
        assert result["cookies"] == [{"name": "sid", "value": "abc"}]
        assert result["selector"]["selector"] == ".item"
        assert "js_snippet" in result["selector"]

    async def test_sop_guard_downgrades_explicit_full_dom_state(self) -> None:
        registry = ToolRegistry()
        history = MagicMock()
        session = MagicMock()
        state = CodeAgentState(sop_guided_dom_steps_remaining=2, sop_guided_reason="先按 SOP 低成本验证")

        register_browser_tools(registry, session, history, state)
        dom_state = registry.as_namespace_dict()["dom_state"]

        async def fake_dom_state(*args, **kwargs):
            assert kwargs["mode"] == "summary"
            return DomActionResult(
                summary="[OK] 页面 | 交互元素 3 | DOM=summary",
                dom=None,
                snapshot={"title": "页面", "interactive_count": 3},
            )

        from hawker_agent.tools import browser_tools as browser_tools_module

        original = browser_tools_module.actions.dom_state
        browser_tools_module.actions.dom_state = fake_dom_state
        try:
            result = await dom_state(mode="full")
        finally:
            browser_tools_module.actions.dom_state = original

        assert "DOM=summary" in result

    async def test_sop_guard_downgrades_explicit_full_nav(self) -> None:
        registry = ToolRegistry()
        history = MagicMock()
        session = MagicMock()
        state = CodeAgentState(sop_guided_dom_steps_remaining=1, sop_guided_reason="该站点优先 SSR 直提")

        register_browser_tools(registry, session, history, state)
        nav = registry.as_namespace_dict()["nav"]

        async def fake_nav(*args, **kwargs):
            assert kwargs["mode"] == "skip"
            return DomActionResult(
                summary="[OK] 页面 | DOM=skipped",
                dom=None,
                snapshot={"title": "页面", "interactive_count": 0},
            )

        from hawker_agent.tools import browser_tools as browser_tools_module

        original = browser_tools_module.actions.nav
        browser_tools_module.actions.nav = fake_nav
        try:
            result = await nav("https://example.com", mode="full")
        finally:
            browser_tools_module.actions.nav = original

        assert "DOM=skipped" in result
