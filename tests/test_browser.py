from __future__ import annotations

import inspect
import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from hawker_agent.browser.actions import (
    DomActionResult,
    _build_search_url,
    _log_js_summary,
)
from hawker_agent.browser.netlog import NETLOG_INJECT_JS
from hawker_agent.tools.browser_tools import register_browser_tools
from hawker_agent.tools.registry import ToolRegistry


# ─── NETLOG_INJECT_JS ─────────────────────────────────────────


class TestNetlogJS:
    def test_is_nonempty_string(self) -> None:
        assert isinstance(NETLOG_INJECT_JS, str)
        assert len(NETLOG_INJECT_JS) > 100

    def test_contains_patched_flag(self) -> None:
        assert "window.__netlog_patched" in NETLOG_INJECT_JS

    def test_contains_max_entries(self) -> None:
        assert "MAX=50" in NETLOG_INJECT_JS

    def test_contains_fetch_patch(self) -> None:
        assert "window.fetch=function" in NETLOG_INJECT_JS

    def test_contains_xhr_patch(self) -> None:
        assert "XMLHttpRequest.prototype.open" in NETLOG_INJECT_JS

    def test_filters_static_assets(self) -> None:
        assert "png" in NETLOG_INJECT_JS
        assert "css" in NETLOG_INJECT_JS
        assert "woff2" in NETLOG_INJECT_JS

    def test_filters_analytics(self) -> None:
        assert "google-analytics.com" in NETLOG_INJECT_JS
        assert "sentry.io" in NETLOG_INJECT_JS

    def test_filters_framework_hotreload(self) -> None:
        assert "/_nuxt/builds/" in NETLOG_INJECT_JS
        assert "/__webpack" in NETLOG_INJECT_JS
        assert "/hot-update." in NETLOG_INJECT_JS


# ─── BrowserSession ───────────────────────────────────────────


class TestBrowserSession:
    def test_raw_raises_before_start(self) -> None:
        with patch("hawker_agent.browser.session.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(headless=True)
            from hawker_agent.browser.session import BrowserSession

            session = BrowserSession()
            with pytest.raises(RuntimeError, match="未启动"):
                _ = session.raw

    def test_default_attributes(self) -> None:
        with patch("hawker_agent.browser.session.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(headless=False)
            from hawker_agent.browser.session import BrowserSession

            session = BrowserSession()
            assert session.netlog_installed is False
            assert session.netlog_cursor == 0

    def test_headless_from_settings(self) -> None:
        with patch("hawker_agent.browser.session.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(headless=True)
            from hawker_agent.browser.session import BrowserSession

            session = BrowserSession()
            assert session._headless is True

    def test_headless_override(self) -> None:
        with patch("hawker_agent.browser.session.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(headless=False)
            from hawker_agent.browser.session import BrowserSession

            session = BrowserSession(headless=True)
            assert session._headless is True


# ─── DomActionResult ──────────────────────────────────────────


class TestDomActionResult:
    def test_with_dom(self) -> None:
        result = DomActionResult(summary="[OK] Test", dom="<html>full dom</html>")
        assert result.summary == "[OK] Test"
        assert result.dom == "<html>full dom</html>"

    def test_without_dom(self) -> None:
        result = DomActionResult(summary="[OK] Test")
        assert result.dom is None

    def test_is_dataclass(self) -> None:
        import dataclasses

        assert dataclasses.is_dataclass(DomActionResult)


# ─── _build_search_url ───────────────────────────────────────


class TestBuildSearchUrl:
    def test_duckduckgo(self) -> None:
        url = _build_search_url("hello world", "duckduckgo")
        assert url == "https://duckduckgo.com/?q=hello+world"

    def test_google(self) -> None:
        url = _build_search_url("hello world", "google")
        assert url is not None
        assert "google.com/search" in url
        assert "q=hello+world" in url

    def test_bing(self) -> None:
        url = _build_search_url("hello world", "bing")
        assert url is not None
        assert "bing.com/search" in url

    def test_unsupported_engine(self) -> None:
        url = _build_search_url("hello", "yahoo")
        assert url is None

    def test_case_insensitive(self) -> None:
        url = _build_search_url("test", "DuckDuckGo")
        assert url is not None
        assert "duckduckgo.com" in url

    def test_special_chars_encoded(self) -> None:
        url = _build_search_url("hello & goodbye", "duckduckgo")
        assert url is not None
        assert "hello+%26+goodbye" in url


# ─── _log_js_summary ─────────────────────────────────────────


class TestLogJsSummary:
    def test_error_result(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="hawker_agent.browser.actions"):
            _log_js_summary("[JS错误] some error")
        assert "[JS错误]" in caplog.text

    def test_list_result(self, caplog: pytest.LogCaptureFixture) -> None:
        data = json.dumps([{"id": 1}, {"id": 2}])
        with caplog.at_level(logging.INFO, logger="hawker_agent.browser.actions"):
            _log_js_summary(data)
        assert "2 条数据" in caplog.text

    def test_dict_result(self, caplog: pytest.LogCaptureFixture) -> None:
        data = json.dumps({"key": "value", "count": 42})
        with caplog.at_level(logging.INFO, logger="hawker_agent.browser.actions"):
            _log_js_summary(data)
        assert "dict" in caplog.text
        assert "2 个键" in caplog.text

    def test_scalar_result(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="hawker_agent.browser.actions"):
            _log_js_summary('"hello"')
        assert "hello" in caplog.text

    def test_plain_text_result(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="hawker_agent.browser.actions"):
            _log_js_summary("not json at all")
        assert "15 字符" in caplog.text

    def test_empty_list(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="hawker_agent.browser.actions"):
            _log_js_summary("[]")
        assert "0 条数据" in caplog.text


# ─── register_browser_tools ──────────────────────────────────


class TestBrowserToolsRegistration:
    def test_registers_all_tools(self) -> None:
        registry = ToolRegistry()
        mock_session = MagicMock()
        mock_history = MagicMock()
        register_browser_tools(registry, mock_session, mock_history)
        expected = {"nav", "dom_state", "nav_search", "js", "click",
                    "click_index", "fill_input", "get_network_log"}
        for name in expected:
            assert name in registry, f"工具 {name} 未注册"
        assert len(registry) == 8

    def test_tool_descriptions_in_chinese(self) -> None:
        registry = ToolRegistry()
        mock_session = MagicMock()
        mock_history = MagicMock()
        register_browser_tools(registry, mock_session, mock_history)
        desc = registry.build_description()
        # 每个工具都应有中文描述
        assert "导航" in desc
        assert "DOM" in desc
        assert "搜索" in desc
        assert "JavaScript" in desc
        assert "点击" in desc
        assert "输入" in desc
        assert "网络请求" in desc

    def test_nav_signature_no_session(self) -> None:
        registry = ToolRegistry()
        mock_session = MagicMock()
        mock_history = MagicMock()
        register_browser_tools(registry, mock_session, mock_history)
        ns = registry.as_namespace_dict()
        sig = inspect.signature(ns["nav"])
        param_names = list(sig.parameters.keys())
        assert "session" not in param_names
        assert "url" in param_names

    def test_click_signature(self) -> None:
        registry = ToolRegistry()
        mock_session = MagicMock()
        mock_history = MagicMock()
        register_browser_tools(registry, mock_session, mock_history)
        ns = registry.as_namespace_dict()
        sig = inspect.signature(ns["click"])
        param_names = list(sig.parameters.keys())
        assert param_names == ["selector", "index"]

    def test_fill_input_signature(self) -> None:
        registry = ToolRegistry()
        mock_session = MagicMock()
        mock_history = MagicMock()
        register_browser_tools(registry, mock_session, mock_history)
        ns = registry.as_namespace_dict()
        sig = inspect.signature(ns["fill_input"])
        param_names = list(sig.parameters.keys())
        assert param_names == ["index", "text"]

    def test_get_network_log_signature(self) -> None:
        registry = ToolRegistry()
        mock_session = MagicMock()
        mock_history = MagicMock()
        register_browser_tools(registry, mock_session, mock_history)
        ns = registry.as_namespace_dict()
        sig = inspect.signature(ns["get_network_log"])
        param_names = list(sig.parameters.keys())
        assert "filter" in param_names
        assert "only_new" in param_names
        assert "session" not in param_names

    def test_dom_state_no_params(self) -> None:
        registry = ToolRegistry()
        mock_session = MagicMock()
        mock_history = MagicMock()
        register_browser_tools(registry, mock_session, mock_history)
        ns = registry.as_namespace_dict()
        sig = inspect.signature(ns["dom_state"])
        assert len(sig.parameters) == 0
