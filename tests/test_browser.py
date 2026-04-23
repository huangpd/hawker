from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hawker_agent.browser.actions import (
    DomActionResult,
    _build_search_url,
    _log_js_summary,
    browser_download,
)
from hawker_agent.browser.netlog import NETLOG_INJECT_JS
from hawker_agent.models.state import CodeAgentState
from hawker_agent.observability import collect_observations
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
    def test_server_overrides_force_headless_without_display(self) -> None:
        with patch("hawker_agent.browser.session.get_settings") as mock_settings, \
             patch("hawker_agent.browser.session._is_linux_server_without_display", return_value=True), \
             patch("hawker_agent.browser.session._is_root_user", return_value=False):
            mock_settings.return_value = MagicMock(headless=False)
            from hawker_agent.browser.session import BrowserSession

            session = BrowserSession()
            assert session._browser_profile_kwargs["headless"] is True
            assert "--disable-dev-shm-usage" in session._browser_profile_kwargs["args"]

    def test_server_overrides_disable_sandbox_for_root(self) -> None:
        with patch("hawker_agent.browser.session.get_settings") as mock_settings, \
             patch("hawker_agent.browser.session._is_linux_server_without_display", return_value=False), \
             patch("hawker_agent.browser.session._is_root_user", return_value=True):
            mock_settings.return_value = MagicMock(headless=True)
            from hawker_agent.browser.session import BrowserSession

            session = BrowserSession()
            assert session._browser_profile_kwargs["chromium_sandbox"] is False
            assert "--no-sandbox" in session._browser_profile_kwargs["args"]

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

    def test_browser_profile_options_from_settings(self) -> None:
        with patch("hawker_agent.browser.session.get_settings") as mock_settings, \
             patch("hawker_agent.browser.session._server_browser_overrides", return_value={}):
            mock_settings.return_value = MagicMock(
                headless=False,
                browser_executable_path=Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                browser_user_data_dir=Path("/Users/test/Library/Application Support/Google/Chrome"),
                browser_profile_directory="Profile 1",
                browser_storage_state=Path("/tmp/browser-state.json"),
                browser_channel="chrome",
                browser_cdp_url="http://127.0.0.1:9222",
            )
            from hawker_agent.browser.session import BrowserSession

            session = BrowserSession()
            assert session._browser_profile_kwargs == {
                "headless": False,
                "executable_path": Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                "user_data_dir": Path("/Users/test/Library/Application Support/Google/Chrome"),
                "profile_directory": "Profile 1",
                "storage_state": Path("/tmp/browser-state.json"),
                "channel": "chrome",
                "cdp_url": "http://127.0.0.1:9222",
                "auto_download_pdfs": False,
            }

    def test_empty_optional_browser_paths_are_none(self) -> None:
        from hawker_agent.config import Settings

        cfg = Settings(
            openai_api_key="test",
            model_name="test-model",
            browser_executable_path="",
            browser_user_data_dir="",
            browser_storage_state="",
        )

        assert cfg.browser_executable_path is None
        assert cfg.browser_user_data_dir is None
        assert cfg.browser_storage_state is None

    @pytest.mark.asyncio
    async def test_aenter_passes_browser_profile_options(self) -> None:
        with patch("hawker_agent.browser.session.get_settings") as mock_settings, \
             patch("hawker_agent.browser.session._server_browser_overrides", return_value={}), \
             patch("hawker_agent.browser.session.BrowserProfile") as mock_profile_cls, \
             patch("hawker_agent.browser.session._UpstreamBrowserSession") as mock_session_cls:
            mock_settings.return_value = MagicMock(
                headless=False,
                browser_executable_path=Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                browser_user_data_dir=Path("/Users/test/Library/Application Support/Google/Chrome"),
                browser_profile_directory="Profile 1",
                browser_storage_state=Path("/tmp/browser-state.json"),
                browser_channel="chrome",
                browser_cdp_url="http://127.0.0.1:9222",
            )
            mock_profile = MagicMock()
            mock_profile_cls.return_value = mock_profile
            mock_upstream = MagicMock()
            mock_upstream.start = AsyncMock()
            mock_upstream.stop = AsyncMock()
            mock_session_cls.return_value = mock_upstream

            from hawker_agent.browser.session import BrowserSession

            session = BrowserSession()
            await session.__aenter__()

            mock_profile_cls.assert_called_once_with(
                headless=False,
                executable_path=Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                user_data_dir=Path("/Users/test/Library/Application Support/Google/Chrome"),
                profile_directory="Profile 1",
                storage_state=Path("/tmp/browser-state.json"),
                channel="chrome",
                cdp_url="http://127.0.0.1:9222",
                auto_download_pdfs=False,
            )
            mock_session_cls.assert_called_once_with(browser_profile=mock_profile)
            mock_upstream.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_browser_download_timeout_raises() -> None:
    session = MagicMock()
    session.target_dir = None
    session.raw = MagicMock()

    async def _fast_sleep(_: float) -> None:
        return None

    with patch(
        "hawker_agent.browser.actions._download_with_browser_session_http",
        AsyncMock(side_effect=TimeoutError("download timed out")),
    ), patch("hawker_agent.browser.actions.asyncio.sleep", _fast_sleep):
        with pytest.raises(TimeoutError, match="download timed out"):
            await browser_download(session, "https://example.com/file.pdf", timeout_s=0.1, attempts=1)


@pytest.mark.asyncio
async def test_browser_download_uses_session_target_dir(tmp_path: Path) -> None:
    session = MagicMock()
    session.target_dir = tmp_path / "run_dir"
    session.target_dir.mkdir()
    session.raw = MagicMock()

    async def _fast_sleep(_: float) -> None:
        return None

    async def _fake_download(*_args, **_kwargs):
        saved = session.target_dir / "saved.pdf"
        saved.write_bytes(b"%PDF-1.4 fake")
        return {
            "ok": True,
            "url": "https://example.com/file.pdf",
            "filename": "saved.pdf",
            "path": str(saved),
            "size": saved.stat().st_size,
        }

    with patch(
        "hawker_agent.browser.actions._download_with_browser_session_http",
        AsyncMock(side_effect=_fake_download),
    ), patch("hawker_agent.browser.actions.asyncio.sleep", _fast_sleep):
        result = await browser_download(session, "https://example.com/file.pdf", filename="saved.pdf")

    saved = session.target_dir / "saved.pdf"
    assert result["ok"] is True
    assert result["filename"] == "saved.pdf"
    assert Path(result["path"]) == saved
    assert saved.exists()


@pytest.mark.asyncio
async def test_browser_download_reuses_registry_entry(tmp_path: Path) -> None:
    state = CodeAgentState()
    session = MagicMock()
    session.target_dir = tmp_path / "run_dir_registry"
    session.target_dir.mkdir()
    session.raw = MagicMock()
    registry = ToolRegistry()
    history = MagicMock()

    calls = 0

    async def _fake_download(*_args, **kwargs):
        nonlocal calls
        calls += 1
        filename = kwargs.get("filename") or "download.pdf"
        saved = session.target_dir / filename
        saved.write_bytes(b"%PDF-1.4 fake")
        url = _args[1] if len(_args) > 1 else kwargs.get("url", "")
        return {
            "ok": True,
            "url": url,
            "filename": saved.name,
            "path": str(saved),
            "size": saved.stat().st_size,
            "method": "curl_cffi",
        }

    with patch(
        "hawker_agent.tools.browser_tools.actions.browser_download",
        AsyncMock(side_effect=_fake_download),
    ):
        register_browser_tools(registry, session, history, state)
        browser_download_fn = registry.as_namespace_dict()["browser_download"]

        first = await browser_download_fn("https://example.com/file.pdf", filename="first.pdf")
        second = await browser_download_fn("https://example.com/file.pdf", filename="second.pdf")

    assert calls == 1
    assert first["ok"] is True
    assert first["filename"] == "first.pdf"
    assert second["ok"] is True
    assert second["url"] == "https://example.com/file.pdf"
    assert second["path"] == str((session.target_dir / "first.pdf").resolve())
    assert second["size"] == (session.target_dir / "first.pdf").stat().st_size
    assert second["reused"] is True
    assert second["filename"] == "first.pdf"
    assert second["requested_filename"] == "second.pdf"
    assert len(state.download_registry) == 1
    assert state.list_downloaded_files()[0]["path"] == str((session.target_dir / "first.pdf").resolve())


def test_download_registry_ignores_noisy_query_params(tmp_path: Path) -> None:
    state = CodeAgentState()
    saved = tmp_path / "paper.pdf"
    saved.write_bytes(b"%PDF-1.4 fake")

    state.register_download(
        url="https://example.com/paper.pdf?utm_source=test&download=1&token=abc",
        filename="paper.pdf",
        path=str(saved),
        size=saved.stat().st_size,
        method="curl_cffi",
    )

    hit = state.get_download_record("https://example.com/paper.pdf?token=xyz&utm_campaign=demo")
    assert hit is not None
    assert hit["filename"] == "paper.pdf"
    assert hit["path"] == str(saved.resolve())


@pytest.mark.asyncio
async def test_browser_download_retries_once_then_succeeds(tmp_path: Path) -> None:
    session = MagicMock()
    session.target_dir = tmp_path / "run_dir_retry"
    session.target_dir.mkdir()
    session.raw = MagicMock()

    async def _fast_sleep(_: float) -> None:
        return None

    attempts_seen = 0

    async def _fake_download(*_args, **_kwargs):
        nonlocal attempts_seen
        attempts_seen += 1
        if attempts_seen == 1:
            raise TimeoutError("first try")
        saved = session.target_dir / "retry.pdf"
        saved.write_bytes(b"%PDF-1.4 fake")
        return {
            "ok": True,
            "url": "https://example.com/retry.pdf",
            "filename": "retry.pdf",
            "path": str(saved),
            "size": saved.stat().st_size,
        }

    with patch(
        "hawker_agent.browser.actions._download_with_browser_session_http",
        AsyncMock(side_effect=_fake_download),
    ), patch("hawker_agent.browser.actions.asyncio.sleep", _fast_sleep):
        result = await browser_download(
            session,
            "https://example.com/retry.pdf",
            filename="retry.pdf",
            timeout_s=0.1,
            attempts=2,
            retry_delay_s=0,
        )

    saved = session.target_dir / "retry.pdf"
    assert result["ok"] is True
    assert result["filename"] == "retry.pdf"
    assert saved.exists()


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
    def test_error_result(self) -> None:
        with collect_observations() as observations:
            _log_js_summary("[JS错误] some error")
        assert "[JS错误]" in "\n".join(observations)

    def test_list_result(self) -> None:
        data = [{"id": 1}, {"id": 2}]
        with collect_observations() as observations:
            _log_js_summary(data)
        assert "2 条数据" in "\n".join(observations)

    def test_dict_result(self) -> None:
        data = {"key": "value", "count": 42}
        with collect_observations() as observations:
            _log_js_summary(data)
        text = "\n".join(observations)
        assert "dict" in text
        assert "2 个键" in text

    def test_scalar_result(self) -> None:
        with collect_observations() as observations:
            _log_js_summary('"hello"')
        assert "hello" in "\n".join(observations)

    def test_plain_text_result(self) -> None:
        with collect_observations() as observations:
            _log_js_summary("not json at all")
        assert "not json at all" in "\n".join(observations)

    def test_empty_list(self) -> None:
        with collect_observations() as observations:
            _log_js_summary([])
        assert "0 条数据" in "\n".join(observations)


# ─── register_browser_tools ──────────────────────────────────


class TestBrowserToolsRegistration:
    def test_registers_all_tools(self) -> None:
        registry = ToolRegistry()
        mock_session = MagicMock()
        mock_history = MagicMock()
        register_browser_tools(registry, mock_session, mock_history)
        expected = {
            "nav", "dom_state", "nav_search", "inspect_page", "js", "click",
            "click_index", "fill_input", "browser_download", "get_network_log",
            "get_selector_from_index", "get_cookies", "list_downloaded_files",
        }
        for name in expected:
            assert name in registry, f"工具 {name} 未注册"
        assert len(registry) == 13

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
        assert "侦察" in desc

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
        assert param_names == ["selector", "index", "mode"]

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
        assert list(sig.parameters.keys()) == ["mode"]
