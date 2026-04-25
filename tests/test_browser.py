from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from hawker_agent.browser.actions import (
    DomActionResult,
    _build_search_url,
    _log_js_summary,
    browser_download,
    nav as action_nav,
)
from hawker_agent.agent.namespace import register_core_actions
from hawker_agent.models.state import CodeAgentState
from hawker_agent.models.history import CodeAgentHistoryList
from hawker_agent.observability import collect_observations
from hawker_agent.tools.browser_tools import register_browser_tools
from hawker_agent.tools.registry import ToolRegistry


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
            assert session.target_dir is None
            assert not hasattr(session, "netlog_installed")
            assert not hasattr(session, "netlog_cursor")

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
                browser_proxy=None,
                browser_timezone_id=None,
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
                "proxy": None,
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
            browser_proxy="",
            browser_timezone_id="",
        )

        assert cfg.browser_executable_path is None
        assert cfg.browser_user_data_dir is None
        assert cfg.browser_storage_state is None
        assert cfg.browser_proxy is None
        assert cfg.browser_timezone_id is None

    def test_empty_cloud_browser_settings_are_normalized(self) -> None:
        from hawker_agent.config import Settings

        cfg = Settings(
            openai_api_key="test",
            model_name="test-model",
            browser_provider="",
            browser_use_api_key="",
            browser_use_base_url="",
            browser_use_profile_id="",
            browser_use_proxy_country_code="",
        )

        assert cfg.browser_provider == "local"
        assert cfg.browser_use_api_key is None
        assert cfg.browser_use_base_url is None
        assert cfg.browser_use_profile_id is None
        assert cfg.browser_use_proxy_country_code is None

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
                browser_proxy="http://user:pass@proxy.example:8080",
                browser_timezone_id="Asia/Shanghai",
            )
            mock_profile = MagicMock()
            mock_profile_cls.return_value = mock_profile
            mock_upstream = MagicMock()
            mock_upstream.start = AsyncMock()
            mock_upstream.stop = AsyncMock()
            mock_cdp_session = MagicMock()
            mock_cdp_session.session_id = "session-1"
            mock_cdp_session.cdp_client = MagicMock()
            mock_cdp_session.cdp_client.send = MagicMock()
            mock_cdp_session.cdp_client.send.Emulation = MagicMock()
            mock_cdp_session.cdp_client.send.Emulation.setTimezoneOverride = AsyncMock()
            mock_upstream.get_or_create_cdp_session = AsyncMock(return_value=mock_cdp_session)
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
                proxy=ANY,
                auto_download_pdfs=False,
            )
            mock_session_cls.assert_called_once_with(browser_profile=mock_profile)
            mock_upstream.start.assert_awaited_once()
            mock_upstream.get_or_create_cdp_session.assert_awaited_once_with(target_id=None, focus=False)
            mock_cdp_session.cdp_client.send.Emulation.setTimezoneOverride.assert_awaited_once_with(
                params={"timezoneId": "Asia/Shanghai"},
                session_id="session-1",
            )

    @pytest.mark.asyncio
    async def test_aenter_bootstraps_browser_use_cloud_session(self) -> None:
        with patch("hawker_agent.browser.session.get_settings") as mock_settings, \
             patch("hawker_agent.browser.session._server_browser_overrides", return_value={}), \
             patch("hawker_agent.browser.session.httpx.AsyncClient") as mock_http_client_cls, \
             patch("hawker_agent.browser.session.BrowserProfile") as mock_profile_cls, \
             patch("hawker_agent.browser.session._UpstreamBrowserSession") as mock_session_cls:
            mock_settings.return_value = MagicMock(
                headless=False,
                browser_provider="browser_use_cloud",
                browser_executable_path=None,
                browser_user_data_dir=None,
                browser_profile_directory="Default",
                browser_storage_state=None,
                browser_channel=None,
                browser_cdp_url=None,
                browser_use_api_key="cloud-key",
                browser_use_base_url="https://api.browser-use.local",
                browser_use_profile_id="profile-123",
                browser_use_proxy_country_code="us",
                browser_use_enable_recording=True,
            )

            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "id": "sess_123",
                "cdpUrl": "wss://cdp.example/ws",
                "liveUrl": "https://live.example",
            }
            mock_http_client = MagicMock()
            mock_http_client.post = AsyncMock(return_value=mock_response)
            mock_http_client.patch = AsyncMock(return_value=MagicMock())
            mock_http_client.aclose = AsyncMock()
            mock_http_client_cls.return_value = mock_http_client

            mock_profile = MagicMock()
            mock_profile_cls.return_value = mock_profile
            mock_upstream = MagicMock()
            mock_upstream.start = AsyncMock()
            mock_upstream.stop = AsyncMock()
            mock_session_cls.return_value = mock_upstream

            from hawker_agent.browser.session import BrowserSession

            session = BrowserSession()
            await session.__aenter__()
            await session.__aexit__(None, None, None)

            mock_http_client_cls.assert_called_once_with(
                base_url="https://api.browser-use.local",
                headers={"X-Browser-Use-API-Key": "cloud-key"},
                timeout=30.0,
            )
            mock_http_client.post.assert_awaited_once_with(
                "/browsers",
                json={
                    "profileId": "profile-123",
                    "proxyCountryCode": "us",
                    "enableRecording": True,
                },
            )
            mock_profile_cls.assert_called_once_with(
                headless=False,
                profile_directory="Default",
                cdp_url="wss://cdp.example/ws",
                auto_download_pdfs=False,
            )
            mock_http_client.patch.assert_awaited_once_with(
                "/browsers/sess_123",
                json={"action": "stop"},
            )
            mock_http_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aexit_cleans_temp_browser_use_profile_dir(self, tmp_path: Path) -> None:
        temp_profile_dir = tmp_path / "browser-use-user-data-dir-test"
        temp_profile_dir.mkdir()
        (temp_profile_dir / "marker.txt").write_text("x", encoding="utf-8")

        with patch("hawker_agent.browser.session.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(headless=True)
            from hawker_agent.browser.session import BrowserSession

            session = BrowserSession()
            mock_upstream = MagicMock()
            mock_upstream.stop = AsyncMock()
            session._session = mock_upstream
            session._temp_user_data_dir = temp_profile_dir

            await session.__aexit__(None, None, None)

            mock_upstream.stop.assert_awaited_once()
            assert not temp_profile_dir.exists()


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


@pytest.mark.asyncio
async def test_nav_skip_mode_avoids_dom_capture() -> None:
    session = MagicMock()
    session.raw = MagicMock()
    session.raw.navigate_to = AsyncMock()

    async def _fast_sleep(_: float) -> None:
        return None

    with patch("hawker_agent.browser.actions.asyncio.sleep", _fast_sleep), \
         patch("hawker_agent.browser.actions._capture_dom_state", AsyncMock(side_effect=AssertionError("should not capture dom"))), \
         patch("hawker_agent.browser.actions._capture_navigation_meta", AsyncMock(return_value={"title": "目标页", "url": "https://example.com/final"})):
        result = await action_nav(session, "https://example.com/start", mode="skip")

    session.raw.navigate_to.assert_awaited_once_with("https://example.com/start")
    assert result.summary == "[OK] 目标页 | DOM=skipped | URL 已变(redirected): https://example.com/final"
    assert result.snapshot["title"] == "目标页"
    assert result.snapshot["url"] == "https://example.com/final"


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


@pytest.mark.asyncio
async def test_inspect_page_returns_nested_dom_payload_shape() -> None:
    state = CodeAgentState()
    history = CodeAgentHistoryList()
    session = type("Session", (), {})()

    reg = ToolRegistry()
    register_core_actions(reg, state, "/tmp/test")

    from hawker_agent.tools import browser_tools as browser_tools_module

    original = browser_tools_module.actions.dom_state
    browser_tools_module.actions.dom_state = AsyncMock(
        return_value=DomActionResult(
            summary="[OK] DOM summary",
            dom="<html>dom</html>",
            snapshot={"title": "OpenAI"},
            context_mode="full",
        )
    )
    try:
        register_browser_tools(reg, session, history, state)
        inspect_page = reg.as_namespace_dict()["inspect_page"]
        payload = await inspect_page(include=["dom"], mode="full")
    finally:
        browser_tools_module.actions.dom_state = original

    assert list(payload.keys()) == ["dom"]
    assert payload["dom"]["summary"] == "[OK] DOM summary"
    assert payload["dom"]["mode"] == "full"
    assert payload["dom"]["snapshot"] == {"title": "OpenAI"}


@pytest.mark.asyncio
async def test_inspect_page_rejects_removed_network_dimension() -> None:
    state = CodeAgentState()
    history = CodeAgentHistoryList()
    session = type("Session", (), {})()
    reg = ToolRegistry()
    register_core_actions(reg, state, "/tmp/test")

    register_browser_tools(reg, session, history, state)
    inspect_page = reg.as_namespace_dict()["inspect_page"]
    result = await inspect_page(include=["network"], mode="summary")

    assert "已移除" in result["error"]
    assert "fetch" in result["error"]


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
            "click_index", "fill_input", "browser_download",
            "get_selector_from_index", "get_cookies", "list_downloaded_files",
        }
        for name in expected:
            assert name in registry, f"工具 {name} 未注册"
        assert "get_network_log" not in registry
        assert len(registry) == 12

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

    def test_nav_description_exposes_mode_guidance_for_llm(self) -> None:
        registry = ToolRegistry()
        mock_session = MagicMock()
        mock_history = MagicMock()
        register_browser_tools(registry, mock_session, mock_history)
        desc = registry.build_description()
        nav_line = next(line for line in desc.splitlines() if "nav(" in line)
        assert "mode" in nav_line
        assert "skip" in nav_line
        assert "summary" in nav_line
        assert "full" in nav_line

    def test_inspect_page_signature_is_shortened_for_prompt(self) -> None:
        registry = ToolRegistry()
        mock_session = MagicMock()
        mock_history = MagicMock()
        register_browser_tools(registry, mock_session, mock_history)
        desc = registry.build_description()
        inspect_line = next(line for line in desc.splitlines() if "inspect_page(" in line)
        assert "selector_index=None" in inspect_line
        assert "mode='summary'" in inspect_line
        assert "**kwargs" in inspect_line
        assert "network" not in inspect_line

    def test_browser_download_docstring_requires_explicit_identity(self) -> None:
        registry = ToolRegistry()
        mock_session = MagicMock()
        mock_history = MagicMock()
        register_browser_tools(registry, mock_session, mock_history)
        ns = registry.as_namespace_dict()
        doc = ns["browser_download"].__doc__ or ""
        assert "必须显式传相同的 ``ref=`` 或 ``entity_key=``" in doc
        assert "不会再根据字段名或 URL 猜测合并" in doc

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

    def test_dom_state_no_params(self) -> None:
        registry = ToolRegistry()
        mock_session = MagicMock()
        mock_history = MagicMock()
        register_browser_tools(registry, mock_session, mock_history)
        ns = registry.as_namespace_dict()
        sig = inspect.signature(ns["dom_state"])
        assert list(sig.parameters.keys()) == ["mode"]


def test_model_prompts_do_not_reintroduce_network_capture_contract() -> None:
    template_dir = Path("hawker_agent/templates")
    texts = "\n".join(path.read_text(encoding="utf-8") for path in template_dir.rglob("*.txt"))
    texts += "\n".join(path.read_text(encoding="utf-8") for path in template_dir.rglob("*.jinja2"))
    texts += "\n".join(path.read_text(encoding="utf-8") for path in template_dir.rglob("*.md"))

    assert "get_network_log" not in texts
    assert "include=[\"network\"]" not in texts
    assert "network=True" not in texts


def test_model_prompts_keep_core_tool_contracts_after_compaction() -> None:
    system_prompt = Path("hawker_agent/templates/system_prompt.jinja2").read_text(encoding="utf-8")
    default_instructions = Path("hawker_agent/templates/default_instructions.txt").read_text(encoding="utf-8")
    observer_prompt = Path("hawker_agent/templates/observer_prompt.jinja2").read_text(encoding="utf-8")

    assert "必须 await" in system_prompt
    assert "先判断页面类型" in system_prompt
    assert "click_index(123)" in system_prompt
    assert "导航后索引失效" in system_prompt
    assert "最小 DOM 交互范式" in system_prompt
    assert "await inspect_page(include=[\"dom\"]" in system_prompt
    assert "多页 DOM 提取范式" in system_prompt
    assert "for page_no in range(3)" in system_prompt
    assert "next_index 必须来自当前页 DOM" in system_prompt
    assert "append_items()" in system_prompt
    assert "ref" in system_prompt or "entity_key" in system_prompt
    assert "final_answer" in default_instructions
    assert "analyze_json_structure" in default_instructions
    assert "fetch(parse=\"body\"" in default_instructions
    assert "ref" in default_instructions or "entity_key" in default_instructions
    assert "先判定网页类型" in default_instructions
    assert "不确定类型时先浏览器驱动分页" in default_instructions
    assert "不要猜 `page=1,2`" in default_instructions
    assert "nextCursor" in default_instructions
    assert "# {{ domain }} — Scraping & Data Extraction" in observer_prompt
    assert "不编造" in observer_prompt
