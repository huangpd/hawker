from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Any

import httpx
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession as _UpstreamBrowserSession

from hawker_agent.config import get_settings

logger = logging.getLogger(__name__)
_BROWSER_USE_V3_BASE_URL = "https://api.browser-use.com/api/v3"


def _is_linux_server_without_display() -> bool:
    """Detect common server environments where headed Chromium will hang on launch."""
    if platform.system() != "Linux":
        return False
    return not any(os.environ.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY"))


def _is_root_user() -> bool:
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        return False
    try:
        return geteuid() == 0
    except OSError:
        return False


def _server_browser_overrides() -> dict[str, object]:
    """Return Linux server-friendly browser launch defaults.

    These defaults make local Chromium launches much more reliable on:
    - headless Linux servers without X11/Wayland
    - containers / root users where Chromium sandbox often fails
    """
    overrides: dict[str, object] = {}
    extra_args = [
        "--disable-dev-shm-usage",
        "--no-first-run",
        "--no-default-browser-check",
    ]

    if _is_linux_server_without_display():
        overrides["headless"] = True

    if _is_root_user():
        overrides["chromium_sandbox"] = False
        extra_args.append("--no-sandbox")

    if extra_args:
        overrides["args"] = extra_args

    return overrides


class BrowserSession:
    """browser_use.BrowserSession 的封装类，支持作为异步上下文管理器使用。

    该类负责启动浏览器会话、自动管理下载目录，并在退出时将下载文件归档到指定的任务目录。

    Attributes:
        target_dir (Path | None): 任务产物的最终保存目录。
        netlog_installed (bool): 是否已安装网络监听脚本。
        netlog_cursor (int): 网络请求日志的读取游标。
    """

    def __init__(self, headless: bool | None = None) -> None:
        """初始化浏览器会话封装对象。

        Args:
            headless (bool | None, optional): 是否以无头模式运行。如果为 None，则从设置中读取。默认为 None。
        """
        settings = get_settings()
        self._settings = settings
        self._headless: bool = headless if headless is not None else settings.headless
        self._browser_provider = settings.browser_provider.lower()
        self._browser_profile_kwargs = {
            "headless": self._headless,
            "executable_path": settings.browser_executable_path,
            "user_data_dir": settings.browser_user_data_dir,
            "profile_directory": settings.browser_profile_directory,
            "storage_state": settings.browser_storage_state,
            "channel": settings.browser_channel,
            "cdp_url": settings.browser_cdp_url,
            # 显式下载由 curl-cffi 完成，关闭 browser-use 的 PDF 自动下载避免竞争。
            "auto_download_pdfs": False,
        }
        self._browser_profile_kwargs.update(_server_browser_overrides())
        self._session: _UpstreamBrowserSession | None = None
        self._cloud_http_client: httpx.AsyncClient | None = None
        self._cloud_session: dict[str, Any] | None = None
        self.target_dir: Path | None = None  # 本次任务产物的最终保存目录
        # 内部状态
        self.netlog_installed: bool = False
        self.netlog_cursor: int = 0

    async def __aenter__(self) -> BrowserSession:
        """进入异步上下文，启动浏览器并准备下载目录。

        Returns:
            BrowserSession: 已启动的会话封装对象。
        """
        profile_kwargs = {
            key: value
            for key, value in self._browser_profile_kwargs.items()
            if value is not None
        }
        if self._browser_provider == "browser_use_cloud":
            await self._start_browser_use_cloud_session()
            profile_kwargs["cdp_url"] = self._require_cloud_cdp_url()

        profile = BrowserProfile(**profile_kwargs)
        self._session = _UpstreamBrowserSession(browser_profile=profile)
        await self._session.start()

        logger.info(
            "浏览器会话已启动 (headless=%s, chromium_sandbox=%s)",
            profile_kwargs.get("headless"),
            profile_kwargs.get("chromium_sandbox", True),
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """退出异步上下文，关闭浏览器并归档下载产物。

        Args:
            exc_type (type[BaseException] | None): 异常类型。
            exc_val (BaseException | None): 异常实例。
            exc_tb (object): 异常堆栈信息。
        """
        # 1. 先安全关闭浏览器，确保所有下载流已关闭并落盘
        if self._session:
            try:
                await self._session.stop()
            except Exception:
                logger.debug("浏览器会话关闭异常", exc_info=True)
            self._session = None

        if self._cloud_http_client and self._cloud_session:
            try:
                await self._cloud_http_client.patch(
                    f"/browsers/{self._cloud_session['id']}",
                    json={"action": "stop"},
                )
            except Exception:
                logger.debug("browser-use cloud session 关闭异常", exc_info=True)
            self._cloud_session = None
        if self._cloud_http_client:
            try:
                await self._cloud_http_client.aclose()
            except Exception:
                logger.debug("browser-use cloud client 关闭异常", exc_info=True)
            self._cloud_http_client = None
            
        self.netlog_installed = False
        self.netlog_cursor = 0
        logger.info("浏览器会话已关闭")

    @property
    def raw(self) -> _UpstreamBrowserSession:
        """获取底层的 browser_use.BrowserSession 实例。

        Returns:
            _UpstreamBrowserSession: 底层会话实例。

        Raises:
            RuntimeError: 如果在进入异步上下文之前访问该属性。
        """
        if self._session is None:
            raise RuntimeError("BrowserSession 未启动，请在 async with 块内使用")
        return self._session

    async def _start_browser_use_cloud_session(self) -> None:
        """Create a browser-use cloud session and hydrate the CDP URL."""
        if self._cloud_session is not None:
            return
        if not self._settings.browser_use_api_key:
            raise RuntimeError("BROWSER_USE_API_KEY 未配置，无法启动 browser_use_cloud")

        self._cloud_http_client = httpx.AsyncClient(
            base_url=self._settings.browser_use_base_url or _BROWSER_USE_V3_BASE_URL,
            headers={"X-Browser-Use-API-Key": self._settings.browser_use_api_key},
            timeout=30.0,
        )
        body: dict[str, Any] = {}
        if self._settings.browser_use_profile_id:
            body["profileId"] = self._settings.browser_use_profile_id
        if self._settings.browser_use_proxy_country_code is not None:
            body["proxyCountryCode"] = self._settings.browser_use_proxy_country_code
        if self._settings.browser_use_enable_recording:
            body["enableRecording"] = True

        response = await self._cloud_http_client.post("/browsers", json=body)
        response.raise_for_status()
        self._cloud_session = response.json()
        logger.info(
            "browser-use cloud session 已创建: id=%s live_url=%s",
            self._cloud_session.get("id", ""),
            self._cloud_session.get("liveUrl", self._cloud_session.get("live_url", "")),
        )

    def _require_cloud_cdp_url(self) -> str:
        cdp_url = None if self._cloud_session is None else self._cloud_session.get(
            "cdpUrl", self._cloud_session.get("cdp_url")
        )
        if not cdp_url:
            raise RuntimeError("browser-use cloud session 未返回 cdp_url")
        return str(cdp_url)
