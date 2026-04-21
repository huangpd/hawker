from __future__ import annotations

import logging
from pathlib import Path

from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession as _UpstreamBrowserSession

from hawker_agent.config import get_settings

logger = logging.getLogger(__name__)


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
        self._headless: bool = headless if headless is not None else settings.headless
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
        self._session: _UpstreamBrowserSession | None = None
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
        profile = BrowserProfile(**profile_kwargs)
        self._session = _UpstreamBrowserSession(browser_profile=profile)
        await self._session.start()
        
        logger.info("浏览器会话已启动 (headless=%s)", self._headless)
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
