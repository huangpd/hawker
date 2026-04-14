from __future__ import annotations

import logging
import shutil
import tempfile
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
        self._session: _UpstreamBrowserSession | None = None
        self._download_path: Path | None = None
        self.target_dir: Path | None = None  # 本次任务产物的最终保存目录
        # 内部状态
        self.netlog_installed: bool = False
        self.netlog_cursor: int = 0

    async def __aenter__(self) -> BrowserSession:
        """进入异步上下文，启动浏览器并准备下载目录。

        Returns:
            BrowserSession: 已启动的会话封装对象。
        """
        # 1. 创建专用的临时下载目录
        self._download_path = Path(tempfile.mkdtemp(prefix="hawker_browser_"))
        
        # 2. 启动浏览器
        profile = BrowserProfile(headless=self._headless)
        self._session = _UpstreamBrowserSession(browser_profile=profile)
        await self._session.start()
        
        logger.info("浏览器会话已启动 (headless=%s, tmp_download=%s)", 
                    self._headless, self._download_path)
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
            
        # 2. 收割产物：将所有下载文件移动到 target_dir
        if self.target_dir and self.target_dir.exists():
            # 扫描我们要搜刮的目录列表
            harvest_targets = []
            if self._download_path and self._download_path.exists():
                harvest_targets.append(self._download_path)
            
            # 同时扫描 browser-use 可能产生的其他随机下载目录
            u_temp_dir = Path(tempfile.gettempdir())
            harvest_targets.extend(list(u_temp_dir.glob("browser-use-downloads-*")))

            for d in harvest_targets:
                if not d.is_dir(): continue
                try:
                    for f in d.rglob("*"):
                        if f.is_file() and f.stat().st_size > 0:
                            # 避免重复移动同一个文件（多个目录可能有重叠）
                            if not f.exists(): continue
                            
                            dest = self.target_dir / f.name
                            # 处理同名冲突：增加随机后缀
                            if dest.exists():
                                short_id = Path(tempfile.mktemp()).name[-6:]
                                dest = self.target_dir / f"{f.stem}_{short_id}{f.suffix}"
                            
                            shutil.move(str(f), str(dest))
                            logger.info("📦 归档下载产物: %s", dest.name)
                except Exception as e:
                    logger.debug("归档目录 %s 时发生错误: %s", d, e)

        # 3. 彻底清理所有临时下载目录
        if self._download_path and self._download_path.exists():
            try:
                shutil.rmtree(self._download_path)
            except Exception: pass

        u_temp_dir = Path(tempfile.gettempdir())
        for p in u_temp_dir.glob("browser-use-downloads-*"):
            if p.is_dir():
                try:
                    shutil.rmtree(p)
                except Exception: pass
                
        self.netlog_installed = False
        self.netlog_cursor = 0
        logger.info("浏览器会话已关闭并完成产物归档")

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
