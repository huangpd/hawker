from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hawker_agent.browser import actions
from hawker_agent.browser.actions import DomActionResult
from hawker_agent.browser.dom_utils import render_dom_summary
from hawker_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from hawker_agent.browser.session import BrowserSession
    from hawker_agent.models.history import CodeAgentHistoryList
    from hawker_agent.models.state import CodeAgentState


def register_browser_tools(
    registry: ToolRegistry,
    session: BrowserSession,
    history: CodeAgentHistoryList,
    state: CodeAgentState | None = None,
) -> None:
    """
    注册浏览器相关工具，并接入 DOM 上下文管理。

    参数:
        registry (ToolRegistry): 工具注册表。
        session (BrowserSession): 浏览器会话。
        history (CodeAgentHistoryList): 对话历史管理器。
        state (CodeAgentState | None): 运行状态，用于记录 DOM 快照。

    返回:
        None: 直接向注册表写入工具。
    """

    def _handle_dom_result(result: DomActionResult) -> str:
        """
        处理浏览器动作结果并同步上下文状态。

        参数:
            result (DomActionResult): 浏览器动作结果。

        返回:
            str: 返回给模型的动作摘要。
        """
        if state is not None and result.snapshot is not None:
            state.last_dom_snapshot = result.snapshot
        if result.dom:
            mode = getattr(result, "context_mode", "full")
            folded = render_dom_summary(result.snapshot) if result.snapshot else result.summary
            history.inject_browser_context(
                result.dom,
                mode=mode,
                folded_content=folded,
            )
        # 人类日志：记录浏览器动作摘要，便于排查“执行了但 Observation 显示[无输出]”的情况
        logger.info("浏览器动作摘要: %s", result.summary)
        return result.summary

    def _previous_snapshot() -> dict | None:
        """
        获取上一次页面快照。

        参数:
            无

        返回:
            dict | None: 上一次页面快照，不存在则返回 None。
        """
        return state.last_dom_snapshot if state is not None else None

    def _resolve_mode(action_name: str, requested_mode: str | None) -> str:
        """
        根据动作类型和运行状态选择实际 DOM 模式。

        参数:
            action_name (str): 动作名称。
            requested_mode (str | None): 用户请求的模式。

        返回:
            str: 实际使用的模式，仅为 summary、diff、full 之一。
        """
        normalized = (requested_mode or "auto").lower()
        if normalized in {"summary", "diff", "full"}:
            return normalized

        has_snapshot = _previous_snapshot() is not None
        no_progress_streak = state.no_progress_streak if state is not None else 0

        if action_name in {"nav", "nav_search"}:
            return "summary"
        if action_name in {"click", "click_index"}:
            if no_progress_streak >= 2:
                return "full"
            return "diff" if has_snapshot else "summary"
        if action_name == "dom_state":
            if no_progress_streak >= 2:
                return "full"
            return "diff" if has_snapshot else "full"
        return "summary"

    def _diagnostic_mode() -> str:
        """
        选择失败后的诊断 DOM 模式。

        参数:
            无

        返回:
            str: 诊断使用的模式，仅为 diff 或 full。
        """
        no_progress_streak = state.no_progress_streak if state is not None else 0
        if no_progress_streak >= 2:
            return "full"
        return "diff" if _previous_snapshot() is not None else "full"

    def _should_collect_diagnostic(action_name: str, requested_mode: str | None, summary: str) -> bool:
        """
        判断当前动作失败后是否需要自动补充 DOM 诊断信息。

        参数:
            action_name (str): 动作名称。
            requested_mode (str | None): 用户请求的模式。
            summary (str): 动作摘要。

        返回:
            bool: 是否需要自动补充诊断上下文。
        """
        normalized = (requested_mode or "auto").lower()
        if action_name not in {"click", "click_index"}:
            return False
        if normalized not in {"auto", "summary"}:
            return False
        return summary.startswith("[失败]")

    async def nav(url: str, mode: str = "auto") -> str:
        """导航到URL。默认 `mode=auto`，系统会优先选择轻量 `summary`。"""
        effective_mode = _resolve_mode("nav", mode)
        result = await actions.nav(
            session,
            url,
            mode=effective_mode,
            previous_snapshot=_previous_snapshot(),
        )
        return _handle_dom_result(result)

    async def dom_state(mode: str = "auto") -> str:
        """获取当前页面状态。默认 `mode=auto`，系统会在 `diff/full` 之间自动选择。"""
        effective_mode = _resolve_mode("dom_state", mode)
        result = await actions.dom_state(
            session,
            mode=effective_mode,
            previous_snapshot=_previous_snapshot(),
        )
        return _handle_dom_result(result)

    async def nav_search(query: str, engine: str = "duckduckgo", mode: str = "auto") -> str:
        """搜索引擎搜索。默认 `mode=auto`，系统会优先选择轻量 `summary`。"""
        effective_mode = _resolve_mode("nav_search", mode)
        result = await actions.nav_search(
            session,
            query,
            engine,
            mode=effective_mode,
            previous_snapshot=_previous_snapshot(),
        )
        return _handle_dom_result(result)

    async def js(code: str) -> str:
        """在当前页面执行JavaScript，返回完整结果，并自动打印摘要"""
        return await actions.js(session, code)

    async def click(selector: str, index: int = 0, mode: str = "auto") -> str:
        """点击页面元素。默认 `mode=auto`，系统会优先使用 `diff`，失败时自动补诊断。"""
        effective_mode = _resolve_mode("click", mode)
        result = await actions.click(
            session,
            selector,
            index,
            mode=effective_mode,
            previous_snapshot=_previous_snapshot(),
        )
        summary = _handle_dom_result(result)
        if _should_collect_diagnostic("click", mode, summary):
            diag_mode = _diagnostic_mode()
            diagnostic = await actions.dom_state(
                session,
                mode=diag_mode,
                previous_snapshot=_previous_snapshot(),
            )
            _handle_dom_result(diagnostic)
            return f"{summary} | 已自动补充 DOM={diag_mode}"
        return summary

    async def click_index(index: int, mode: str = "auto") -> str:
        """通过 DOM 索引点击元素。默认 `mode=auto`，系统会优先使用 `diff`，失败时自动补诊断。"""
        effective_mode = _resolve_mode("click_index", mode)
        result = await actions.click_index(
            session,
            index,
            mode=effective_mode,
            previous_snapshot=_previous_snapshot(),
        )
        summary = _handle_dom_result(result)
        if _should_collect_diagnostic("click_index", mode, summary):
            diag_mode = _diagnostic_mode()
            diagnostic = await actions.dom_state(
                session,
                mode=diag_mode,
                previous_snapshot=_previous_snapshot(),
            )
            _handle_dom_result(diagnostic)
            return f"{summary} | 已自动补充 DOM={diag_mode}"
        return summary

    async def fill_input(index: int, text: str) -> str:
        """通过 DOM 索引 [i_*] 向输入框填写文本"""
        return await actions.fill_input(session, index, text)

    async def browser_download(url: str, filename: str | None = None, **kwargs: object) -> str:
        """利用浏览器会话下载文件"""
        from hawker_agent.tools import http_tools
        # 1. 提取 Cookie
        cookies = await actions.get_cookies(session)
        # 2. 调用通用下载引擎（自动注入 run_dir 已在 namespace 层处理，此处透传 kwargs）
        return await http_tools.download_file(url, filename, cookies=cookies, **kwargs)  # type: ignore

    async def get_network_log(filter: str = "", only_new: bool = False) -> list:
        """读取页面拦截到的 Fetch/XHR 网络请求日志。返回 list """
        return await actions.get_network_log(session, filter, only_new)

    async def get_cookies() -> list[dict]:
        """
        获取当前浏览器会话的所有 Cookie
        适用于：当你需要使用 http_json() 或 http_request() 发送请求，且需要继承浏览器的登录状态时。
        用法示例:
            cookies = await get_cookies()
            cookie_dict = {c['name']: c['value'] for c in cookies}
            res = await http_json(url, cookies=cookie_dict)
        """
        return await actions.get_cookies(session)

    async def get_selector_from_index(index: int) -> dict:
        """
        通过 DOM 索引 [i_*] 获取该元素的严谨 CSS 选择器及访问路径
        返回: {"selector": str, "shadow_path": list, "js_snippet": str}
        用法示例:
            info = await get_selector_from_index(45)
            # 在循环或 js() 中使用 info['js_snippet'] 确保定位稳定性
            # 例如: data = await js(f"return {info['js_snippet']}.innerText")
        适用于: 1. 循环点击翻页按钮; 2. 需要精准穿透 Shadow DOM; 3. 复杂的 JS 提取逻辑。
        """
        from hawker_agent.browser.dom_utils import get_selector_from_index as _get_selector
        res = await _get_selector(session.raw, index)
        
        # 额外生成一个方便 LLM 直接 copy-paste 的 JS 访问表达式
        js_snippet = "document"
        for host in res["shadow_path"]:
            js_snippet += f'.querySelector("{host}").shadowRoot'
        js_snippet += f'.querySelector("{res["selector"]}")'
        
        res["js_snippet"] = js_snippet
        return res

    # 注册所有工具
    tools_to_register = [
        (nav, "导航与页面"),
        (dom_state, "导航与页面"),
        (nav_search, "导航与页面"),
        (js, "导航与页面"),
        (click, "交互"),
        (click_index, "交互"),
        (fill_input, "交互"),
        (browser_download, "网络 & 数据"),
        (get_network_log, "网络 & 数据"),
        (get_selector_from_index, "交互"),
        (get_cookies, "网络 & 数据")
    ]
    for fn, category in tools_to_register:
        registry.register(fn, category=category)
