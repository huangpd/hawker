from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
        has_snapshot = _previous_snapshot() is not None
        no_progress_streak = state.no_progress_streak if state is not None else 0
        memory_guard_active = bool(state and state.memory_guided_dom_steps_remaining > 0)

        def _guard_mode(preferred: str) -> str:
            if not memory_guard_active:
                return preferred
            if preferred != "full":
                return preferred
            if action_name in {"nav", "nav_search"}:
                guarded = "summary"
            elif action_name == "dom_state":
                guarded = "diff" if has_snapshot else "summary"
            else:
                guarded = "diff" if has_snapshot else "summary"
            logger.info(
                "记忆引导DOM护栏生效: action=%s requested=%s -> effective=%s reason=%s",
                action_name,
                preferred,
                guarded,
                state.memory_guided_reason if state is not None else "",
            )
            return guarded

        if normalized in {"summary", "diff", "full"}:
            return _guard_mode(normalized)

        if action_name in {"nav", "nav_search"}:
            return _guard_mode("summary")
        if action_name in {"click", "click_index"}:
            if no_progress_streak >= 2:
                return _guard_mode("full")
            return _guard_mode("diff" if has_snapshot else "summary")
        if action_name == "dom_state":
            if no_progress_streak >= 2:
                return _guard_mode("full")
            return _guard_mode("diff" if has_snapshot else "full")
        return _guard_mode("summary")

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
        """导航到 URL，返回摘要字符串；页面上下文会写入下一轮 DOM Workspace。"""
        effective_mode = _resolve_mode("nav", mode)
        result = await actions.nav(
            session,
            url,
            mode=effective_mode,
            previous_snapshot=_previous_snapshot(),
        )
        return _handle_dom_result(result)

    async def dom_state(mode: str = "auto") -> str:
        """获取页面状态摘要；页面上下文会写入下一轮 DOM Workspace。"""
        effective_mode = _resolve_mode("dom_state", mode)
        result = await actions.dom_state(
            session,
            mode=effective_mode,
            previous_snapshot=_previous_snapshot(),
        )
        return _handle_dom_result(result)

    async def nav_search(query: str, engine: str = "google", mode: str = "full") -> str:
        """执行搜索并返回摘要字符串；结果页上下文会写入下一轮 DOM Workspace。"""
        effective_mode = _resolve_mode("nav_search", mode)
        result = await actions.nav_search(
            session,
            query,
            engine,
            mode=effective_mode,
            previous_snapshot=_previous_snapshot(),
        )
        return _handle_dom_result(result)

    async def inspect_page(
        include: list[str] | str | None = None,
        selector_index: int | None = None,
        mode: str = "summary",
        only_new_network: bool = True,
        cookie_domain: str = "",
        network_method: str | list[str] | None = None,
        network_status_range: tuple[int, int] | list[int] | int | str | None = None,
        network_content_type_contains: str = "",
        network_only_with_body: bool = False,
        network_max_entries: int = 20,
        *,
        dom: bool | None = None,
        network: bool | None = None,
        cookies: bool | None = None,
    ) -> dict[str, Any]:
        """统一页面侦察入口，按 ``include=`` 组合 DOM、抓包、Cookie、选择器。

        用法示例::

            await inspect_page(include=["dom"])
            await inspect_page(include=["network"], network_method="POST", network_status_range=(200,299))
            await inspect_page(include=["cookies"], cookie_domain="example.com")

        Args:
            include (list[str] | str | None, optional): 要采集的维度，可传
                ``"dom" | "network" | "cookies"`` 中的任意组合。默认 ``["dom"]``。
                也支持字符串 ``"dom,network"``。
            selector_index (int | None, optional): 若设置，将额外为该索引返回
                ``selector + shadow_path + js_snippet``。
            mode (str, optional): DOM 模式，默认 ``"summary"``，可选
                ``"summary" | "diff" | "full" | "auto"``。
            only_new_network (bool, optional): 仅拉取自上次读取以来的新请求。默认 True。
            cookie_domain (str, optional): Cookie 过滤域（同源/子域匹配）。
            network_method / network_status_range / network_content_type_contains /
            network_only_with_body / network_max_entries: 透传给 ``get_network_log``。

        向后兼容：仍接受旧的布尔参数 ``dom=/network=/cookies=``，当同时传入
        ``include=`` 时优先使用 ``include``。
        """
        # 旧布尔参数的兼容路径
        legacy_keys: list[str] = []
        if dom:
            legacy_keys.append("dom")
        if network:
            legacy_keys.append("network")
        if cookies:
            legacy_keys.append("cookies")

        if include is None:
            keys = legacy_keys or ["dom"]
        elif isinstance(include, str):
            keys = [k.strip().lower() for k in include.split(",") if k.strip()]
        else:
            keys = [str(k).lower() for k in include if str(k).strip()]

        unknown = [k for k in keys if k not in {"dom", "network", "cookies"}]
        if unknown:
            return {
                "error": (
                    f"inspect_page(include=...) 不支持 {unknown}，"
                    "合法值：'dom' / 'network' / 'cookies'。"
                )
            }

        payload: dict[str, Any] = {}

        if "dom" in keys:
            effective_mode = _resolve_mode("dom_state", mode)
            dom_result = await actions.dom_state(
                session,
                mode=effective_mode,
                previous_snapshot=_previous_snapshot(),
            )
            payload["dom"] = {
                "summary": _handle_dom_result(dom_result),
                "mode": dom_result.context_mode,
                "snapshot": dom_result.snapshot or {},
            }

        if "network" in keys:
            payload["network"] = await actions.get_network_log(
                session,
                "",
                only_new_network,
                method=network_method,
                status_range=network_status_range,
                content_type_contains=network_content_type_contains,
                only_with_body=network_only_with_body,
                max_entries=network_max_entries,
            )

        if "cookies" in keys:
            payload["cookies"] = await actions.get_cookies(
                session, domain=cookie_domain
            )

        if selector_index is not None:
            from hawker_agent.browser.dom_utils import get_selector_from_index as _get_selector

            res = await _get_selector(session.raw, selector_index)
            js_snippet = "document"
            for host in res["shadow_path"]:
                js_snippet += f'.querySelector("{host}").shadowRoot'
            js_snippet += f'.querySelector("{res["selector"]}")'
            res["js_snippet"] = js_snippet
            payload["selector"] = res

        if not payload:
            return {"summary": "未请求任何页面状态"}
        return payload
    async def js(code: str) -> Any:
        """在当前页面执行 JavaScript,返回值会自动转换为 Python 类型（如 JS 的 Array 变为 Python List）,无需在 JS 中使用 JSON.stringify
        
        注意：
        1. 必须使用 `await js(...)` 调用。
        2. 返回值会自动转换为 Python 类型（如 JS 的 Array 变为 Python List）。
        3. 直接返回所需数据即可，无需在 JS 中使用 JSON.stringify。
        
        用法示例:
            items = await js("Array.from(document.querySelectorAll('a')).map(a => a.href)")
        """
        return await actions.js(session, code)

    async def click(selector: str, index: int = 0, mode: str = "auto") -> str:
        """点击元素并返回摘要字符串；页面变化上下文会写入下一轮 DOM Workspace。"""
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
        """按 DOM 索引点击元素并返回摘要字符串；页面变化上下文会写入下一轮 DOM Workspace。"""
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
        return await actions.browser_download(session, url, filename=filename, **kwargs)

    async def get_network_log(
        filter: str = "",
        only_new: bool = False,
        method: str | list[str] | None = None,
        status_range: tuple[int, int] | list[int] | int | str | None = None,
        content_type_contains: str = "",
        only_with_body: bool = False,
        max_entries: int = 20,
    ) -> dict[str, Any]:
        """读取页面拦截到的 Fetch/XHR 网络请求日志，支持多维过滤。

        返回形如 ``{"entries": [...], "summary": {...}, "_truncated": bool,
        "filters": {...}}``。``summary`` 中会包含 ``likely_data_api``
        （启发式识别到的业务数据接口）和 ``errors``（状态码 >= 400）。
        """
        return await actions.get_network_log(
            session,
            filter,
            only_new,
            method=method,
            status_range=status_range,
            content_type_contains=content_type_contains,
            only_with_body=only_with_body,
            max_entries=max_entries,
        )

    async def get_cookies(domain: str = "", verbose: bool = False) -> list[dict[str, Any]]:
        """
        获取当前浏览器会话的 Cookie，可按 ``domain=`` 过滤。

        默认只返回 ``name/value/domain/path`` 4 个关键字段，避免长字段
        撑爆上下文。需要完整字段（secure/httpOnly/sameSite/expires 等）
        请显式传 ``verbose=True``。

        Args:
            domain (str, optional): 仅返回同域或子域的 Cookie。例如
                ``domain="example.com"`` 会命中 ``.example.com``、
                ``api.example.com``；留空则不过滤。
            verbose (bool, optional): 是否返回完整字段。默认 False。

        用法示例:
            cookies = await get_cookies(domain="example.com")
            cookie_dict = {c['name']: c['value'] for c in cookies}
            res = await http_json(url, cookies=cookie_dict)
        """
        return await actions.get_cookies(session, domain=domain, verbose=verbose)

    async def get_selector_from_index(index: int) -> dict[str, Any]:
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
        (nav, "导航与页面", True),
        (dom_state, "导航与页面", False),
        (nav_search, "导航与页面", True),
        (inspect_page, "导航与页面", True),
        (js, "导航与页面", True),
        (click, "交互", False),
        (click_index, "交互", True),
        (fill_input, "交互", True),
        (browser_download, "网络 & 数据", True),
        (get_network_log, "网络 & 数据", False),
        (get_selector_from_index, "交互", False),
        (get_cookies, "网络 & 数据", False),
    ]
    for fn, category, expose_in_prompt in tools_to_register:
        registry.register(fn, category=category, expose_in_prompt=expose_in_prompt)
