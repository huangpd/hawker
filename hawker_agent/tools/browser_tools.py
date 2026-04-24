from __future__ import annotations

import json
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
            str: 实际使用的模式，仅为 skip、summary、diff、full 之一。
        """
        normalized = (requested_mode or "auto").lower()
        has_snapshot = _previous_snapshot() is not None
        no_progress_streak = state.no_progress_streak if state is not None else 0
        sop_guard_active = bool(state and state.sop_guided_dom_steps_remaining > 0)

        def _guard_mode(preferred: str) -> str:
            if not sop_guard_active:
                return preferred
            if action_name in {"nav", "nav_search"}:
                guarded = "skip"
            elif preferred != "full":
                return preferred
            elif action_name == "dom_state":
                guarded = "diff" if has_snapshot else "summary"
            else:
                guarded = "diff" if has_snapshot else "summary"
            logger.info(
                "SOP 引导 DOM 护栏生效: action=%s requested=%s -> effective=%s reason=%s",
                action_name,
                preferred,
                guarded,
                state.sop_guided_reason if state is not None else "",
            )
            return guarded

        if normalized in {"skip", "summary", "diff", "full"}:
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
        """导航到目标 URL 并返回结果摘要。`mode` 可选：`auto` 让系统按当前状态选择最省 token 的模式；`skip` 只导航并做轻量页面确认，不读取 DOM；`summary` 返回轻量页面摘要；`diff` 返回相对上一次页面快照的变化摘要；`full` 返回完整 DOM。命中高置信 SOP 且下一步直接提取时，优先用 `mode="skip"`。"""
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

    async def nav_search(query: str, engine: str = "google", mode: str = "auto") -> str:
        """用搜索引擎打开结果页；仅在 `search_web(...)` 不足或必须依赖浏览器交互时再用它。"""
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
        network_structured: bool = False,
        cookie_verbose: bool = True,
        *,
        dom: bool | None = None,
        network: bool | None = None,
        cookies: bool | None = None,
    ) -> dict[str, Any]:
        """统一页面侦察入口；优先用它一次拿到 DOM 摘要、抓包、Cookie 和选择器，而不是分散多次探测。

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
            network_result = await actions.get_network_log(
                session,
                "",
                only_new_network,
                method=network_method,
                status_range=network_status_range,
                content_type_contains=network_content_type_contains,
                only_with_body=network_only_with_body,
                max_entries=network_max_entries,
            )
            if network_structured:
                payload["network"] = network_result
            elif isinstance(network_result, dict):
                payload["network"] = network_result.get("entries", [])
            else:
                payload["network"] = network_result

        if "cookies" in keys:
            payload["cookies"] = await actions.get_cookies(
                session, domain=cookie_domain, verbose=cookie_verbose
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
    async def js(code: str, *fn_args: object, args: list[Any] | tuple[Any, ...] | None = None) -> Any:
        """在当前页面执行 JavaScript,返回值会自动转换为 Python 类型（如 JS 的 Array 变为 Python List）,无需在 JS 中使用 JSON.stringify
        
        注意：
        1. 必须使用 `await js(...)` 调用。
        2. 返回值会自动转换为 Python 类型（如 JS 的 Array 变为 Python List）。
        3. 直接返回所需数据即可，无需在 JS 中使用 JSON.stringify。
        4. 兼容 ``js(function_code, arg1, arg2)`` 和 ``js(function_code, args=[...])`` 两种调用形式。
        
        用法示例:
            items = await js("Array.from(document.querySelectorAll('a')).map(a => a.href)")
        """
        call_args = list(args) if args is not None else list(fn_args)
        if not call_args:
            return await actions.js(session, code)

        wrapped_code = (
            "(() => {\n"
            f"  const __hawker_args = {json.dumps(call_args, ensure_ascii=False)};\n"
            f"  const __hawker_fn = ({code});\n"
            "  if (typeof __hawker_fn !== 'function') {\n"
            "    throw new TypeError('js(code, args=...) requires `code` to evaluate to a function');\n"
            "  }\n"
            "  return __hawker_fn(...__hawker_args);\n"
            "})()"
        )
        return await actions.js(session, wrapped_code)

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

    async def browser_download(url: str, filename: str | None = None, **kwargs: object) -> dict[str, Any]:
        """利用浏览器会话下载文件。

        If a matching URL has already been downloaded in this run, reuse the
        recorded path instead of downloading again.
        """
        if state is not None:
            cached = state.get_download_record(url)
            if cached is not None:
                reused = {
                    "ok": True,
                    "url": cached.get("source_url") or url,
                    "requested_filename": filename or cached.get("requested_filename") or cached.get("filename"),
                    "filename": cached.get("filename"),
                    "path": cached.get("path"),
                    "size": cached.get("size", 0),
                    "method": cached.get("method", "registry"),
                    "reused": True,
                }
                logger.info(
                    "浏览器下载命中注册表复用: url=%s file=%s",
                    url,
                    reused.get("filename"),
                )
                return reused

        result = await actions.browser_download(session, url, filename=filename, **kwargs)
        if state is not None and result.get("ok"):
            state.register_download(
                url=str(result.get("url") or url),
                filename=str(result.get("filename") or filename or ""),
                path=str(result.get("path") or ""),
                size=int(result.get("size") or 0),
                method=str(result.get("method") or "unknown"),
                requested_filename=filename,
            )
        return result

    async def list_downloaded_files() -> list[dict[str, Any]]:
        """返回当前运行已下载文件的注册表快照。"""
        return state.list_downloaded_files() if state is not None else []

    async def get_network_log(
        filter: str = "",
        only_new: bool = False,
        method: str | list[str] | None = None,
        status_range: tuple[int, int] | list[int] | int | str | None = None,
        content_type_contains: str = "",
        only_with_body: bool = False,
        max_entries: int = 20,
        structured: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """读取页面拦截到的 Fetch/XHR 网络请求日志，支持多维过滤。

        默认返回兼容旧接口的 ``list[dict]``。如需 richer 结果，请显式传
        ``structured=True``，此时返回形如
        ``{"entries": [...], "summary": {...}, "_truncated": bool, "filters": {...}}``。
        """
        result = await actions.get_network_log(
            session,
            filter,
            only_new,
            method=method,
            status_range=status_range,
            content_type_contains=content_type_contains,
            only_with_body=only_with_body,
            max_entries=max_entries,
        )
        return result if structured else result.get("entries", [])

    async def get_cookies(domain: str = "", verbose: bool = True) -> list[dict[str, Any]]:
        """
        获取当前浏览器会话的 Cookie，可按 ``domain=`` 过滤。

        默认返回完整字段，以保持兼容性。若希望降低上下文体积，请显式传
        ``verbose=False``，仅保留 ``name/value/domain/path``。

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
        (list_downloaded_files, "网络 & 数据", False),
        (get_network_log, "网络 & 数据", False),
        (get_selector_from_index, "交互", False),
        (get_cookies, "网络 & 数据", False),
    ]
    for fn, category, expose_in_prompt in tools_to_register:
        registry.register(fn, category=category, expose_in_prompt=expose_in_prompt)
