from __future__ import annotations

from typing import TYPE_CHECKING

from hawker_agent.browser import actions
from hawker_agent.browser.actions import DomActionResult
from hawker_agent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from hawker_agent.browser.session import BrowserSession
    from hawker_agent.models.history import CodeAgentHistoryList


def register_browser_tools(
    registry: ToolRegistry,
    session: BrowserSession,
    history: CodeAgentHistoryList,
) -> None:
    """注册所有浏览器工具到 registry，绑定 session 和 history。"""

    def _handle_dom_result(result: DomActionResult) -> str:
        """处理包含 DOM 的动作结果：注入到 history，返回摘要。"""
        if result.dom:
            history.inject_dom(result.dom)
        return result.summary

    async def nav(url: str) -> str:
        """导航到URL。返回完整DOM状态摘要。"""
        result = await actions.nav(session, url)
        return _handle_dom_result(result)

    async def dom_state() -> str:
        """获取当前页面的完整DOM状态（不导航）。用于点击/翻页后刷新DOM。"""
        result = await actions.dom_state(session)
        return _handle_dom_result(result)

    async def nav_search(query: str, engine: str = "duckduckgo") -> str:
        """搜索引擎搜索。engine可选duckduckgo(默认)/google/bing。返回搜索结果页DOM。"""
        result = await actions.nav_search(session, query, engine)
        return _handle_dom_result(result)

    async def js(code: str) -> str:
        """在当前页面执行JavaScript，返回完整结果，并自动打印摘要。"""
        return await actions.js(session, code)

    async def click(selector: str, index: int = 0) -> str:
        """点击页面上匹配CSS选择器的元素。"""
        result = await actions.click(session, selector, index)
        return _handle_dom_result(result)

    async def click_index(index: int) -> str:
        """通过 DOM 索引 [i_*] 点击元素（推荐，比 CSS 选择器精确）。"""
        result = await actions.click_index(session, index)
        return _handle_dom_result(result)

    async def fill_input(index: int, text: str) -> str:
        """通过 DOM 索引 [i_*] 向输入框填写文本（模拟逐字输入，兼容 React/Vue）。"""
        return await actions.fill_input(session, index, text)

    async def browser_download(url: str, filename: str | None = None, **kwargs: object) -> str:
        """利用浏览器会话下载文件（自动继承 Cookie/登录态，推荐用于 PDF/图片下载）。"""
        from hawker_agent.tools import http_tools
        # 1. 提取 Cookie
        cookies = await actions.get_cookies(session)
        # 2. 调用通用下载引擎（自动注入 run_dir 已在 namespace 层处理，此处透传 kwargs）
        return await http_tools.download_file(url, filename, cookies=cookies, **kwargs)  # type: ignore

    async def get_network_log(filter: str = "", only_new: bool = False) -> str:
        """读取页面拦截到的 Fetch/XHR 网络请求日志。返回 JSON 字符串。"""
        return await actions.get_network_log(session, filter, only_new)

    async def get_cookies() -> list[dict]:
        """
        获取当前浏览器会话的所有 Cookie。
        适用于：当你需要使用 http_json() 或 http_request() 发送请求，且需要继承浏览器的登录状态时。
        用法示例:
            cookies = await get_cookies()
            cookie_dict = {c['name']: c['value'] for c in cookies}
            res = await http_json(url, cookies=cookie_dict)
        """
        return await actions.get_cookies(session)

    async def get_selector_from_index(index: int) -> dict:
        """
        通过 DOM 索引 [i_*] 获取该元素的严谨 CSS 选择器及访问路径。
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
        nav, dom_state, nav_search, js, click, click_index, 
        fill_input, browser_download, get_network_log,
        get_selector_from_index, get_cookies
    ]
    for fn in tools_to_register:
        registry.register(fn)
