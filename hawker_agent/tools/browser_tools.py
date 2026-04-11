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

    async def browser_download(url: str) -> str:
        """通过浏览器会话下载文件。文件将自动归档到任务目录。"""
        return await actions.browser_download(session, url)

    async def get_network_log(filter: str = "", only_new: bool = False) -> list:
        """读取页面拦截到的 Fetch/XHR 网络请求日志。返回解析后的列表。"""
        return await actions.get_network_log(session, filter, only_new)

    # 注册所有工具
    tools_to_register = [
        nav, dom_state, nav_search, js, click, click_index, 
        fill_input, browser_download, get_network_log
    ]
    for fn in tools_to_register:
        registry.register(fn)
