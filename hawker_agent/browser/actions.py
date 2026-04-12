from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from hawker_agent.browser.cdp import get_cdp, run_js
from hawker_agent.browser.netlog import ensure_network_monitor
from hawker_agent.observability import emit_observation
from hawker_agent.tools.data_tools import get_type_signature

if TYPE_CHECKING:
    from hawker_agent.browser.session import BrowserSession

logger = logging.getLogger(__name__)


@dataclass
class DomActionResult:
    """浏览器动作结果，包含摘要和可选的完整 DOM 状态。"""

    summary: str
    dom: str | None = None


# ─── 私有辅助 ──────────────────────────────────────────────────


async def _get_dom_state(session: BrowserSession) -> str:
    """调用 browser_use 的 DomService 获取完整的 DOM 状态表示，并增加细腻的感知上下文。"""
    try:
        state = await session.raw.get_browser_state_summary(include_screenshot=False)
        assert state.dom_state is not None
        dom_repr = state.dom_state.eval_representation()
        if not dom_repr:
            dom_repr = "(空 DOM，页面可能仍在加载)"

        lines = [f"[OK] {state.title}", f"URL: {state.url}"]

        # 1. 空间感知：滚动位置元数据
        pages_above = 0.0
        pages_below = 0.0
        if state.page_info:
            pi = state.page_info
            vh = pi.viewport_height or 1
            pages_above = pi.pixels_above / vh
            pages_below = pi.pixels_below / vh
            if pages_above > 0 or pages_below > 0:
                lines.append(f"滚动: 上方 {pages_above:.1f} 页, 下方 {pages_below:.1f} 页")

        # 2. 加载感知：网络请求监控
        if state.pending_network_requests:
            lines.append(f"⚠️ 页面仍在加载: {len(state.pending_network_requests)} 个待处理请求")
            # 显示前 3 个耗时请求以辅助决策
            sorted_requests = sorted(state.pending_network_requests, key=lambda x: x.get('duration', 0), reverse=True)
            for req in sorted_requests[:3]:
                url_short = req['url'][:60] + "..." if len(req['url']) > 60 else req['url']
                duration = req.get('duration', 0)
                lines.append(f"  - [{duration:.1f}s] {url_short}")

        # 多标签页
        if len(state.tabs) > 1:
            lines.append(f"标签页: {len(state.tabs)} 个")
            for tab in state.tabs[:5]:
                lines.append(f"  - {tab.target_id[-4:]}: {tab.title[:30]}")

        # 3.2 压缩重复的空白
        dom_repr = re.sub(r"\n[\t ]+", "\n", dom_repr)
        
        dom_parts = []
        if pages_above > 0.1:
            dom_parts.append(f"\n... ({pages_above:.1f} pages of content above current view) ...\n")
        
        dom_parts.append(dom_repr)
        
        if pages_below > 0.1:
            dom_parts.append(f"\n... ({pages_below:.1f} pages of content below current view) ...\n")
        else:
            dom_parts.append("\n[End of page]")

        full_dom = "".join(dom_parts)
        
        # 截断保护
        lines.append("")
        max_dom = 15000
        if len(full_dom) > max_dom:
            lines.append(full_dom[:max_dom])
            lines.append(f"\n[DOM 截断，共 {len(full_dom)} 字符，用 js() 探索更多]")
        else:
            lines.append(full_dom)

        return "\n".join(lines)
    except Exception as e:
        # 降级到简单模式
        raw = await run_js(
            session,
            "(function(){return JSON.stringify({title:document.title,"
            "text:document.body?document.body.innerText.length:0})})()",
        )
        try:
            info = json.loads(raw)
            return (
                f"[OK] {info.get('title', '')} | "
                f"文本:{info.get('text', 0)}字符 (DOM服务不可用: {e})"
            )
        except Exception:
            return f"[OK] 已导航 (DOM服务不可用: {e})"


def _log_js_summary(raw: str) -> None:
    """记录 JS 执行结果摘要。"""
    if raw.startswith("[JS错误]"):
        emit_observation(raw[:200])
        return
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            sample = ""
            sig = ""
            if parsed:
                sample = json.dumps(parsed[0], ensure_ascii=False)
                if len(sample) > 120:
                    sample = sample[:120] + "..."
                if isinstance(parsed[0], dict):
                    sig = get_type_signature(parsed[0])
            emit_observation(
                f"[js] 返回 {len(parsed)} 条数据"
                + (f" | 签名: {sig}" if sig else "")
                + (f" | 样本: {sample}" if sample else "")
            )
        elif isinstance(parsed, dict):
            emit_observation(f"[js] 返回 dict, {len(parsed)} 个键: {get_type_signature(parsed)}")
        else:
            emit_observation(f"[js] 返回: {str(parsed)[:120]}")
    except (json.JSONDecodeError, ValueError):
        preview = raw[:120].replace("\n", " ")
        emit_observation(f"[js] 返回 {len(raw)} 字符: {preview}{'...' if len(raw) > 120 else ''}")


def _build_search_url(query: str, engine: str) -> str | None:
    """构建搜索引擎 URL，不支持的引擎返回 None。"""
    encoded = urllib.parse.quote_plus(query)
    urls = {
        "duckduckgo": f"https://duckduckgo.com/?q={encoded}",
        "google": f"https://www.google.com/search?q={encoded}&udm=14",
        "bing": f"https://www.bing.com/search?q={encoded}",
    }
    return urls.get(engine.lower())


# ─── 公开动作 ──────────────────────────────────────────────────


async def nav(session: BrowserSession, url: str) -> DomActionResult:
    """导航到 URL，返回 DOM 状态摘要 + 完整 DOM。"""
    await ensure_network_monitor(session)
    await session.raw.navigate_to(url)
    await asyncio.sleep(1)
    session.netlog_cursor = 0  # 页面跳转后 __netlog 被清空，重置游标
    dom_text = await _get_dom_state(session)
    first_line = dom_text.split("\n", 1)[0] if dom_text else "[OK]"
    return DomActionResult(summary=first_line, dom=dom_text)


async def dom_state(session: BrowserSession) -> DomActionResult:
    """获取当前页面的完整 DOM 状态（不导航）。"""
    dom_text = await _get_dom_state(session)
    first_line = dom_text.split("\n", 1)[0] if dom_text else "[OK]"
    return DomActionResult(summary=first_line, dom=dom_text)


async def nav_search(
    session: BrowserSession, query: str, engine: str = "duckduckgo"
) -> DomActionResult:
    """用搜索引擎搜索，导航到结果页并返回 DOM 状态。"""
    url = _build_search_url(query, engine)
    if not url:
        return DomActionResult(
            summary=f"[错误] 不支持的搜索引擎: {engine}，可用: duckduckgo, google, bing"
        )
    return await nav(session, url)


async def js(session: BrowserSession, code: str) -> str:
    """在当前页面执行 JavaScript，返回完整结果。"""
    raw = await run_js(session, code)
    _log_js_summary(raw)
    return raw


async def click(session: BrowserSession, selector: str, index: int = 0) -> DomActionResult:
    """点击页面上匹配 CSS 选择器的元素。"""
    raw = await run_js(
        session,
        f"""(function(){{
        var els=document.querySelectorAll('{selector}');
        if(els.length===0) return JSON.stringify({{error:"未找到元素: {selector}"}});
        var idx={index};
        if(idx>=els.length) idx=els.length-1;
        var el=els[idx];
        var tag=el.tagName.toLowerCase();
        var text=(el.textContent||'').trim().substring(0,30);
        el.scrollIntoView({{block:'center'}});
        el.click();
        return JSON.stringify({{ok:true,tag:tag,text:text,total:els.length}});
    }})()""",
    )
    try:
        r = json.loads(raw)
        if "error" in r:
            return DomActionResult(summary=f"[失败] {r['error']}")
        await asyncio.sleep(1)
        # 点击后页面可能变化，刷新 DOM
        dom_text: str | None = None
        try:
            dom_text = await _get_dom_state(session)
        except Exception:
            pass
        summary = f"[OK] 点击了 <{r['tag']}>{r['text']}</{r['tag']}> (共{r['total']}个匹配)"
        return DomActionResult(summary=summary, dom=dom_text)
    except Exception:
        return DomActionResult(summary=f"[OK] 已点击 {selector}")


async def click_index(session: BrowserSession, index: int) -> DomActionResult:
    """通过 DOM 索引 [i_*] 点击元素（比 CSS 选择器可靠）。"""
    node = await session.raw.get_element_by_index(index)
    if node is None:
        return DomActionResult(
            summary=f"[失败] 未找到索引 [i_{index}] 的元素，可能页面已变化，用 dom_state() 刷新"
        )
    try:
        cdp = await session.raw.cdp_client_for_node(node)
    except Exception:
        cdp = await get_cdp(session)
    bid = node.backend_node_id
    sid = cdp.session_id
    # 滚入视图
    try:
        await cdp.cdp_client.send.DOM.scrollIntoViewIfNeeded(
            params={"backendNodeId": bid}, session_id=sid
        )
    except Exception:
        pass
    # 解析为 JS 对象并点击
    try:
        result = await cdp.cdp_client.send.DOM.resolveNode(
            params={"backendNodeId": bid}, session_id=sid
        )
        oid = result["object"]["objectId"]
        info = await cdp.cdp_client.send.Runtime.callFunctionOn(
            params={
                "functionDeclaration": """function(){
                    var t=this.tagName.toLowerCase();
                    var txt=(this.textContent||'').trim().substring(0,30);
                    this.click();
                    return JSON.stringify({tag:t,text:txt});
                }""",
                "objectId": oid,
                "returnByValue": True,
            },
            session_id=sid,
        )
        r = json.loads(info["result"]["value"])
        tag, text = r["tag"], r["text"]
    except Exception as e:
        return DomActionResult(summary=f"[失败] 点击 [i_{index}] 出错: {e}")
    await asyncio.sleep(0.5)
    # 点击后刷新 DOM
    dom_text: str | None = None
    try:
        dom_text = await _get_dom_state(session)
    except Exception:
        pass
    return DomActionResult(summary=f"[OK] 点击了 [i_{index}] <{tag}>{text}</{tag}>", dom=dom_text)


async def fill_input(session: BrowserSession, index: int, text: str) -> str:
    """通过 DOM 索引 [i_*] 向输入框填写文本，模拟逐字输入以触发 React/Vue 事件。"""
    node = await session.raw.get_element_by_index(index)
    if node is None:
        return f"[失败] 未找到索引 [i_{index}] 的元素，用 dom_state() 刷新"
    try:
        cdp = await session.raw.cdp_client_for_node(node)
    except Exception:
        cdp = await get_cdp(session)
    bid = node.backend_node_id
    sid = cdp.session_id
    # 滚入视图
    try:
        await cdp.cdp_client.send.DOM.scrollIntoViewIfNeeded(
            params={"backendNodeId": bid}, session_id=sid
        )
    except Exception:
        pass
    # 解析为 JS 对象
    try:
        result = await cdp.cdp_client.send.DOM.resolveNode(
            params={"backendNodeId": bid}, session_id=sid
        )
        oid = result["object"]["objectId"]
    except Exception as e:
        return f"[失败] 解析 [i_{index}] 出错: {e}"
    # 聚焦 + 清空 + 通过 CDP Input.dispatchKeyEvent 逐字输入
    await cdp.cdp_client.send.Runtime.callFunctionOn(
        params={
            "functionDeclaration": """function(){
                this.focus();
                this.value = '';
                this.dispatchEvent(new Event('focus', {bubbles:true}));
            }""",
            "objectId": oid,
        },
        session_id=sid,
    )
    # 逐字符通过 CDP 发送按键事件
    for ch in text:
        await cdp.cdp_client.send.Input.dispatchKeyEvent(
            params={"type": "keyDown", "text": ch}, session_id=sid
        )
        await cdp.cdp_client.send.Input.dispatchKeyEvent(
            params={"type": "keyUp", "text": ch}, session_id=sid
        )
    # 触发 input/change 事件确保框架绑定生效
    await cdp.cdp_client.send.Runtime.callFunctionOn(
        params={
            "functionDeclaration": """function(){
                this.dispatchEvent(new Event('input', {bubbles:true}));
                this.dispatchEvent(new Event('change', {bubbles:true}));
            }""",
            "objectId": oid,
        },
        session_id=sid,
    )
    masked = text if len(text) <= 3 else text[:2] + "*" * (len(text) - 2)
    return f"[OK] 已在 [i_{index}] 输入 '{masked}' ({len(text)}字符)"


async def get_cookies(session: BrowserSession) -> list[dict]:
    """
    提取当前浏览器会话的所有 Cookie。
    采用多级降级策略确保在不同版本的驱动下均能稳定获取。
    """
    # 50年架构师提示：Cookie 是会话的灵魂，必须保证提取的鲁棒性
    playwright_cookies = []
    try:
        if hasattr(session.raw, "get_cookies"):
            playwright_cookies = await session.raw.get_cookies()
        elif hasattr(session.raw, "_cdp_get_cookies"):
            playwright_cookies = await session.raw._cdp_get_cookies()
        elif hasattr(session.raw, "cookies"):
            cookies_attr = session.raw.cookies
            playwright_cookies = await cookies_attr() if callable(cookies_attr) else cookies_attr
    except Exception as e:
        logger.error("获取 Cookie 失败: %s", e)
    
    emit_observation(f"[get_cookies] 已提取 {len(playwright_cookies)} 条 Cookie")
    return playwright_cookies


async def get_network_log(
    session: BrowserSession,
    filter_str: str = "",
    only_new: bool = False,
) -> list:
    """读取页面拦截到的 Fetch/XHR 网络请求日志。"""
    if filter_str:
        js_code = f"""(function(){{
var log=window.__netlog||[];
var f='{filter_str}'.toLowerCase();
var r=log.filter(function(e){{return e.url.toLowerCase().indexOf(f)!==-1;}});
return JSON.stringify(r);
}})()"""
    else:
        js_code = "(function(){return JSON.stringify(window.__netlog||[]);})()"
    raw = await run_js(session, js_code)
    try:
        all_entries = json.loads(raw)
        if only_new:
            entries = all_entries[session.netlog_cursor :]
        else:
            entries = all_entries
        session.netlog_cursor = len(all_entries)
        label = "[network_log]"
        if only_new:
            label += "(new)"
        
        # 使用专业的 emit_observation 汇报给 LLM
        emit_observation(
            f"{label} {len(entries)} 条请求"
            + (f" (filter='{filter_str}')" if filter_str else "")
        )
        return entries
    except Exception:
        return []
