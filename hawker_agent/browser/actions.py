from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from curl_cffi import AsyncSession

from hawker_agent.browser.cdp import get_cdp, run_js
from hawker_agent.browser.dom_utils import (
    build_dom_snapshot,
    render_dom_diff,
)
from hawker_agent.observability import emit_observation, emit_tool_observation
from hawker_agent.tools.data_tools import get_type_signature

if TYPE_CHECKING:
    from hawker_agent.browser.session import BrowserSession

logger = logging.getLogger(__name__)


def _safe_download_filename(value: str, fallback: str = "download") -> str:
    """返回适合本地保存的单文件名。"""
    name = Path(value).name or fallback
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def _download_destination(
    session: BrowserSession,
    *,
    url: str,
    filename: str | None,
    run_dir: str | None,
) -> Path:
    target_dir_value = run_dir or (str(session.target_dir) if getattr(session, "target_dir", None) else None)
    target_dir = Path(target_dir_value) if target_dir_value else Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)
    fallback_name = Path(urllib.parse.urlparse(url).path).name or "download"
    desired_name = _safe_download_filename(filename or fallback_name)
    return target_dir / desired_name


def _filename_from_content_disposition(value: str) -> str | None:
    match = re.search(r'filename\*=UTF-8\'\'([^;\n]+)', value, re.I)
    if match:
        return urllib.parse.unquote(match.group(1).strip().strip('"'))
    match = re.search(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', value, re.I)
    if match:
        return match.group(1).strip().strip("'\"")
    return None


async def _browser_user_agent(session: BrowserSession) -> str:
    try:
        page = await session.raw.get_current_page()
        if page is not None:
            value = await page.evaluate("() => navigator.userAgent")
            if value:
                return value
    except Exception:
        logger.debug("读取浏览器 User-Agent 失败", exc_info=True)
    return "Mozilla/5.0 (compatible; HawkerAgent/1.0)"


async def _download_with_browser_session_http(
    session: BrowserSession,
    *,
    url: str,
    filename: str | None,
    run_dir: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    """使用浏览器会话 Cookie/UA 与 Chrome TLS 指纹下载并直接落盘。"""
    parsed = urllib.parse.urlparse(url)
    cookies = await get_cookies(session, domain=parsed.hostname or "", verbose=False)
    cookie_map = {
        str(cookie.get("name")): str(cookie.get("value"))
        for cookie in cookies
        if cookie.get("name") and cookie.get("value") is not None
    }
    headers = {
        "User-Agent": await _browser_user_agent(session),
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": f"{parsed.scheme}://{parsed.netloc}/" if parsed.scheme and parsed.netloc else url,
    }
    destination = _download_destination(session, url=url, filename=filename, run_dir=run_dir)
    async with AsyncSession(impersonate="chrome120", timeout=timeout_s) as client:
        async with client.stream(
            "GET",
            url,
            headers=headers,
            cookies=cookie_map,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            if filename is None:
                header_name = _filename_from_content_disposition(
                    response.headers.get("content-disposition", "")
                )
                if header_name:
                    destination = destination.with_name(_safe_download_filename(header_name))
            if destination.exists():
                destination = destination.with_name(f"{destination.stem}_dup{destination.suffix}")
            temp_path = destination.with_suffix(destination.suffix + ".part")
            total = 0
            with temp_path.open("wb") as fh:
                async for chunk in response.aiter_content():
                    if not chunk:
                        continue
                    fh.write(chunk)
                    total += len(chunk)
            if total <= 0:
                temp_path.unlink(missing_ok=True)
                raise RuntimeError("empty download body")
            temp_path.replace(destination)

    if not destination.exists() or destination.stat().st_size <= 0:
        raise FileNotFoundError(f"browser session HTTP post-check failed: {destination}")
    return {
        "ok": True,
        "url": url,
        "requested_filename": filename or destination.name,
        "filename": destination.name,
        "path": str(destination),
        "size": destination.stat().st_size,
        "method": "curl_cffi",
    }


@dataclass
class DomActionResult:
    """浏览器动作执行结果。

    包含动作摘要，以及根据上下文模式可选的完整 DOM 状态或快照。

    Attributes:
        summary (str): 动作结果的简短描述。
        dom (str | None): DOM 内容，取决于 context_mode。默认为 None。
        snapshot (dict | None): 表示页面快照的字典。默认为 None。
        context_mode (str): DOM 表示模式（summary, diff, full）。默认为 "summary"。
    """

    summary: str
    dom: str | None = None
    snapshot: dict | None = None
    context_mode: str = "summary"


def _escape_js_string(value: str) -> str:
    """转义字符串以便安全地插入到单引号 JS 字符串字面量中。

    Args:
        value (str): 要转义的字符串。

    Returns:
        str: 转义后的字符串。
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")


# ─── 私有辅助 ──────────────────────────────────────────────────


async def _capture_dom_state(session: BrowserSession) -> tuple[str, dict]:
    """捕获当前页面的完整 DOM 文本和语义快照。

    Args:
        session (BrowserSession): 浏览器会话对象。

    Returns:
        tuple[str, dict]: 包含完整 DOM 文本和语义快照字典的元组。

    Raises:
        Exception: 当获取浏览器状态失败时，将降级到简单模式并返回基础信息。
    """
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

        # 3 压缩重复的空白
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

        full_dom = "\n".join(lines)
        snapshot = build_dom_snapshot(
            title=state.title,
            url=state.url,
            dom_repr=dom_repr,
            pages_above=pages_above,
            pages_below=pages_below,
            pending_requests=len(state.pending_network_requests or []),
            tabs=len(state.tabs),
        )
        return full_dom, snapshot
    except Exception as e:
        # 降级到简单模式
        raw = await run_js(
            session,
            "(function(){return JSON.stringify({title:document.title,"
            "text:document.body?document.body.innerText.length:0})})()",
        )
        try:
            info = json.loads(raw)
            title = info.get("title", "")
            fallback_text = (
                f"[OK] {title} | "
                f"文本:{info.get('text', 0)}字符 (DOM服务不可用: {e})"
            )
            snapshot = build_dom_snapshot(title=title, url="", dom_repr="")
            return fallback_text, snapshot
        except Exception:
            fallback_text = f"[OK] 已导航 (DOM服务不可用: {e})"
            snapshot = build_dom_snapshot(title="", url="", dom_repr="")
            return fallback_text, snapshot


async def _capture_navigation_meta(session: BrowserSession) -> dict[str, Any]:
    """只读取最轻量的页面元信息，不构建 DOM summary。"""
    try:
        raw = await run_js(
            session,
            "(function(){return JSON.stringify({"
            "title: document.title || '',"
            "url: location.href || ''"
            "})})()",
        )
        info = json.loads(raw)
        return {
            "title": str(info.get("title", "") or ""),
            "url": str(info.get("url", "") or ""),
        }
    except Exception:
        logger.debug("读取轻量页面元信息失败", exc_info=True)
        return {"title": "", "url": ""}


def _build_dom_action_result(
    *,
    full_dom: str,
    snapshot: dict,
    mode: str,
    previous_snapshot: dict | None = None,
) -> DomActionResult:
    """根据提供的参数构建 DomActionResult。

    Args:
        full_dom (str): 完整的 DOM 内容。
        snapshot (dict): 当前页面快照。
        mode (str): 上下文模式（summary, diff, full）。
        previous_snapshot (dict | None, optional): 用于差异对比的上一个页面快照。默认为 None。

    Returns:
        DomActionResult: 包含摘要和可选 DOM 上下文的对象。
    """
    mode = (mode or "summary").lower()
    title = snapshot.get("title") or "(无标题)"
    interactive_count = snapshot.get("interactive_count", 0)

    if mode == "skip":
        context = None
        summary = f"[OK] {title} | DOM=skipped"
    elif mode == "full":
        context = full_dom
        summary = f"[OK] {title} | 交互元素 {interactive_count} | DOM=full"
    elif mode == "diff":
        context = render_dom_diff(previous_snapshot, snapshot)
        summary = f"[OK] {title} | 交互元素 {interactive_count} | DOM=diff"
    else:
        context = None
        summary = f"[OK] {title} | 交互元素 {interactive_count} | DOM=summary"

    return DomActionResult(
        summary=summary,
        dom=context,
        snapshot=snapshot,
        context_mode=mode,
    )


_JS_RAW_MAX_CHARS = 10_000
_JS_SAMPLE_CHARS = 2_000


def _log_js_summary(data: Any) -> None:
    """记录 JavaScript 执行结果的摘要。

    Args:
        data (Any): JS 执行后的原始数据对象。
    """
    if isinstance(data, str) and data.startswith("[JS错误]"):
        emit_observation(data[:200])
        return
    
    if isinstance(data, list):
        sample = ""
        sig = ""
        if data:
            sample_obj = data[0]
            sample = json.dumps(sample_obj, ensure_ascii=False)
            if len(sample) > 120:
                sample = sample[:120] + "..."
            if isinstance(sample_obj, dict):
                sig = get_type_signature(sample_obj)
        emit_observation(
            f"[js] 返回 {len(data)} 条数据"
            + (f" | 签名: {sig}" if sig else "")
            + (f" | 样本: {sample}" if sample else "")
        )
    elif isinstance(data, dict):
        emit_observation(f"[js] 返回 dict, {len(data)} 个键: {get_type_signature(data)}")
    else:
        emit_observation(f"[js] 返回: {str(data)[:120]}")


def _truncate_js_raw(data: Any, max_chars: int = _JS_RAW_MAX_CHARS) -> Any:
    """若 js() 返回的原始文本过长，则折叠为带提示的结构化摘要。

    针对未能解析成 JSON、直接被吐出的超长字符串进行兜底，避免把整份
    HTML / JS 源码 / base64 塞进 Observation。
    """
    if not isinstance(data, str):
        return data
    if len(data) <= max_chars:
        return data
    sample = data[:_JS_SAMPLE_CHARS]
    return {
        "_truncated": True,
        "len": len(data),
        "sample": sample,
        "hint": (
            f"js() 返回了 {len(data)} 字符的原始文本，已折叠为前 {len(sample)} 字符样本。"
            "建议在 JS 内先用 JSON.stringify 结构化提取所需字段，或按页分片返回。"
        ),
    }


def _build_search_url(query: str, engine: str) -> str | None:
    """构建搜索引擎查询的 URL。

    Args:
        query (str): 搜索关键词。
        engine (str): 搜索引擎名称（duckduckgo, google, bing）。

    Returns:
        str | None: 格式化后的搜索 URL，如果引擎不支持则返回 None。
    """
    encoded = urllib.parse.quote_plus(query)
    urls = {
        "duckduckgo": f"https://duckduckgo.com/?q={encoded}",
        "google": f"https://www.google.com/search?q={encoded}&udm=14",
        "bing": f"https://www.bing.com/search?q={encoded}",
    }
    return urls.get(engine.lower())


# ─── 公开动作 ──────────────────────────────────────────────────


async def nav(
    session: BrowserSession,
    url: str,
    mode: str = "summary",
    previous_snapshot: dict | None = None,
) -> DomActionResult:
    """导航到指定 URL，并按模式返回页面上下文。

    Args:
        session (BrowserSession): 浏览器会话对象。
        url (str): 目标页面 URL。
        mode (str, optional): 上下文模式，支持 "skip"、"summary"、"diff"、"full"。默认为 "summary"。
        previous_snapshot (dict | None, optional): 上一个页面的快照，用于 "diff" 模式。默认为 None。

    Returns:
        DomActionResult: 包含导航结果摘要和可选页面上下文的对象。
            ``summary`` 末尾会附加 URL 变化/重定向事实，例如
            ``| URL=未变`` 或 ``| URL 已变(redirected): <final>``，方便 Agent
            立即感知登录墙跳转、SPA 路由等关键事件。
    """
    await session.raw.navigate_to(url)
    await asyncio.sleep(1)
    if (mode or "summary").lower() == "skip":
        meta = await _capture_navigation_meta(session)
        snapshot = build_dom_snapshot(
            title=meta.get("title", ""),
            url=meta.get("url", ""),
            dom_repr="",
        )
        result = _build_dom_action_result(
            full_dom="",
            snapshot=snapshot,
            mode="skip",
            previous_snapshot=previous_snapshot,
        )
    else:
        full_dom, snapshot = await _capture_dom_state(session)
        result = _build_dom_action_result(
            full_dom=full_dom,
            snapshot=snapshot,
            mode=mode,
            previous_snapshot=previous_snapshot,
        )
    final_url = snapshot.get("url") or ""
    changed = bool(final_url) and _urls_differ(url, final_url)
    if changed:
        result.summary = f"{result.summary} | URL 已变(redirected): {final_url}"
    elif final_url:
        result.summary = f"{result.summary} | URL=未变"
    if snapshot is not None:
        snapshot["requested_url"] = url
        snapshot["redirected"] = changed
    return result


def _urls_differ(requested: str, final: str) -> bool:
    """判断导航前后 URL 是否发生实质变化（忽略末尾斜杠）。"""
    if not requested or not final:
        return False
    req = requested.split("#", 1)[0].rstrip("/")
    fin = final.split("#", 1)[0].rstrip("/")
    return req != fin


async def dom_state(
    session: BrowserSession,
    mode: str = "summary",
    previous_snapshot: dict | None = None,
) -> DomActionResult:
    """获取当前页面状态，并按模式返回页面上下文。

    Args:
        session (BrowserSession): 浏览器会话对象。
        mode (str, optional): 上下文模式，支持 "summary"、"diff"、"full"。默认为 "summary"。
        previous_snapshot (dict | None, optional): 上一个页面的快照，用于 "diff" 模式。默认为 None。

    Returns:
        DomActionResult: 包含当前页面状态摘要和可选页面上下文的对象。
    """
    full_dom, snapshot = await _capture_dom_state(session)
    return _build_dom_action_result(
        full_dom=full_dom,
        snapshot=snapshot,
        mode=mode,
        previous_snapshot=previous_snapshot,
    )


async def nav_search(
    session: BrowserSession,
    query: str,
    engine: str = "google",
    mode: str = "summary",
    previous_snapshot: dict | None = None,
) -> DomActionResult:
    """使用搜索引擎进行检索，并返回搜索结果页的状态。

    Args:
        session (BrowserSession): 浏览器会话对象。
        query (str): 搜索关键词。
        engine (str, optional): 搜索引擎名称。默认为 "google"。
        mode (str, optional): 上下文模式，支持 "summary"、"diff"、"full"。默认为 "summary"。
        previous_snapshot (dict | None, optional): 上一个页面的快照。默认为 None。

    Returns:
        DomActionResult: 包含搜索结果页摘要和可选页面上下文的对象。
    """
    url = _build_search_url(query, engine)
    if not url:
        return DomActionResult(
            summary=f"[错误] 不支持的搜索引擎: {engine}，可用: duckduckgo, google, bing"
        )
    return await nav(session, url, mode=mode, previous_snapshot=previous_snapshot)

async def browser_download(
    session: BrowserSession,
    url: str,
    filename: str | None = None,
    run_dir: str | None = None,
    timeout_s: float = 90.0,
    attempts: int = 3,
    retry_delay_s: float = 1.5,
) -> dict[str, Any]:
    """使用浏览器原生下载能力下载文件，并等待文件落地。

    Args:
        session (BrowserSession): 浏览器会话对象。
        url (str): 下载链接。
        filename (str | None): 期望保存的文件名。
        run_dir (str | None): 目标目录；若提供则下载完成后立即归档到该目录。
        timeout_s (float): 最长等待下载完成时间。

    Returns:
        dict[str, Any]: 结构化下载结果。
    """
    total_attempts = max(1, attempts)
    for attempt in range(1, total_attempts + 1):
        try:
            result = await _download_with_browser_session_http(
                session,
                url=url,
                filename=filename,
                run_dir=run_dir,
                timeout_s=timeout_s,
            )
            emit_tool_observation(
                "browser_download",
                "OK",
                f"method=curl_cffi attempt={attempt}/{total_attempts} size={result['size']}",
                f"file={result['filename']}",
            )
            return result
        except Exception as exc:
            emit_tool_observation(
                "browser_download",
                "RETRY" if attempt < total_attempts else "FAILED",
                f"method=curl_cffi reason={type(exc).__name__} attempt={attempt}/{total_attempts}",
                f"url={url}",
            )
            if attempt < total_attempts:
                await asyncio.sleep(retry_delay_s)
                continue
            raise exc

    raise RuntimeError(f"browser_download failed: {url}")


async def js(session: BrowserSession, code: str) -> Any:
    """在当前页面执行 JavaScript 代码，并返回完整执行结果。

    当返回值是超长字符串（>10KB）时，会自动折叠为
    ``{"_truncated": True, "len": N, "sample": "...", "hint": "..."}``，
    避免整份 HTML / 源码 / base64 直接塞进 Observation。如果是列表或 dict
    则按原值返回（由调用方自行处理更大结构）。

    Args:
        session (BrowserSession): 浏览器会话对象。
        code (str): 要执行的 JavaScript 代码。

    Returns:
        Any: JS 执行后的原生 Python 对象（List, Dict, Str, etc.）。
    """
    raw = await run_js(session, code)
    payload = _truncate_js_raw(raw)
    _log_js_summary(payload if isinstance(payload, (list, dict)) else raw)
    return payload


async def click(
    session: BrowserSession,
    selector: str,
    index: int = 0,
    mode: str = "summary",
    previous_snapshot: dict | None = None,
) -> DomActionResult:
    """点击匹配 CSS 选择器的元素，并按模式返回页面上下文。

    Args:
        session (BrowserSession): 浏览器会话对象。
        selector (str): CSS 选择器。
        index (int, optional): 匹配到的元素列表中的索引（从 0 开始）。默认为 0。
        mode (str, optional): 上下文模式，支持 "summary"、"diff"、"full"。默认为 "summary"。
        previous_snapshot (dict | None, optional): 上一个页面的快照。默认为 None。

    Returns:
        DomActionResult: 包含点击动作结果摘要和可选页面上下文的对象。
    """
    safe_selector = _escape_js_string(selector)
    raw = await run_js(
        session,
        f"""(function(){{
        var els=document.querySelectorAll('{safe_selector}');
        if(els.length===0) return JSON.stringify({{error:"未找到元素: {safe_selector}"}});
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
        # 点击后页面可能变化，刷新页面快照
        try:
            full_dom, snapshot = await _capture_dom_state(session)
        except Exception:
            summary = f"[OK] 点击了 <{r['tag']}>{r['text']}</{r['tag']}> (共{r['total']}个匹配)"
            return DomActionResult(summary=summary)
        result = _build_dom_action_result(
            full_dom=full_dom,
            snapshot=snapshot,
            mode=mode,
            previous_snapshot=previous_snapshot,
        )
        result.summary = (
            f"[OK] 点击了 <{r['tag']}>{r['text']}</{r['tag']}> (共{r['total']}个匹配)"
            f" | DOM={mode}"
        )
        return result
    except Exception:
        return DomActionResult(summary=f"[OK] 已点击 {selector}")


async def click_index(
    session: BrowserSession,
    index: int,
    mode: str = "summary",
    previous_snapshot: dict | None = None,
) -> DomActionResult:
    """通过 DOM 索引点击元素，并按模式返回页面上下文。

    Args:
        session (BrowserSession): 浏览器会话对象。
        index (int): 元素的 DOM 索引值。
        mode (str, optional): 上下文模式，支持 "summary"、"diff"、"full"。默认为 "summary"。
        previous_snapshot (dict | None, optional): 上一个页面的快照。默认为 None。

    Returns:
        DomActionResult: 包含点击动作结果摘要和可选页面上下文的对象。
    """
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
    # 点击后刷新页面快照
    try:
        full_dom, snapshot = await _capture_dom_state(session)
    except Exception:
        return DomActionResult(summary=f"[OK] 点击了 [i_{index}] <{tag}>{text}</{tag}>")
    result = _build_dom_action_result(
        full_dom=full_dom,
        snapshot=snapshot,
        mode=mode,
        previous_snapshot=previous_snapshot,
    )
    result.summary = f"[OK] 点击了 [i_{index}] <{tag}>{text}</{tag}> | DOM={mode}"
    return result


async def fill_input(session: BrowserSession, index: int, text: str) -> str:
    """通过 DOM 索引向输入框填写文本。

    模拟逐字输入以触发 React/Vue 等前端框架的事件。

    Args:
        session (BrowserSession): 浏览器会话对象。
        index (int): 目标输入框的 DOM 索引。
        text (str): 要填写的文本。

    Returns:
        str: 动作执行结果的描述字符串。
    """
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


def _cookie_domain_matches(cookie_domain: str, needle: str) -> bool:
    """以"同源或子域"的方式判断 cookie domain 是否匹配过滤条件。"""
    if not needle:
        return True
    cookie = (cookie_domain or "").lower().lstrip(".")
    target = needle.lower().lstrip(".")
    if not cookie:
        return False
    return cookie == target or cookie.endswith("." + target)


def _project_cookie(cookie: dict[str, Any], *, verbose: bool) -> dict[str, Any]:
    """按 verbose 开关裁剪单条 cookie 字段。"""
    if verbose:
        return cookie
    slim: dict[str, Any] = {}
    for key in ("name", "value", "domain", "path"):
        if key in cookie:
            slim[key] = cookie[key]
    return slim


async def get_cookies(
    session: BrowserSession,
    *,
    domain: str = "",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """提取当前浏览器会话的 Cookie，可按域名过滤。

    采用多级降级策略确保在不同版本的驱动下均能稳定获取。默认只返回
    ``name/value/domain/path`` 以降低 Token 熵；如需完整字段（secure、
    httpOnly、sameSite 等）请显式传 ``verbose=True``。

    Args:
        session (BrowserSession): 浏览器会话对象。
        domain (str, optional): 仅返回 ``cookie.domain`` 等于或属于该域子域
            的条目。例如 ``domain="example.com"`` 会命中 ``.example.com``、
            ``api.example.com``。默认为 ""（不过滤）。
        verbose (bool, optional): 是否返回 Cookie 的全部字段。默认为 False。

    Returns:
        list[dict[str, Any]]: 筛选后的 Cookie 字典列表。
    """
    playwright_cookies: list[dict[str, Any]] = []
    try:
        if hasattr(session.raw, "get_cookies"):
            playwright_cookies = await session.raw.get_cookies()
        elif hasattr(session.raw, "_cdp_get_cookies"):
            playwright_cookies = await session.raw._cdp_get_cookies()
        elif hasattr(session.raw, "cookies"):
            cookies_attr = session.raw.cookies
            playwright_cookies = (
                await cookies_attr() if callable(cookies_attr) else cookies_attr
            )
    except Exception as e:
        logger.error("获取 Cookie 失败: %s", e)

    total = len(playwright_cookies)
    if domain:
        playwright_cookies = [
            c
            for c in playwright_cookies
            if isinstance(c, dict) and _cookie_domain_matches(str(c.get("domain") or ""), domain)
        ]
    projected = [
        _project_cookie(c, verbose=verbose) for c in playwright_cookies if isinstance(c, dict)
    ]

    details = [f"{len(projected)} 条"]
    if domain:
        details.append(f"domain~'{domain}' (全部 {total})")
    if not verbose:
        details.append("字段=slim(name,value,domain,path)")
    emit_observation(f"[get_cookies] 已提取 {' | '.join(details)}")
    return projected
