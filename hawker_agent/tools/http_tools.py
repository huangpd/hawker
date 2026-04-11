from __future__ import annotations

import json
import logging

import httpx

from hawker_agent.observability import emit_observation
from hawker_agent.tools.data_tools import clean_items, ensure, parse_http_response, summarize_json
from hawker_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """延迟创建 httpx 异步客户端单例。"""
    global _client  # noqa: PLW0603
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    return _client


async def http_request(
    url: str, method: str = "GET", headers: str = "", body: str = ""
) -> str:
    """发送 HTTP 请求，返回完整响应文本，并自动打印摘要。"""
    h: dict[str, str] = {}
    if headers:
        try:
            h = json.loads(headers)
        except Exception:
            pass
    h.setdefault("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                               " AppleWebKit/537.36 (KHTML, like Gecko) Chrome/"
                               "146.0.0.0 Safari/537.36")

    client = _get_client()
    try:
        if method.upper() == "POST":
            resp = await client.post(url, headers=h, content=body)
        else:
            resp = await client.get(url, headers=h)
        content_type = resp.headers.get("Content-Type", "unknown").split(";")[0]
        
        # 使用专业的 emit_observation
        emit_observation(
            f"[http_request] {resp.status_code} | {len(resp.text)} 字符 | {content_type}"
        )
        return f"[{resp.status_code}]\n{resp.text}"
    except Exception as e:
        logger.exception("HTTP 请求失败: %s %s", method.upper(), url)
        return f"[错误] {e}"


async def http_json(
    url: str, method: str = "GET", headers: str = "", body: str = ""
) -> object:
    """发送 HTTP 请求并返回解析后的 JSON 对象。"""
    raw = await http_request(url, method, headers, body)
    status, text = parse_http_response(raw)
    ensure(200 <= status < 300, f"HTTP {status}: {text[:200]}")
    data = json.loads(text)
    if isinstance(data, list):
        data = clean_items(data)
    
    # 使用专业的 emit_observation
    emit_observation(summarize_json(data))
    return data


def register_http_tools(registry: ToolRegistry) -> None:
    """注册 HTTP 工具到 registry。"""
    registry.register(http_request)
    registry.register(http_json)
