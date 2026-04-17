from __future__ import annotations

import asyncio
import ipaddress
import json as json_lib
import logging
import socket
import urllib.parse as _urlparse
from typing import Any

import httpx

from hawker_agent.observability import emit_observation, emit_tool_observation
from hawker_agent.tools.data_tools import clean_items, ensure, parse_http_response, summarize_json
from hawker_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_clients_by_loop: dict[int, httpx.AsyncClient] = {}


_BLOCKED_HOSTS = {
    "metadata.google.internal",
    "metadata.goog",
}


def _is_private_ip(host: str) -> bool:
    """Return True if *host* resolves to a private, loopback, or link-local address."""
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
    except ValueError:
        return False


def _default_port_for_scheme(scheme: str) -> int | None:
    """为常见协议返回默认端口，供 DNS 校验时使用。"""
    normalized = (scheme or "").lower()
    if normalized == "http":
        return 80
    if normalized == "https":
        return 443
    return None


async def _resolve_host_ips(hostname: str, port: int | None) -> set[str]:
    """解析主机名并返回去重后的 IP 集合。"""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(
        hostname,
        port,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    return {
        sockaddr[0]
        for _family, _socktype, _proto, _canonname, sockaddr in infos
        if sockaddr
    }


async def _validate_url(url: str) -> None:
    """Block requests to private networks, cloud metadata endpoints, and other reserved ranges."""
    parsed = _urlparse.urlparse(url)
    hostname = (parsed.hostname or "").lower().strip(".")
    if not hostname:
        raise ValueError(f"Invalid URL (no hostname): {url}")
    if hostname in _BLOCKED_HOSTS:
        raise ValueError(f"Blocked request to reserved host: {hostname}")
    if _is_private_ip(hostname):
        raise ValueError(f"Blocked request to private/reserved IP: {hostname}")
    try:
        resolved_ips = await _resolve_host_ips(hostname, parsed.port or _default_port_for_scheme(parsed.scheme))
    except socket.gaierror as exc:
        raise ValueError(f"Failed to resolve hostname: {hostname}") from exc
    for ip in resolved_ips:
        if _is_private_ip(ip):
            raise ValueError(f"Blocked request to private/reserved IP via DNS: {hostname} -> {ip}")


async def _get_client() -> httpx.AsyncClient:
    """按 event loop 复用 httpx 异步客户端，避免跨 loop 复用已绑定客户端。"""
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    client = _clients_by_loop.get(loop_id)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        _clients_by_loop[loop_id] = client
    return client


async def close_http_clients() -> None:
    """关闭当前进程内缓存的所有 httpx 客户端。"""
    clients = list(_clients_by_loop.values())
    _clients_by_loop.clear()
    for client in clients:
        if not client.is_closed:
            await client.aclose()


def _parse_cookies(cookies_input: dict | str | list | None) -> dict | None:
    """
    解析多种格式的 Cookie 输入，统一转换为 httpx 支持的 dict 格式。
    特别是兼容 browser-use (Playwright) 导出的 list[dict] 格式。
    """
    if not cookies_input:
        return None
    if isinstance(cookies_input, dict):
        return cookies_input
    if isinstance(cookies_input, list):
        # 处理 Playwright 格式: [{'name': 'a', 'value': '1'}, ...]
        result = {}
        for c in cookies_input:
            if isinstance(c, dict) and "name" in c and "value" in c:
                result[c["name"]] = c["value"]
        return result
    if isinstance(cookies_input, str):
        # 处理原始 Cookie 字符串: "a=1; b=2"
        result = {}
        for part in cookies_input.split(";"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                result[k.strip()] = v.strip()
        return result
    return None


def _parse_headers(headers_input: dict | str | None) -> dict[str, str]:
    """解析 headers 输入并规范化为字符串字典。"""
    if not headers_input:
        return {}
    if isinstance(headers_input, dict):
        return {str(k): str(v) for k, v in headers_input.items()}
    try:
        parsed = json_lib.loads(headers_input)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        pass
    return {}


def _build_request_payload(
    *,
    data: Any = None,
    json_payload: Any = None,
    content: str | bytes | None = None,
    legacy_body: Any = None,
) -> tuple[Any, Any, str | bytes | None]:
    """统一处理 http_request/http_json 的请求体参数。"""
    provided = [
        ("json", json_payload is not None),
        ("data", data is not None),
        ("content", content is not None),
        ("body", legacy_body is not None),
    ]
    active = [name for name, enabled in provided if enabled]
    if len(active) > 1:
        raise ValueError(f"Request payload is ambiguous; use only one of json/data/content/body, got: {', '.join(active)}")

    if json_payload is not None:
        return None, json_payload, None
    if data is not None:
        return data, None, None
    if content is not None:
        return None, None, content
    if legacy_body is None:
        return None, None, None
    if isinstance(legacy_body, (dict, list)):
        return None, legacy_body, None
    return None, None, legacy_body


async def http_request(
    url: str,
    method: str = "GET",
    headers: str | dict | None = None,
    params: dict | str | None = None,
    data: Any = None,
    json: Any = None,
    content: str | bytes | None = None,
    cookies: dict | str | list | None = None,
    **kwargs: Any
) -> str:
    """
    发送 HTTP 请求，返回完整响应文本。

    兼容说明：
    - 旧参数 `body=` 仍可用，会自动映射到 `json` 或 `content`
    - 额外的 `timeout`、`files` 等参数会透传给 httpx
    """
    legacy_body = kwargs.pop("body", None)
    h = _parse_headers(headers)
    c = _parse_cookies(cookies)
    query_params = params
    if isinstance(params, str):
        try:
            parsed_params = json_lib.loads(params)
            query_params = parsed_params if isinstance(parsed_params, dict) else params
        except Exception:
            query_params = params

    await _validate_url(url)
    client = await _get_client()
    try:
        request_data, request_json, request_content = _build_request_payload(
            data=data,
            json_payload=json,
            content=content,
            legacy_body=legacy_body,
        )
        h.setdefault("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36")
        if request_json is not None:
            h.setdefault("Accept", "application/json, text/plain, */*")
        resp = await client.request(
            method.upper(),
            url,
            headers=h,
            params=query_params,
            data=request_data,
            json=request_json,
            content=request_content,
            cookies=c,
            **kwargs,
        )
        content_type = resp.headers.get("Content-Type", "unknown").split(";")[0]
        emit_tool_observation("http_request", str(resp.status_code), f"size={len(resp.text)}", f"type={content_type}")
        return f"[{resp.status_code}]\n{resp.text}"
    except Exception as e:
        logger.exception("HTTP 请求失败: %s %s", method.upper(), url)
        return f"[错误] {e}"


async def http_json(
    url: str,
    method: str = "GET",
    headers: str | dict | None = None,
    params: dict | str | None = None,
    data: Any = None,
    json: Any = None,
    content: str | bytes | None = None,
    cookies: dict | str | list | None = None,
    **kwargs: Any
) -> Any:
    """
    发送 HTTP 请求并返回解析后的 JSON 对象。
    """
    raw = await http_request(
        url,
        method=method,
        headers=headers,
        params=params,
        data=data,
        json=json,
        content=content,
        cookies=cookies,
        **kwargs,
    )
    status, text = parse_http_response(raw)
    ensure(200 <= status < 300, f"HTTP {status}: {text[:200]}")
    parsed = json_lib.loads(text)
    if isinstance(parsed, list):
        parsed = clean_items(parsed)

    count = len(parsed) if isinstance(parsed, (list, dict)) else 1
    emit_tool_observation("http_json", "OK", f"items={count}", summarize_json(parsed))
    return parsed


async def fetch(
    url: str,
    method: str = "GET",
    parse: str = "json",
    headers: str | dict | None = None,
    params: dict | str | None = None,
    data: Any = None,
    json: Any = None,
    content: str | bytes | None = None,
    cookies: dict | str | list | None = None,
    **kwargs: Any,
) -> Any:
    """统一 HTTP 请求入口，优先使用此工具请求接口数据。"""
    normalized_parse = (parse or "json").lower()
    if normalized_parse == "json":
        return await http_json(
            url,
            method=method,
            headers=headers,
            params=params,
            data=data,
            json=json,
            content=content,
            cookies=cookies,
            **kwargs,
        )
    if normalized_parse in {"text", "raw"}:
        return await http_request(
            url,
            method=method,
            headers=headers,
            params=params,
            data=data,
            json=json,
            content=content,
            cookies=cookies,
            **kwargs,
        )
    raise ValueError(f"fetch(parse=...) 仅支持 json / text / raw，收到: {parse}")


def register_http_tools(registry: ToolRegistry) -> None:
    """注册 HTTP 工具到 registry。"""
    registry.register(fetch, category="网络 & 数据")
    registry.register(http_request, category="网络 & 数据", expose_in_prompt=False)
    registry.register(http_json, category="网络 & 数据", expose_in_prompt=False)
