from __future__ import annotations

import asyncio
import ipaddress
import json as json_lib
import logging
import socket
import urllib.parse as _urlparse
from typing import Any

import httpx

from hawker_agent.observability import emit_tool_observation
from hawker_agent.tools.data_tools import clean_items, ensure, parse_http_response, summarize_json
from hawker_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_clients_by_loop: dict[int, httpx.AsyncClient] = {}


_BLOCKED_HOSTS = {
    "metadata.google.internal",
    "metadata.goog",
}


# ─── ACI：响应截断与语义化错误映射 ──────────────────────────────

HTTP_RESPONSE_MAX_CHARS = 20_000
HTTP_JSON_MAX_ITEMS = 100


_STATUS_HINTS: dict[int, str] = {
    400: "请求被服务端拒绝(400)。请检查 URL/params/body 是否符合接口协议；常见原因：字段缺失、类型不匹配、签名错误。",
    401: "未授权(401)。请先通过浏览器登录目标站点，然后用 `get_cookies(domain=...)` 或 `inspect_page(include=['network'])` 从最近的真实请求中复刻 Cookie/Authorization 头，再重试。",
    403: "访问被拒(403)。可能是缺少 Referer / Origin / 反爬 token，或账号无权限。建议用 `inspect_page(include=['network'])` 查看一条成功请求的全部 headers 后复刻。",
    404: "资源不存在(404)。确认 URL 路径、路径参数和接口版本号是否正确，或站点是否已改版。",
    405: "方法不被允许(405)。对照接口文档切换 GET/POST；若无文档，可从网络日志中找到原始方法名。",
    408: "请求超时(408)。可增大 timeout 或减小 batch，避免一次抓太多。",
    409: "资源冲突(409)。通常是并发写入或重复提交引起；重试前请先拉取最新状态。",
    410: "资源已下线(410)。别再重试这个 URL，改换其它入口。",
    413: "请求体过大(413)。请减少单次发送的数据量（如减少 page_size）。",
    415: "不支持的媒体类型(415)。检查 Content-Type 是否与 body 编码一致（JSON 应为 application/json）。",
    418: "服务端明确拒绝爬虫(418 I'm a teapot)。不要对同一 URL 继续重试。",
    422: "请求被语义校验拒绝(422)。请对照响应体中的字段级错误逐项修正。",
    429: "被限流(429)。请指数退避（asyncio.sleep(min(60, 2**attempt))）或降低并发；如有 Retry-After 头请严格遵守。",
    500: "服务端内部错误(500)。先退避重试 1-2 次；若持续 5xx，请换入口或等待一段时间。",
    502: "网关错误(502)。通常是上游挂了，退避重试；持续失败则考虑切换节点/域名。",
    503: "服务不可用(503)。通常是限流或维护；按 Retry-After 退避，切勿高频重试。",
    504: "网关超时(504)。减小批量/页大小、增大 timeout，再退避重试。",
}


def _status_hint_for(code: int) -> str | None:
    """根据 HTTP 状态码给出行动建议，若无精确匹配则按区间兜底。"""
    if code in _STATUS_HINTS:
        return _STATUS_HINTS[code]
    if 500 <= code < 600:
        return _STATUS_HINTS[500]
    if 400 <= code < 500:
        return _STATUS_HINTS[400]
    return None


def _classify_exception(exc: BaseException) -> str:
    """把底层网络异常转化为面向 Agent 的行动建议。"""
    if isinstance(exc, httpx.ConnectTimeout):
        return "连接超时。请检查域名可达性，或增大 timeout。对方可能已下线 / 被墙。"
    if isinstance(exc, httpx.ReadTimeout):
        return "读取超时。接口响应过慢，建议减小批量或增大 timeout。"
    if isinstance(exc, httpx.TimeoutException):
        return "请求超时。建议增大 timeout 后重试，或减小本次请求的数据量。"
    if isinstance(exc, httpx.ConnectError):
        return "连接失败。可能是域名解析失败 / 对方拒绝 / 网络不可达，请确认 URL 正确且网络可达。"
    if isinstance(exc, httpx.TooManyRedirects):
        return "重定向过多。可能被鉴权墙拦截；请先携带登录态（Cookie）再重试。"
    if isinstance(exc, httpx.HTTPError):
        return "HTTP 协议错误。请复查 URL/headers/body 格式。"
    return ""


def _truncate_http_response(text: str, *, limit: int = HTTP_RESPONSE_MAX_CHARS) -> tuple[str, bool]:
    """截断 http_request 响应文本，返回 (body, truncated?)。"""
    if len(text) <= limit:
        return text, False
    sample = text[:limit]
    tail = f"\n... [截断，共 {len(text)} 字符，已保留前 {limit}]"
    return sample + tail, True


def _traverse_pick_path(data: Any, path: str) -> Any:
    """按 ``a.b.0.c`` 风格路径钻取 JSON；找不到时抛 KeyError 方便外层转化提示。"""
    current: Any = data
    tokens = [t for t in path.split(".") if t != ""]
    for token in tokens:
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(f"pick path 找不到键 '{token}' (当前层是 dict, 键: {list(current.keys())[:8]})")
            current = current[token]
            continue
        if isinstance(current, list):
            try:
                idx = int(token)
            except ValueError as e:
                raise KeyError(f"pick path 需要数字索引，但得到 '{token}' (当前层是 list, 长度 {len(current)})") from e
            if idx < -len(current) or idx >= len(current):
                raise KeyError(f"pick path 索引 {idx} 越界 (list 长度 {len(current)})")
            current = current[idx]
            continue
        raise KeyError(f"pick path '{token}' 无法在 {type(current).__name__} 上下钻")
    return current


def _traverse_json_pointer(data: Any, pointer: str) -> Any:
    """RFC 6901 JSON Pointer 访问，如 ``/data/items/0``。"""
    if pointer in ("", "/"):
        return data
    if not pointer.startswith("/"):
        raise KeyError(f"json_pointer 必须以 '/' 开头，收到 '{pointer}'")
    current: Any = data
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(f"json_pointer 找不到键 '{token}' (当前层是 dict, 键: {list(current.keys())[:8]})")
            current = current[token]
            continue
        if isinstance(current, list):
            try:
                idx = int(token)
            except ValueError as e:
                raise KeyError(f"json_pointer 需要数字索引，但得到 '{token}' (当前层是 list, 长度 {len(current)})") from e
            if idx < 0 or idx >= len(current):
                raise KeyError(f"json_pointer 索引 {idx} 越界 (list 长度 {len(current)})")
            current = current[idx]
            continue
        raise KeyError(f"json_pointer '{token}' 无法在 {type(current).__name__} 上下钻")
    return current


def _truncate_json_payload(value: Any, *, max_items: int) -> tuple[Any, bool]:
    """对 JSON 列表做长度截断，附带 ``_truncated`` 提示；dict 保持原样。"""
    if max_items <= 0:
        return value, False
    if isinstance(value, list) and len(value) > max_items:
        head = value[:max_items]
        dropped = len(value) - max_items
        return (
            head
            + [
                {
                    "_truncated": True,
                    "hint": f"列表已截断：保留前 {max_items} 条，丢弃 {dropped} 条。需要更多请分页重放。",
                }
            ],
            True,
        )
    return value, False


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
    *,
    max_chars: int = HTTP_RESPONSE_MAX_CHARS,
    **kwargs: Any,
) -> str:
    """
    发送 HTTP 请求，返回完整响应文本。

    面向 Agent 的改进：
    - 响应体超过 ``max_chars`` 字符时自动截断并追加 ``[截断，共 N 字符]`` 尾巴，
      避免整份 HTML/JSON 吃掉上下文。
    - 非 2xx 响应会附加 ``\n[hint] ...`` 行动建议（401/403/429/5xx 等）。
    - 底层异常（超时/连接失败/重定向过多）不会再把原始 traceback 扔给模型，
      而是翻译成 ``[错误] {reason}\n[hint] {建议}`` 的语义摘要。

    兼容说明：
    - 旧参数 ``body=`` 仍可用，会自动映射到 ``json`` 或 ``content``。
    - 额外的 ``timeout``、``files`` 等参数会透传给 httpx。
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

    try:
        await _validate_url(url)
    except ValueError as e:
        emit_tool_observation("http_request", "URL_BLOCKED", f"url={url}")
        return f"[错误] {e}\n[hint] 目标地址被安全策略拒绝，请确认 URL 是公网可访问地址。"

    try:
        request_data, request_json, request_content = _build_request_payload(
            data=data,
            json_payload=json,
            content=content,
            legacy_body=legacy_body,
        )
    except ValueError as e:
        emit_tool_observation("http_request", "BAD_PAYLOAD")
        return (
            f"[错误] {e}\n"
            "[hint] 请求体只接受一个来源：json=/data=/content=/body= 任选其一。"
        )

    client = await _get_client()
    try:
        h.setdefault(
            "User-Agent",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
        )
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
    except Exception as e:
        logger.exception("HTTP 请求失败: %s %s", method.upper(), url)
        hint = _classify_exception(e) or "建议检查 URL / 网络连通性 / 超时设置后重试。"
        emit_tool_observation("http_request", "EXCEPTION", f"err={type(e).__name__}")
        return f"[错误] {type(e).__name__}: {e}\n[hint] {hint}"

    content_type = resp.headers.get("Content-Type", "unknown").split(";")[0]
    body_text, truncated = _truncate_http_response(resp.text, limit=max_chars)
    status_note = str(resp.status_code)
    metrics = f"size={len(resp.text)}"
    if truncated:
        metrics += f",truncated->{max_chars}"
    emit_tool_observation("http_request", status_note, metrics, f"type={content_type}")
    rendered = f"[{resp.status_code}]\n{body_text}"
    if resp.status_code >= 400:
        hint = _status_hint_for(resp.status_code)
        if hint:
            rendered += f"\n[hint] {hint}"
    return rendered


async def http_json(
    url: str,
    method: str = "GET",
    headers: str | dict | None = None,
    params: dict | str | None = None,
    data: Any = None,
    json: Any = None,
    content: str | bytes | None = None,
    cookies: dict | str | list | None = None,
    *,
    pick: str | None = None,
    json_pointer: str | None = None,
    max_items: int = HTTP_JSON_MAX_ITEMS,
    **kwargs: Any,
) -> Any:
    """发送 HTTP 请求并返回解析后的 JSON 对象。

    面向 Agent 的改进：
    - ``pick="data.items"`` 直接在 Python 层钻取字段路径（点号分隔，索引用整数），
      失败时会抛出可读 ValueError，包含 ``可用键``/``索引越界`` 等线索。
    - ``json_pointer="/data/items/0"`` 支持 RFC 6901 JSON Pointer，适合字段含点号
      或需要精准路径的场景。
    - ``max_items`` 默认 100：若顶层是 list 且长度超限，则保留前 ``max_items`` 条，
      并在末尾追加一条 ``{"_truncated": True, "hint": "..."}`` 提示条目，防止大列表
      吃掉上下文；传 ``max_items=0`` 可关闭截断。
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
    ensure(
        200 <= status < 300,
        f"HTTP {status}: {text[:200]}",
    )
    # http_request 可能为超长响应追加 "\n... [截断...]" 或 "\n[hint] ..."；
    # 解析前先还原到最接近原始 JSON 的尾部。
    payload_text = text
    for marker in ("\n... [截断", "\n[hint] "):
        idx = payload_text.rfind(marker)
        if idx > 0:
            payload_text = payload_text[:idx]
    try:
        parsed = json_lib.loads(payload_text)
    except json_lib.JSONDecodeError as e:
        raise ValueError(
            f"HTTP {status} 响应不是合法 JSON：{e.msg} (position {e.pos})。"
            "建议改用 `http_request(...)` 原文查看或 `fetch(parse='text')`。"
        ) from e

    if pick and json_pointer:
        raise ValueError("pick= 与 json_pointer= 互斥，请只传其中一个。")
    if pick:
        try:
            parsed = _traverse_pick_path(parsed, pick)
        except KeyError as e:
            raise ValueError(f"pick='{pick}' 失败：{e}") from e
    elif json_pointer:
        try:
            parsed = _traverse_json_pointer(parsed, json_pointer)
        except KeyError as e:
            raise ValueError(f"json_pointer='{json_pointer}' 失败：{e}") from e

    if isinstance(parsed, list):
        parsed = clean_items(parsed)

    original_count: int | None = len(parsed) if isinstance(parsed, list) else None
    parsed, truncated = _truncate_json_payload(parsed, max_items=max_items)

    if isinstance(parsed, list):
        count = len(parsed)
    elif isinstance(parsed, dict):
        count = len(parsed)
    else:
        count = 1
    metrics_parts = [f"items={count}"]
    if truncated and original_count is not None:
        metrics_parts.append(f"truncated->{max_items} of {original_count}")
    if pick:
        metrics_parts.append(f"pick={pick}")
    if json_pointer:
        metrics_parts.append(f"json_pointer={json_pointer}")
    emit_tool_observation(
        "http_json", "OK", ",".join(metrics_parts), summarize_json(parsed)
    )
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
    *,
    pick: str | None = None,
    json_pointer: str | None = None,
    max_items: int = HTTP_JSON_MAX_ITEMS,
    max_chars: int = HTTP_RESPONSE_MAX_CHARS,
    **kwargs: Any,
) -> Any:
    """统一 HTTP 请求入口，优先使用此工具请求接口数据。

    可选参数：
    - ``pick="data.items"``：解析 JSON 后沿点号路径钻取（仅 json 模式）。
    - ``json_pointer="/data/0"``：RFC 6901 JSON Pointer 取值（仅 json 模式）。
    - ``max_items=100``：JSON 列表截断阈值；``0`` 表示不截断。
    - ``max_chars=20000``：响应正文截断阈值（text/raw 模式及 json 内部 http_request）。
    """
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
            pick=pick,
            json_pointer=json_pointer,
            max_items=max_items,
            max_chars=max_chars,
            **kwargs,
        )
    if normalized_parse in {"text", "raw"}:
        if pick or json_pointer:
            raise ValueError("pick= / json_pointer= 仅在 parse='json' 下可用。")
        return await http_request(
            url,
            method=method,
            headers=headers,
            params=params,
            data=data,
            json=json,
            content=content,
            cookies=cookies,
            max_chars=max_chars,
            **kwargs,
        )
    raise ValueError(f"fetch(parse=...) 仅支持 json / text / raw，收到: {parse}")


def register_http_tools(registry: ToolRegistry) -> None:
    """注册 HTTP 工具到 registry。"""
    registry.register(fetch, category="网络 & 数据")
    registry.register(http_request, category="网络 & 数据", expose_in_prompt=False)
    registry.register(http_json, category="网络 & 数据", expose_in_prompt=False)
