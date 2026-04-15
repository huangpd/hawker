from __future__ import annotations

import ipaddress
import json as json_lib
import logging
import urllib.parse as _urlparse
from typing import Any

import httpx

from hawker_agent.observability import emit_observation, emit_tool_observation
from hawker_agent.tools.data_tools import clean_items, ensure, parse_http_response, summarize_json
from hawker_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


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


def _validate_url(url: str) -> None:
    """Block requests to private networks, cloud metadata endpoints, and other reserved ranges."""
    parsed = _urlparse.urlparse(url)
    hostname = (parsed.hostname or "").lower().strip(".")
    if not hostname:
        raise ValueError(f"Invalid URL (no hostname): {url}")
    if hostname in _BLOCKED_HOSTS:
        raise ValueError(f"Blocked request to reserved host: {hostname}")
    if _is_private_ip(hostname):
        raise ValueError(f"Blocked request to private/reserved IP: {hostname}")


def _get_client() -> httpx.AsyncClient:
    """延迟创建 httpx 异步客户端单例。"""
    global _client  # noqa: PLW0603
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    return _client


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

    _validate_url(url)
    client = _get_client()
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
    data = json_lib.loads(text)
    if isinstance(data, list):
        data = clean_items(data)
    
    count = len(data) if isinstance(data, (list, dict)) else 1
    emit_tool_observation("http_json", "OK", f"items={count}", summarize_json(data))
    return data


async def download_file(
    url: str, 
    filename: str | None = None, 
    run_dir: str | None = None, 
    cookies: dict | str | list = "",
    **kwargs: Any
) -> str:
    """
    通用流式下载工具。
    支持别名参数: json, data (自动映射到请求体)
    """
    from pathlib import Path
    import re
    import urllib.parse

    client = _get_client()

    # 处理 Agent 常见的参数名习惯
    request_body = None
    if "json" in kwargs:
        request_body = json_lib.dumps(kwargs["json"])
    elif "data" in kwargs:
        request_body = kwargs["data"] if isinstance(kwargs["data"], str) else json_lib.dumps(kwargs["data"])

    c = _parse_cookies(cookies)

    _validate_url(url)
    try:
        method = "POST" if request_body else "GET"
        async with client.stream(method, url, cookies=c, content=request_body) as resp:
            if resp.status_code != 200:
                return f"[失败] HTTP {resp.status_code}: {url}"
            
            # 1. 确定并清洗文件名
            final_filename = filename
            if not final_filename:
                # 尝试从 Content-Disposition 提取
                cd = resp.headers.get("content-disposition", "")
                if "filename=" in cd:
                    match = re.search(r'filename=["\']?([^"\';]+)["\']?', cd)
                    if match: final_filename = match.group(1)
                
                if not final_filename:
                    # 从 URL 路径提取
                    path_str = urllib.parse.urlparse(url).path
                    final_filename = path_str.split("/")[-1] or "downloaded_file"

            # 移除非法字符
            final_filename = re.sub(r'[\\/*?:"<>|]', '_', final_filename)
            # Strip path traversal: only keep the basename
            final_filename = Path(final_filename).name or "downloaded_file"
            
            # 2. 处理保存路径
            if run_dir:
                save_path = Path(run_dir) / final_filename
            else:
                save_path = Path(final_filename)
                
            save_path.parent.mkdir(parents=True, exist_ok=True)
                
            # 3. 执行块写入 (Chunked Write)
            total_bytes = 0
            with open(save_path, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
                    total_bytes += len(chunk)
            
            emit_tool_observation("download_file", str(resp.status_code), f"size={total_bytes}", f"file={save_path.name}")
            return f"[OK] 已保存 {total_bytes} 字节到 {save_path}"
    except Exception as e:
        logger.exception("下载文件失败: %s", url)
        return f"[错误] {e}"


def register_http_tools(registry: ToolRegistry) -> None:
    """注册 HTTP 工具到 registry。"""
    registry.register(http_request, category="网络 & 数据")
    registry.register(http_json, category="网络 & 数据")
    registry.register(download_file, category="网络 & 数据")
