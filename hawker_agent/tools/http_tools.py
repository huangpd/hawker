from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from hawker_agent.observability import emit_observation, emit_tool_observation
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
    url: str, 
    method: str = "GET", 
    headers: str = "", 
    body: str = "",
    cookies: dict | str = "",
    **kwargs: Any
) -> str:
    """
    发送 HTTP 请求，返回完整响应文本。
    支持别名参数: json, data (自动映射到 body)
    """
    # 处理 Agent 常见的参数名习惯
    final_body = body
    if "json" in kwargs:
        final_body = json.dumps(kwargs["json"])
    elif "data" in kwargs:
        final_body = kwargs["data"] if isinstance(kwargs["data"], str) else json.dumps(kwargs["data"])
    
    h: dict[str, str] = {}
    if headers:
        try:
            h = json.loads(headers) if isinstance(headers, str) else headers
        except Exception: pass
    
    # 50年架构师提示：如果存在 body 但没有指定 Content-Type，自动补全为 JSON 以提升 Agent 成功率
    if final_body:
        h.setdefault("Content-Type", "application/json")
        h.setdefault("Accept", "application/json, text/plain, */*")
    
    c: dict[str, str] = {}
    if cookies:
        try:
            c = json.loads(cookies) if isinstance(cookies, str) else cookies
        except Exception: pass

    h.setdefault("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36")

    client = _get_client()
    try:
        if method.upper() == "POST":
            resp = await client.post(url, headers=h, content=final_body, cookies=c)
        else:
            resp = await client.get(url, headers=h, cookies=c)
            
        content_type = resp.headers.get("Content-Type", "unknown").split(";")[0]
        emit_tool_observation("http_request", str(resp.status_code), f"size={len(resp.text)}", f"type={content_type}")
        return f"[{resp.status_code}]\n{resp.text}"
    except Exception as e:
        logger.exception("HTTP 请求失败: %s %s", method.upper(), url)
        return f"[错误] {e}"


async def http_json(
    url: str, 
    method: str = "GET", 
    headers: str = "", 
    body: str = "",
    cookies: dict | str = "",
    **kwargs: Any
) -> object:
    """
    发送 HTTP 请求并返回解析后的 JSON 对象。
    - cookies: 可选。支持从 get_cookies() 获取的会话凭证。
    """
    raw = await http_request(url, method, headers, body, cookies, **kwargs)
    status, text = parse_http_response(raw)
    ensure(200 <= status < 300, f"HTTP {status}: {text[:200]}")
    data = json.loads(text)
    if isinstance(data, list):
        data = clean_items(data)
    
    count = len(data) if isinstance(data, (list, dict)) else 1
    emit_tool_observation("http_json", "OK", f"items={count}", summarize_json(data))
    return data


async def download_file(
    url: str, 
    filename: str | None = None, 
    run_dir: str | None = None, 
    cookies: dict | str = "",
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
        request_body = json.dumps(kwargs["json"])
    elif "data" in kwargs:
        request_body = kwargs["data"] if isinstance(kwargs["data"], str) else json.dumps(kwargs["data"])

    c: dict[str, str] = {}
    if cookies:
        try:
            c = json.loads(cookies) if isinstance(cookies, str) else cookies
        except Exception: pass

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
            
            # 2. 处理保存路径
            path = Path(final_filename)
            if run_dir and not path.is_absolute():
                save_path = Path(run_dir) / path
            else:
                save_path = path
                
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
    registry.register(http_request)
    registry.register(http_json)
    registry.register(download_file)
