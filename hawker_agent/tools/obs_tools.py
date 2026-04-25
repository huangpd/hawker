from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from hawker_agent.config import get_settings
from hawker_agent.observability import emit_tool_observation
from hawker_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


async def obs_stream_download(url: str, object_key: str, **kwargs: Any) -> dict[str, Any]:
    """将 URL 资源流式下载并同步上传到华为云 OBS，不经过本地磁盘。

    具备自动重试与断点续传能力，适合处理易中断的大文件下载。
    """
    from pyobs import StreamUploader
    
    cfg = get_settings()
    ak = cfg.obs_ak
    sk = cfg.obs_sk
    server = kwargs.get("server") or cfg.obs_server
    bucket = kwargs.get("bucket") or cfg.obs_bucket
    max_retries = kwargs.get("max_retries", 5)
    
    if not ak or not sk or not server or not bucket:
        raise RuntimeError("OBS 配置不完整。请确保 OBS_AK, OBS_SK, OBS_SERVER, OBS_BUCKET 环境变量已正确设置。")

    uploader = StreamUploader(ak=ak, sk=sk, server=server, bucket_name=bucket)
    loop = asyncio.get_running_loop()
    
    attempt = 0
    last_offset = -1
    
    while attempt < max_retries:
        # 1. 询问 OBS 进度（断点续传核心）
        context = await loop.run_in_executor(None, uploader.init_upload, object_key)
        
        # 如果进度停滞不前，说明可能是死循环或严重错误
        if context.offset == last_offset and attempt > 0:
            logger.warning("OBS 进度未增加 (offset=%d)，等待后重试...", context.offset)
        last_offset = context.offset

        try:
            def _perform_chunk_upload():
                with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                    headers = {}
                    if context.offset > 0:
                        headers["Range"] = f"bytes={context.offset}-"
                        logger.info("正在执行接力下载: %s from offset %d", object_key, context.offset)
                    
                    with client.stream("GET", url, headers=headers) as response:
                        if response.status_code == 416: # Range 不可满足，可能已下载完
                            return context.offset
                        response.raise_for_status()
                        
                        cl = response.headers.get("Content-Length")
                        total_size = int(cl) + context.offset if cl else 0
                        
                        uploader.upload_stream(
                            context=context,
                            stream_iterator=response.iter_bytes(chunk_size=5*1024*1024),
                            total_size=total_size,
                            mode="ab" if context.offset > 0 else "wb"
                        )
                        return total_size or context.offset

            actual_size = await loop.run_in_executor(None, _perform_chunk_upload)
            
            # 成功完成
            result = {
                "download": {
                    "obs_bucket": bucket,
                    "size": actual_size,
                    "method": "pyobs_stream",
                }
            }
            emit_tool_observation("obs_stream_download", "OK", f"key={object_key}, size={actual_size}")
            return result

        except (httpx.NetworkError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            attempt += 1
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in {403, 404}:
                raise # 不可恢复错误不重试
            
            wait_time = 2 ** attempt
            logger.warning("流式下载中断 (Attempt %d/%d): %s. %ds 后重试...", attempt, max_retries, e, wait_time)
            await asyncio.sleep(wait_time)

    raise RuntimeError(f"在 {max_retries} 次尝试后仍未能完成 OBS 流式下载: {object_key}")


def register_obs_tools(registry: ToolRegistry) -> None:
    """注册 OBS 相关工具。"""
    registry.register(obs_stream_download, category="网络 & 数据")
