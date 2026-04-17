from __future__ import annotations

import atexit
import logging
from typing import Any

from hawker_agent.config import get_settings

logger = logging.getLogger(__name__)

_LANGFUSE = None
_REGISTERED = False


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    atexit.register(shutdown_langfuse)
    _REGISTERED = True


def get_langfuse_client() -> Any | None:
    """懒加载 Langfuse 客户端，未配置时返回 None。"""
    global _LANGFUSE
    if _LANGFUSE is not None:
        return _LANGFUSE

    cfg = get_settings()
    if not cfg.langfuse_public_key or not cfg.langfuse_secret_key:
        return None

    try:
        from langfuse import Langfuse

        _LANGFUSE = Langfuse(
            public_key=cfg.langfuse_public_key,
            secret_key=cfg.langfuse_secret_key,
            base_url=cfg.langfuse_base_url,
            environment=cfg.langfuse_environment,
            release=cfg.langfuse_release,
        )
        _ensure_registered()
        logger.info("Langfuse 已启用: base_url=%s", cfg.langfuse_base_url or "default")
    except Exception as exc:
        logger.warning("Langfuse 初始化失败，已降级为本地观测: %s", exc)
        _LANGFUSE = None
    return _LANGFUSE


def start_observation(
    *,
    name: str,
    input: Any | None = None,
    metadata: dict[str, Any] | None = None,
    as_type: str = "span",
    model: str | None = None,
    parent_observation: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """创建 Langfuse observation 上下文并进入。"""
    client = get_langfuse_client()
    if client is None:
        return None, None

    payload: dict[str, Any] = {
        "name": name,
        "as_type": as_type,
    }
    if input is not None:
        payload["input"] = input
    if metadata:
        payload["metadata"] = metadata
    if model:
        payload["model"] = model

    try:
        if parent_observation is not None:
            ctx = parent_observation.start_as_current_observation(**payload)
        else:
            ctx = client.start_as_current_observation(**payload)
        obs = ctx.__enter__()
        return obs, ctx
    except Exception as exc:
        logger.warning("Langfuse observation 启动失败: %s", exc)
        return None, None


def update_observation(observation: Any | None, **kwargs: Any) -> None:
    """更新 Langfuse observation。"""
    if observation is None:
        return
    payload = {k: v for k, v in kwargs.items() if v is not None}
    if not payload:
        return
    try:
        observation.update(**payload)
    except Exception as exc:
        logger.debug("Langfuse observation 更新失败: %s", exc)


def end_observation(context_manager: Any | None, *, error: BaseException | None = None) -> None:
    """结束 Langfuse observation 上下文。"""
    if context_manager is None:
        return
    try:
        context_manager.__exit__(type(error) if error else None, error, error.__traceback__ if error else None)
    except Exception as exc:
        logger.debug("Langfuse observation 结束失败: %s", exc)


def flush_langfuse() -> None:
    """刷新 Langfuse 缓冲。"""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.debug("Langfuse flush 失败: %s", exc)


def shutdown_langfuse() -> None:
    """关闭 Langfuse 客户端。"""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
        client.shutdown()
    except Exception as exc:
        logger.debug("Langfuse shutdown 失败: %s", exc)

