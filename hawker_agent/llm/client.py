from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass

import litellm
from litellm import acompletion as _litellm_completion

from hawker_agent.config import Settings, get_settings
from hawker_agent.exceptions import LLMError
from hawker_agent.llm.cost import calculate_cost
from hawker_agent.observability import trace

# --- 彻底静音 LiteLLM ---
litellm.set_verbose = False
litellm.suppress_debug_info = True
litellm.add_status_to_exception = False
os.environ["LITELLM_LOG"] = "ERROR"
# 针对特定的 LiteLLM logger 进行暴力压制
for l in ["LiteLLM", "litellm", "litellm.utils", "litellm.main"]:
    logging.getLogger(l).setLevel(logging.WARNING)
    logging.getLogger(l).propagate = False

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """
    LLM 层原始输出。
    """
    text: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    total_tokens: int
    cost: float
    is_truncated: bool = False
    truncate_reason: str | None = None
    raw: object = None


def _normalize_api_base(base_url: str | None) -> str | None:
    """清理 API base URL。"""
    if not base_url:
        return None
    normalized = base_url.rstrip("/")
    # 如果用户只给了域名，补全 /v1
    if not (normalized.endswith("/v1") or "/v1/" in normalized):
        normalized += "/v1"
    return normalized


def _normalize_model_name(model_name: str) -> str:
    """确保模型名称包含 provider 前缀，OpenAI 兼容接口统一使用 openai/ 前缀。"""
    if "/" in model_name:
        return model_name
    return f"openai/{model_name}"


def _usage_to_dict(usage: object) -> dict:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()  # type: ignore[union-attr]
    if hasattr(usage, "dict"):
        return usage.dict()  # type: ignore[union-attr]
    return {}


def _extract_usage(usage_dict: dict) -> tuple[int, int, int, int]:
    """从 usage 提取 token 统计，确保返回均为整数。"""
    prompt_details = usage_dict.get("prompt_tokens_details")
    cached = 0
    if isinstance(prompt_details, dict):
        cached = prompt_details.get("cached_tokens") or 0
    if not cached and isinstance(usage_dict.get("cache_read_input_tokens"), int):
        cached = usage_dict["cache_read_input_tokens"]
    if not cached and isinstance(usage_dict.get("cached_tokens"), int):
        cached = usage_dict["cached_tokens"]
    
    input_tokens = usage_dict.get("prompt_tokens") or usage_dict.get("input_tokens") or 0
    output_tokens = usage_dict.get("completion_tokens") or usage_dict.get("output_tokens") or 0
    total_tokens = usage_dict.get("total_tokens") or (int(input_tokens) + int(output_tokens))
    
    return int(input_tokens), int(output_tokens), int(cached), int(total_tokens)


def _detect_truncation(response_obj: object) -> tuple[bool, str | None]:
    """通过 finish_reason 检测是否截断。"""
    choices = getattr(response_obj, "choices", [])
    if choices:
        finish_reason = getattr(choices[0], "finish_reason", None)
        if finish_reason == "length":
            return True, "响应被截断 (finish_reason=length)"
    return False, None


class LLMClient:
    """
    通用 OpenAI 兼容 LLM 调用封装。
    """

    def __init__(self, cfg: Settings | None = None) -> None:
        self._cfg = cfg or get_settings()

    async def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """
        异步调用 LLM，返回 LLMResponse。
        包含针对 429 的自动重试。
        """
        model = _normalize_model_name(self._cfg.model_name)
        api_base = _normalize_api_base(self._cfg.openai_base_url)

        with trace("llm_generation", model=model) as span:
            # LiteLLM 警告: Gemini 3 系列模型 (如 gemini-3-flash-preview) 必须使用 temperature=1.0
            temperature = 1.0 if "gemini" in model.lower() else 0.7

            kwargs: dict = {
                "model": model,
                "messages": messages,
                "api_key": self._cfg.openai_api_key,
                "base_url": api_base,
                "timeout": 180,
                "temperature": temperature,
                "max_tokens": 1500,  # 强制压制长篇大论，缩短生成时间
            }
            
            if self._cfg.reasoning_effort:
                kwargs["reasoning_effort"] = self._cfg.reasoning_effort

            max_retries = 3
            retry_delay = 5

            response = None
            for attempt in range(max_retries):
                try:
                    # 记录核心动作到 DEBUG
                    logger.debug("LLM 请求开始: model=%s attempt=%d", model, attempt + 1)
                    response = await _litellm_completion(**kwargs)
                    break
                except Exception as exc:
                    exc_str = str(exc)
                    if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                        if attempt < max_retries - 1:
                            logger.warning("遇到限额，%ds 后重试...", retry_delay)
                            await asyncio.sleep(retry_delay)
                            continue
                    logger.exception("LLM 请求失败")
                    raise LLMError(f"LiteLLM 请求失败: {exc}") from exc
            else:
                raise LLMError("LLM 请求重试次数耗尽")

            # 提取数据
            text = response.choices[0].message.content or ""
            usage_dict = _usage_to_dict(getattr(response, "usage", None))
            input_t, output_t, cached_t, total_t = _extract_usage(usage_dict)
            cost = calculate_cost(response)
            is_truncated, truncate_reason = _detect_truncation(response)

            # 丰富 Span 数据
            span.data.update({
                "input_tokens": input_t,
                "output_tokens": output_t,
                "cached_tokens": cached_t,
                "cost": cost,
                "truncated": is_truncated,
            })

            logger.info(
                "LLM 请求完成: model=%s in=%d(cached=%d) out=%d total=%d cost=$%.4f truncated=%s",
                model, input_t, cached_t, output_t, total_t, cost, "yes" if is_truncated else "no"
            )

            return LLMResponse(
                text=text,
                input_tokens=input_t,
                output_tokens=output_t,
                cached_tokens=cached_t,
                total_tokens=total_t,
                cost=cost,
                is_truncated=is_truncated,
                truncate_reason=truncate_reason,
                raw=response,
            )
