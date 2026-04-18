from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass

import litellm
from litellm import acompletion as _litellm_completion
from litellm import aresponses as _litellm_responses

from hawker_agent.config import Settings, get_settings
from hawker_agent.exceptions import LLMError
from hawker_agent.langfuse_client import update_observation
from hawker_agent.llm.cost import calculate_cost
from hawker_agent.observability import bind_log_context, get_current_span, get_log_context, trace

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
    """LLM 层的原始输出结果封装。

    Attributes:
        text (str): 生成的文本内容。
        input_tokens (int): 输入消耗的 Token 数。
        output_tokens (int): 输出生成的 Token 数。
        cached_tokens (int): 命中的缓存 Token 数。
        total_tokens (int): 总消耗 Token 数。
        cost (float): 本次请求的预估费用（美元）。
        is_truncated (bool): 响应是否被截断。默认为 False。
        truncate_reason (str | None): 截断原因描述。默认为 None。
        raw (object): 底层 SDK 返回的原始响应对象。默认为 None。
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
    """规范化 API 基础 URL。

    Args:
        base_url (str | None): 原始的 API 基础 URL。

    Returns:
        str | None: 规范化后的 URL，如果输入为空则返回 None。
    """
    if not base_url:
        return None
    normalized = base_url.rstrip("/")
    # 如果用户只给了域名，补全 /v1
    if not (normalized.endswith("/v1") or "/v1/" in normalized):
        normalized += "/v1"
    return normalized


def _normalize_model_name(model_name: str) -> str:
    """规范化模型名称，确保包含 provider 前缀。

    Args:
        model_name (str): 原始模型名称。

    Returns:
        str: 带有 provider 前缀的模型名称。
    """
    if "/" in model_name:
        return model_name
    return f"openai/{model_name}"


def _usage_to_dict(usage: object) -> dict:
    """将 usage 对象转换为字典格式。

    Args:
        usage (object): LiteLLM 或 OpenAI 返回的 usage 对象。

    Returns:
        dict: 转换后的字典。
    """
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
    """从 usage 字典中提取 Token 统计信息。

    Args:
        usage_dict (dict): 包含 Token 统计信息的字典。

    Returns:
        tuple[int, int, int, int]: 包含 (输入 Token, 输出 Token, 缓存 Token, 总 Token) 的元组。
    """
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
    """通过 finish_reason 检测响应是否被截断。

    Args:
        response_obj (object): LLM 响应对象。

    Returns:
        tuple[bool, str | None]: (是否截断, 截断原因) 的元组。
    """
    choices = getattr(response_obj, "choices", [])
    if choices:
        finish_reason = getattr(choices[0], "finish_reason", None)
        if finish_reason == "length":
            return True, "响应被截断 (finish_reason=length)"
    return False, None


def _extract_text(response_obj: object) -> str:
    """兼容 Chat Completions 与 Responses API 两种返回结构，提取文本内容。"""
    choices = getattr(response_obj, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message else None
        if isinstance(content, str):
            return content

    output_text = getattr(response_obj, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text

    output = getattr(response_obj, "output", None)
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)

    return ""


def _should_retry_with_responses(exc: Exception) -> bool:
    """判断当前错误是否表明应改走 Responses API。"""
    exc_str = str(exc).lower()
    indicators = (
        "unsupported parameter: 'messages'",
        "parameter has moved to 'input'",
        "responses api",
    )
    return any(indicator in exc_str for indicator in indicators)


class LLMClient:
    """通用 OpenAI 兼容的 LLM 调用封装类。

    Attributes:
        _cfg (Settings): 包含 LLM 配置信息的设置对象。
    """

    def __init__(self, cfg: Settings | None = None) -> None:
        """初始化 LLM 客户端。

        Args:
            cfg (Settings | None, optional): LLM 配置对象。如果为 None，则获取默认设置。默认为 None。
        """
        self._cfg = cfg or get_settings()

    async def _complete_internal(
        self,
        messages: list[dict[str, str]],
        *,
        model_name: str | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        trace_name: str = "llm_generation",
    ) -> LLMResponse:
        """异步调用 LLM 生成补全结果。

        包含针对 429 (Rate Limit) 的自动重试逻辑。

        Args:
            messages (list[dict[str, str]]): 发送给 LLM 的消息列表，每个元素包含 "role" 和 "content"。

        Returns:
            LLMResponse: 封装后的 LLM 响应结果。

        Raises:
            LLMError: 当请求失败或重试次数耗尽时抛出。
        """
        raw_model_name = model_name or self._cfg.model_name
        model = _normalize_model_name(raw_model_name)
        api_base = _normalize_api_base(self._cfg.openai_base_url)
        current_step = get_log_context().step

        with trace(trace_name, model=model, as_type="generation", input={"messages": messages}) as span:
            step_context = bind_log_context(step=current_step) if current_step != "-" else bind_log_context()
            with step_context:
                chat_kwargs: dict = {
                    "model": model,
                    "messages": messages,
                    "api_key": self._cfg.openai_api_key,
                    "base_url": api_base,
                    "timeout": 180,
                    # 让 LiteLLM 按模型能力自动丢弃不支持的 OpenAI 参数，减少跨模型适配分支。
                    "drop_params": True,
                    "max_tokens": 2500,  # 提高上限，降低 finish_reason=length 频率
                }
                if temperature is not None:
                    chat_kwargs["temperature"] = temperature
                
                effective_reasoning_effort = self._cfg.reasoning_effort if reasoning_effort is None else reasoning_effort
                if effective_reasoning_effort:
                    chat_kwargs["reasoning_effort"] = effective_reasoning_effort

                responses_kwargs: dict = {
                    "model": model,
                    "input": messages,
                    "api_key": self._cfg.openai_api_key,
                    "base_url": api_base,
                    "timeout": 180,
                    "drop_params": True,
                    "max_output_tokens": 2500,
                }
                if temperature is not None:
                    responses_kwargs["temperature"] = temperature
                if effective_reasoning_effort:
                    responses_kwargs["reasoning_effort"] = effective_reasoning_effort

                max_retries = 3
                retry_delay = 5

                response = None
                use_responses_api = False
                for attempt in range(max_retries):
                    try:
                        # 记录核心动作到 DEBUG
                        logger.debug(
                            "LLM 请求开始: model=%s attempt=%d mode=%s",
                            model,
                            attempt + 1,
                            "responses" if use_responses_api else "chat",
                        )
                        if use_responses_api:
                            response = await _litellm_responses(**responses_kwargs)
                        else:
                            response = await _litellm_completion(**chat_kwargs)
                        break
                    except Exception as exc:
                        exc_str = str(exc)
                        if not use_responses_api and _should_retry_with_responses(exc):
                            logger.warning("LLM 请求提示当前模型需要 Responses API，自动切换重试: model=%s", model)
                            use_responses_api = True
                            continue
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
                text = _extract_text(response)
                usage_dict = _usage_to_dict(getattr(response, "usage", None))
                input_t, output_t, cached_t, total_t = _extract_usage(usage_dict)
                cost = calculate_cost(
                    response,
                    model=raw_model_name,
                    messages=messages,
                    completion=text,
                )
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

                current_span = get_current_span()
                observation = current_span.external_observation if current_span else span.external_observation
                update_observation(
                    observation,
                    output=text,
                    usage_details={
                        "prompt_tokens": input_t,
                        "completion_tokens": output_t,
                        "total_tokens": total_t,
                        "cached_tokens": cached_t,
                    },
                    cost_details={"total": cost},
                    model=model,
                    metadata={"truncated": is_truncated, "truncate_reason": truncate_reason},
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

    async def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        """使用主模型完成标准 Agent 推理。"""
        return await self._complete_internal(messages)

    async def complete_with_model(
        self,
        messages: list[dict[str, str]],
        *,
        model_name: str,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        trace_name: str = "llm_generation",
    ) -> LLMResponse:
        """使用指定模型完成一次补全，用于旁路任务如 Healing。"""
        return await self._complete_internal(
            messages,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            trace_name=trace_name,
        )
