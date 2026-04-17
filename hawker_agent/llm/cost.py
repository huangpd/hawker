from __future__ import annotations

import logging

from litellm import completion_cost as _litellm_completion_cost

logger = logging.getLogger(__name__)


def calculate_cost(
    response: object | None = None,
    *,
    model: str | None = None,
    messages: list[dict[str, str]] | None = None,
    completion: str = "",
) -> float:
    """计算 LLM 调用产生的费用。

    Args:
        response (object | None): LLM 响应对象。
        model (str | None): 模型名，用于 response 算价失败时回退。
        messages (list[dict[str, str]] | None): 原始请求消息，用于回退估算。
        completion (str): 输出文本，用于回退估算。

    Returns:
        float: 预估的费用（美元）。如果计算失败，则返回 0.0。
    """
    try:
        return float(_litellm_completion_cost(completion_response=response))
    except Exception as primary_exc:
        if model and messages is not None:
            try:
                return float(
                    _litellm_completion_cost(
                        model=model,
                        messages=messages,
                        completion=completion,
                    )
                )
            except Exception as fallback_exc:
                logger.debug(
                    "LiteLLM 算价失败: response=%s, fallback=%s, model=%s",
                    primary_exc,
                    fallback_exc,
                    model,
                )
                return 0.0
        logger.debug("LiteLLM 算价失败: %s", primary_exc)
        return 0.0
