from __future__ import annotations

from litellm import completion_cost as _litellm_completion_cost


def calculate_cost(response: object) -> float:
    """计算 LLM 调用产生的费用。

    Args:
        response (object): LLM 响应对象。

    Returns:
        float: 预估的费用（美元）。如果计算失败，则返回 0.0。
    """
    try:
        return float(_litellm_completion_cost(completion_response=response))
    except Exception:
        return 0.0
