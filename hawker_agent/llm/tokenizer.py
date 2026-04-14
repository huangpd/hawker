from __future__ import annotations

from litellm import token_counter as _litellm_token_counter


def count_tokens(messages: list[dict[str, str]], model_name: str) -> int:
    """使用 litellm.token_counter 精确计算消息列表的 token 数。

    Args:
        messages (list[dict[str, str]]): 待计算的消息列表，每个消息包含 role 和 content。
        model_name (str): 用于计算 token 的模型名称。

    Returns:
        int: 计算出的 token 总数。
    """
    try:
        return _litellm_token_counter(model=model_name, messages=messages)
    except Exception:
        # fallback: 中文约 1.5 字符/token，英文约 4 字符/token，取保守值 2
        total = 0
        for msg in messages:
            total += 4 + max(1, len(msg.get("content", "")) // 2)
        return total


def count_tokens_text(text: str, model_name: str) -> int:
    """单段文本的 token 估算。

    Args:
        text (str): 待估算的文本字符串。
        model_name (str): 用于估算的模型名称。

    Returns:
        int: 估算的 token 数。
    """
    try:
        return _litellm_token_counter(
            model=model_name,
            messages=[{"role": "user", "content": text}],
        )
    except Exception:
        return max(1, len(text) // 2)
