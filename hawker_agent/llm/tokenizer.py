from __future__ import annotations

from litellm import token_counter as _litellm_token_counter


def count_tokens(messages: list[dict[str, str]], model_name: str) -> int:
    """使用 litellm.token_counter 精确计算 token 数。"""
    try:
        return _litellm_token_counter(model=model_name, messages=messages)
    except Exception:
        # fallback: 中文约 1.5 字符/token，英文约 4 字符/token，取保守值 2
        total = 0
        for msg in messages:
            total += 4 + max(1, len(msg.get("content", "")) // 2)
        return total


def count_tokens_text(text: str, model_name: str) -> int:
    """单段文本的 token 估算。"""
    try:
        return _litellm_token_counter(
            model=model_name,
            messages=[{"role": "user", "content": text}],
        )
    except Exception:
        return max(1, len(text) // 2)
