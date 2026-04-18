from __future__ import annotations

import difflib
import json
import logging
from typing import TYPE_CHECKING, Any

from hawker_agent.agent.parser import parse_response
from hawker_agent.config import get_settings
from hawker_agent.llm.client import LLMClient

if TYPE_CHECKING:
    from hawker_agent.agent.namespace import HawkerNamespace
    from hawker_agent.models.state import CodeAgentState

logger = logging.getLogger(__name__)

_HEALABLE_ERROR_TYPES = (
    "SyntaxError",
    "NameError",
    "KeyError",
    "IndexError",
    "TypeError",
    "AttributeError",
)

_NON_HEALABLE_HINTS = (
    "[安全限制]",
    "禁止导入模块",
    "Blocked request",
    "HTTP 401",
    "HTTP 403",
)

_MAX_HEAL_CHANGE_RATIO = 0.55


def is_healable_error(error_text: str) -> bool:
    """判断当前错误是否适合交给廉价模型做局部修复。"""
    if not error_text or "[执行错误]" not in error_text:
        return False
    if any(hint in error_text for hint in _NON_HEALABLE_HINTS):
        return False
    return any(err in error_text for err in _HEALABLE_ERROR_TYPES)


def build_namespace_snapshot(namespace: HawkerNamespace, max_entries: int = 30) -> dict[str, str]:
    """提取适合提供给 Healing 模型的轻量变量摘要。"""
    snapshot: dict[str, str] = {}
    for name, value in list(namespace.get_llm_view().items())[:max_entries]:
        try:
            type_name = type(value).__name__
            if isinstance(value, (list, dict, tuple, set)):
                snapshot[name] = f"{type_name}(len={len(value)})"
            else:
                preview = repr(value)
                snapshot[name] = f"{type_name}={preview[:120]}"
        except Exception:
            snapshot[name] = type(value).__name__
    return snapshot


def build_healing_messages(*, code: str, error: str, namespace_snapshot: dict[str, str]) -> list[dict[str, str]]:
    """构造局部代码修复请求。"""
    snapshot_text = json.dumps(namespace_snapshot, ensure_ascii=False, indent=2)
    return [
        {
            "role": "system",
            "content": (
                "你是一个极简 Python 代码修复器。"
                "你的任务是修复当前这一个代码单元的局部错误。"
                "只能做最小修改：修语法、补判空、修变量引用、修索引/属性访问。"
                "尽量保留原有代码结构、变量名、循环和工具调用，不要整段重写。"
                "禁止改任务目标、禁止改字段 schema、禁止引入新策略、禁止输出解释。"
                "只输出一个 ```python``` 代码块。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请修复下面这段代码。\n\n"
                f"[可用变量摘要]\n{snapshot_text}\n\n"
                f"[原代码]\n```python\n{code}\n```\n\n"
                f"[报错]\n{error}\n"
            ),
        },
    ]


def estimate_change_ratio(original: str, candidate: str) -> float:
    """估算修复候选相对原代码的变更比例。"""
    if not original.strip():
        return 1.0
    return 1.0 - difflib.SequenceMatcher(a=original, b=candidate).ratio()


async def try_heal_code(
    *,
    code: str,
    error: str,
    namespace: HawkerNamespace,
    state: CodeAgentState,
) -> str | None:
    """尝试使用廉价模型修复当前代码单元。

    返回修复后的代码；若不适合修复或修复失败，则返回 None。
    """
    cfg = get_settings()
    if not cfg.healer_enabled:
        return None
    if not cfg.small_model_name:
        logger.info("Healing 已启用但未配置 small_model_name，跳过旁路修复")
        return None
    if not is_healable_error(error):
        return None

    snapshot = build_namespace_snapshot(namespace)
    client = LLMClient(cfg)
    logger.info("Healing 启动: error_type=%s", next((err for err in _HEALABLE_ERROR_TYPES if err in error), "unknown"))

    for attempt in range(1, cfg.healer_max_attempts + 1):
        messages = build_healing_messages(code=code, error=error, namespace_snapshot=snapshot)
        try:
            response = await client.complete_with_model(
                messages,
                model_name=cfg.small_model_name,
                reasoning_effort=cfg.healer_reasoning_effort,
                trace_name="healing_generation",
            )
        except Exception as exc:
            logger.warning("Healing attempt %d 失败: %s", attempt, exc)
            state.healing_records.append(
                {
                    "attempt": attempt,
                    "status": "llm_error",
                    "error": str(exc),
                }
            )
            continue

        parsed = parse_response(response.text)
        healed_code = parsed.code.strip()
        change_ratio = estimate_change_ratio(code, healed_code) if healed_code else 1.0
        state.healing_records.append(
            {
                "attempt": attempt,
                "status": "candidate",
                "input_error": error,
                "original_code": code,
                "candidate_code": healed_code,
                "change_ratio": round(change_ratio, 4),
                "model": cfg.small_model_name,
                "tokens": {
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                    "cached": response.cached_tokens,
                    "total": response.total_tokens,
                    "cost": response.cost,
                },
            }
        )
        if not healed_code or healed_code == code:
            logger.info("Healing attempt %d 未产生有效修复", attempt)
            continue
        if change_ratio > _MAX_HEAL_CHANGE_RATIO:
            logger.warning(
                "Healing attempt %d 被拒绝: 改动过大 (change_ratio=%.2f)",
                attempt,
                change_ratio,
            )
            state.healing_records.append(
                {
                    "attempt": attempt,
                    "status": "rejected_large_change",
                    "change_ratio": round(change_ratio, 4),
                }
            )
            continue
        if healed_code and healed_code != code:
            logger.info(
                "Healing success: attempt=%d change_ratio=%.2f",
                attempt,
                change_ratio,
            )
            state.healing_records.append(
                {
                    "attempt": attempt,
                    "status": "accepted",
                    "change_ratio": round(change_ratio, 4),
                    "model": cfg.small_model_name,
                }
            )
            return healed_code

    logger.info("Healing 放弃: 未找到可接受的局部修复")
    return None
