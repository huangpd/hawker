from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from hawker_agent.config import get_settings
from hawker_agent.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class FinalEvaluation:
    accept: bool
    reason: str
    missing_requirements: list[str] | None = None
    raw_text: str = ""


@dataclass
class TaskRequirements:
    required_fields: list[str]
    expected_count_hint: int | None = None
    expects_inline_json: bool = False
    delivery_mode: str = "summary_with_structured_items"


def extract_task_requirements(task: str) -> TaskRequirements:
    """从用户任务里提炼轻量交付要求。"""
    fields: list[str] = []
    seen: set[str] = set()
    lines = [line.strip() for line in task.splitlines() if line.strip()]

    in_field_block = False
    for line in lines:
        lowered = line.lower()
        if "提取字段" in line or "字段:" in line:
            in_field_block = True
            continue
        if in_field_block:
            normalized = line.lstrip("-*•").strip()
            if not normalized:
                continue
            if ":" in normalized:
                field_name = normalized.split(":", 1)[0].strip().strip("`'\"")
                if field_name and field_name not in seen:
                    fields.append(field_name)
                    seen.add(field_name)
                continue
            if normalized.startswith(("1.", "2.", "3.", "步骤")):
                in_field_block = False
        if not in_field_block:
            for match in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", line):
                if match not in seen:
                    fields.append(match)
                    seen.add(match)

    expected_count_hint = None
    count_patterns = [
        r"前\s*(\d+)\s*条",
        r"返回\s*(\d+)\s*条",
        r"提取到\s*(\d+)\s*条",
    ]
    for pattern in count_patterns:
        m = re.search(pattern, task)
        if m:
            expected_count_hint = int(m.group(1))
            break

    expects_inline_json = bool(re.search(r"返回\s*json|直接返回\s*json|输出\s*json", task, re.I))
    delivery_mode = "inline_json" if expects_inline_json else "summary_with_structured_items"
    return TaskRequirements(
        required_fields=fields,
        expected_count_hint=expected_count_hint,
        expects_inline_json=expects_inline_json,
        delivery_mode=delivery_mode,
    )


def build_final_evaluation_messages(
    *,
    task: str,
    final_answer: str,
    items: list[dict[str, Any]],
    recent_observations: list[str],
) -> list[dict[str, str]]:
    """构造最终交付评估请求。"""
    sample_items = items[:3]
    obs_text = "\n".join(f"- {obs}" for obs in recent_observations if obs.strip()) or "- (无)"
    requirements = extract_task_requirements(task)
    return [
        {
            "role": "system",
            "content": (
                "你是 Hawker 的最终交付评估器。"
                "你只负责判断当前 final_answer 是否应该被放行。"
                "不要改代码，不要规划下一步。"
                "只输出 JSON：{\"accept\": true|false, \"reason\": \"...\", \"missing_requirements\": [\"...\"]}。"
                "优先依据任务要求验收产出物，而不是依据模型自述。"
                "若 delivery_mode=summary_with_structured_items，则 final_answer 默认只需摘要，结构化数据由系统记录的 items/artifact 承载。"
                "若 delivery_mode=inline_json，则 final_answer 必须满足任务对内联 JSON 的要求。"
                "这里提供的“样本”仅是 items 的抽样预览，不代表全量条数，不能因为样本条数少于 items_count 就拒绝。"
                "只有在样本明显与任务字段不符、items 为空、final_answer 与已采集数据明显矛盾，或交付证据严重不足时才拒绝。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"[任务]\n{task}\n\n"
                f"[任务要求摘要]\n{json.dumps(requirements.__dict__, ensure_ascii=False, indent=2)}\n\n"
                f"[final_answer]\n{final_answer}\n\n"
                f"[items_count]\n{len(items)}\n\n"
                f"[样本]\n{json.dumps(sample_items, ensure_ascii=False, indent=2)}\n\n"
                f"[最近观察]\n{obs_text}\n"
            ),
        },
    ]


def _parse_final_evaluation(text: str) -> FinalEvaluation | None:
    """解析评估模型返回的 JSON 结果。"""
    stripped = text.strip()
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                stripped = part
                break
            if "\n" in part:
                maybe = part.split("\n", 1)[1].strip()
                if maybe.startswith("{") and maybe.endswith("}"):
                    stripped = maybe
                    break
    try:
        payload = json.loads(stripped)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    accept = bool(payload.get("accept"))
    reason = str(payload.get("reason") or "").strip() or "评估器未提供原因"
    missing = payload.get("missing_requirements")
    missing_requirements = [str(x) for x in missing] if isinstance(missing, list) else None
    return FinalEvaluation(
        accept=accept,
        reason=reason,
        missing_requirements=missing_requirements,
        raw_text=text,
    )


async def evaluate_final_delivery(
    *,
    task: str,
    final_answer: str,
    items: list[dict[str, Any]],
    recent_observations: list[str],
    state: Any,
) -> FinalEvaluation | None:
    """使用小模型评估最终交付是否应被放行。"""
    cfg = get_settings()
    if not cfg.final_evaluator_enabled:
        return None
    if not cfg.small_model_name:
        logger.info("Final Evaluator 已启用但未配置 small_model_name，跳过评估")
        return None

    messages = build_final_evaluation_messages(
        task=task,
        final_answer=final_answer,
        items=items,
        recent_observations=recent_observations,
    )
    client = LLMClient(cfg)
    try:
        response = await client.complete_with_model(
            messages,
            model_name=cfg.small_model_name,
            reasoning_effort=cfg.final_evaluator_reasoning_effort,
            trace_name="final_evaluator_generation",
        )
    except Exception as exc:
        logger.warning("Final Evaluator 调用失败，跳过放行评估: %s", exc)
        state.evaluator_records.append(
            {"status": "llm_error", "error": str(exc)}
        )
        return None

    result = _parse_final_evaluation(response.text)
    state.evaluator_records.append(
        {
            "status": "parsed" if result else "parse_failed",
            "model": cfg.small_model_name,
            "raw_text": response.text,
            "tokens": {
                "input": response.input_tokens,
                "output": response.output_tokens,
                "cached": response.cached_tokens,
                "total": response.total_tokens,
                "cost": response.cost,
            },
            "result": {
                "accept": result.accept,
                "reason": result.reason,
            } if result else None,
        }
    )
    if result:
        logger.info("Final Evaluator 判定: accept=%s reason=%s", result.accept, result.reason)
    else:
        logger.warning("Final Evaluator 返回无法解析，跳过放行评估")
    return result
