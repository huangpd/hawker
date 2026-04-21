from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

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
    delivery_mode: str = "summary_with_structured_items"
    expected_output_format: Literal["text", "json", "markdown"] | None = None


def _detect_expected_output_format(
    task: str,
    expects_inline_json: bool,
) -> Literal["text", "json", "markdown"] | None:
    """只识别明确的 JSON 契约，不再从自然语言猜 Markdown/Text。"""
    if expects_inline_json:
        return "json"
    return None


def extract_task_requirements(task: str) -> TaskRequirements:
    """从任务中提炼仍会影响交付链路的最小契约。"""
    expects_inline_json = bool(re.search(r"返回\s*json|直接返回\s*json|输出\s*json", task, re.I))
    delivery_mode = "inline_json" if expects_inline_json else "summary_with_structured_items"
    expected_output_format = _detect_expected_output_format(task, expects_inline_json)
    return TaskRequirements(
        delivery_mode=delivery_mode,
        expected_output_format=expected_output_format,
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
                "你是 Hawker 的最终交付评估器。基于当前任务来分析，不要基于过于外部知识做评估"
                "你只负责判断当前 final_answer 是否应该被放行。"
                "不要改代码，不要规划下一步。"
                "只输出 JSON：{\"accept\": true|false, \"reason\": \"...\", \"missing_requirements\": [\"...\"]}。"
                "优先依据任务要求验收产出物，而不是依据模型自述。"
                "若 delivery_mode=summary_with_structured_items，则 final_answer 默认只需摘要，结构化数据由系统记录的 items/artifact 承载。"
                "若 delivery_mode=inline_json，则 final_answer 必须满足任务对内联 JSON 的要求。"
                "这里提供的“样本”仅是 items 的抽样预览，不代表全量条数，不能因为样本条数少于 items_count 就拒绝。"
                "拒绝必须基于任务文本、items 样本或最近观察中的显式证据；不要用你自己的外部知识或启发式推断替代现场证据。"
                "尤其不要仅凭 URL 编号、slug、文件名、ID 前缀、发布日期编码规则等启发式去否定结果，除非这些规则已在任务或观察中被明确证实。"
                "若某条数据只是看起来可疑，但缺少显式证据，请在 reason 中标注疑点，或直接放行，而不是硬拒绝。"
                "如果任务要求下载、保存或导出文件，则必须看到显式完成证据。"
                "仅有 pdf_url、download_url、文件链接或可下载地址，只能说明“可下载”，不能说明“已经下载完成”。"
                "下载完成证据应来自 items 中的 downloaded_file/download_status，或最近观察中的明确下载成功记录。"
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
