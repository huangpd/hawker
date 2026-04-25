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
    expected_output_format: Literal["json"] | None = None


def _detect_expected_output_format(
    task: str,
    expects_inline_json: bool,
) -> Literal["json"] | None:
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
    run_dir: str | None = None,
) -> list[dict[str, str]]:
    """构造最终交付评估请求。"""
    sample_items = _select_sample_items(items, limit=3)
    obs_text = "\n".join(f"- {obs}" for obs in recent_observations if obs.strip()) or "- (无)"
    requirements = extract_task_requirements(task)
    from hawker_agent.tools.data_tools import check_files_on_disk
    file_report = {}
    if run_dir:
        file_report = check_files_on_disk(run_dir, items)
    evidence_report = _build_evidence_report(items, file_report)

    return [
        {
            "role": "system",
            "content": (
                "你是 Hawker 的最终交付评估器。只判断当前 final_answer 是否可以放行，"
                "不要改代码，不要规划下一步。只输出 JSON："
                "{\"accept\": true|false, \"reason\": \"...\", \"missing_requirements\": [\"...\"]}。\n\n"
                "验收原则：\n"
                "1. 以系统状态为准：items 是当前实体状态，样本只是抽样预览，不得因样本量少于 items_count 拒绝。\n"
                "2. summary_with_structured_items 模式下，final_answer 只需做用户可读总结；结构化数据完整性由 items/证据统计承载。\n"
                "3. inline_json 模式下，final_answer 必须满足任务要求的 JSON 契约。\n"
                "4. 只有任务明确要求下载、保存、上传文件，或 final_answer 明确宣称文件已交付时，才强制验收文件证据。\n"
                "5. 文件证据可以来自 download、artifacts.file、facts、OBS 记录或磁盘校验报告；不要要求固定字段名。\n"
                "6. 单独的业务字段链接不构成文件交付证据；只有协议字段（download、artifacts.file、facts）和校验报告才算。\n\n"
                "拒绝标准：\n"
                "1. 明确要求的文件交付缺少证据，或磁盘报告显示文件 missing/empty。\n"
                "2. final_answer 的数量、结论或完成状态与系统统计、证据统计明显矛盾。\n"
                "3. 任务要求结构化数据，但 items_count 为 0 \n"
                "4. inline_json 模式下 final_answer 没有有效 JSON 或字段严重缺失。\n\n"
                "放行原则：\n"
                "- 非关键疑点不要硬拒绝；可以在 reason 中说明疑点后放行。\n"
                "- 不要用外部知识、URL 编号、文件名猜测、发布时间编码等启发式否定现场证据。\n"
                "- 不要因为样本没有展示全量字段而拒绝；应结合 items_count、证据统计和最近观察判断。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"[任务]\n{task}\n\n"
                f"[交付模式]\n{requirements.delivery_mode}\n\n"
                f"[证据统计]\n{json.dumps(evidence_report, ensure_ascii=False, indent=2)}\n\n"
                f"[磁盘/对象存储校验报告]\n{json.dumps(file_report, ensure_ascii=False, indent=2)}\n\n"
                f"[最终回复 (final_answer)]\n{final_answer}\n\n"
                f"[系统统计]\n- 采集总数: {len(items)}\n- 最近观察: {obs_text}\n\n"
                f"[数据样本 (抽样预览)]\n{json.dumps(sample_items, ensure_ascii=False, indent=2)}\n"
            ),
        },
    ]


def _build_evidence_report(items: list[dict[str, Any]], file_report: dict[str, Any]) -> dict[str, Any]:
    file_evidence_items = 0
    for item in items:
        has_file_evidence = False
        if isinstance(item.get("download"), dict):
            has_file_evidence = True
        if isinstance(item.get("artifacts"), dict):
            has_file_evidence = True
        facts = item.get("facts")
        if isinstance(facts, dict):
            if facts.get("downloaded") is True:
                has_file_evidence = True
        if has_file_evidence:
            file_evidence_items += 1
    return {
        "items_count": len(items),
        "file_evidence_items": file_evidence_items,
        "verified_files": int(file_report.get("verified_count") or 0),
        "verified_obs_files": int(file_report.get("obs_verified_count") or 0),
        "missing_files": len(file_report.get("missing_files") or []),
        "empty_files": len(file_report.get("empty_files") or []),
    }


def _is_informative_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "unknown", "none", "null", "n/a", "missing", "missing_on_disk"}
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _evidence_score(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_evidence_score(v) for v in value.values()) + len(
            [key for key, subvalue in value.items() if _is_informative_value(subvalue) and not str(key).startswith("_")]
        )
    if isinstance(value, (list, tuple, set)):
        return sum(_evidence_score(v) for v in value)
    return 1 if _is_informative_value(value) else 0


def _select_sample_items(items: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Pick the most information-dense current-state items for evaluator sampling."""
    ranked = sorted(
        enumerate(items),
        key=lambda pair: (-_evidence_score(pair[1]), pair[0]),
    )
    return [items[index] for index, _item in ranked[:limit]]


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

    run_dir = str(state.run_dir) if hasattr(state, "run_dir") and state.run_dir else None
    messages = build_final_evaluation_messages(
        task=task,
        final_answer=final_answer,
        items=items,
        recent_observations=recent_observations,
        run_dir=run_dir,
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
