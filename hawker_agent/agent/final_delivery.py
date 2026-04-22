from __future__ import annotations

import logging

from hawker_agent.agent.artifact import recover_items_from_artifact
from hawker_agent.agent.evaluator import evaluate_final_delivery, extract_task_requirements
from hawker_agent.models.history import CodeAgentHistoryList
from hawker_agent.models.state import CodeAgentState
from hawker_agent.models.step import CodeAgentStepMetadata
from hawker_agent.tools.data_tools import normalize_items

logger = logging.getLogger(__name__)

def replace_state_items(state: CodeAgentState, items: list[dict]) -> None:
    """用最终交付结果覆盖运行态 items。

    这是系统内部一致性收敛，不暴露成模型工具。
    在最终交付被接受后调用，确保 result.json / items_count / final_answer 口径一致。
    """
    state.items.clear()
    state.items.append(normalize_items(items))


def resolve_final_items(
    *,
    final_artifact: dict | None,
    fallback_items: list[dict] | None = None,
) -> list[dict]:
    """从唯一结构化真相源 final_artifact 恢复 items。

    final_answer 只作为展示文本；结构化结果一律以 final_artifact 为准。
    若 final_artifact 无法恢复出 items，才回落到运行态 items。
    """
    artifact_items = recover_items_from_artifact(final_artifact)
    if artifact_items:
        return artifact_items

    return normalize_items(fallback_items or [])


async def process_final_answer_request(
    *,
    task: str,
    step: int,
    state: CodeAgentState,
    step_meta: CodeAgentStepMetadata,
    history: CodeAgentHistoryList,
    observation: str,
) -> str:
    """处理模型提交的 final_answer 申请。

    ``final_answer()`` 只是提交候选结果；本函数负责执行错误拦截、最终评估、
    最终 items 收敛，并在放行后把候选结果晋升为正式结果。
    """
    if not state.final_answer_requested:
        return observation

    if step_meta.error:
        logger.warning("Step %d: final_answer 被拒绝，因为代码执行报错", step)
        state.final_answer_requested = None
        state.final_artifact_requested = None
        return f"{observation}\n[final_answer已拒绝] 本步有执行错误"

    final_answer_text = state.final_answer_requested or ""
    delivery_items = resolve_final_items(
        final_artifact=state.final_artifact_requested,
        fallback_items=state.items.to_list(),
    )
    evaluation = await evaluate_final_delivery(
        task=task,
        final_answer=final_answer_text,
        items=delivery_items,
        recent_observations=[],
        state=state,
    )
    if evaluation and not evaluation.accept:
        logger.warning("Step %d: final_answer 被评估器拒绝: %s", step, evaluation.reason)
        state.final_answer_requested = None
        state.final_artifact_requested = None
        history.add_user(
            "[System 提示] 最终交付已被评估器拒绝。\n"
            f"原因: {evaluation.reason}\n"
            "请基于当前样本、字段完整性和最近 observation 修正后再重新提交 final_answer。"
        )
        return f"{observation}\n[final_answer已拒绝] {evaluation.reason}"

    requirements = extract_task_requirements(task)
    if delivery_items:
        # 无论是 inline_json 还是摘要模式，只要 final_artifact 已明确携带
        # 最终结果集，就应覆盖运行态 items，避免探索期累积数据污染最终导出。
        replace_state_items(state, delivery_items)
        logger.info(
            "Step %d: %s 交付已覆盖运行态 items，最终条数=%d",
            step,
            requirements.delivery_mode,
            len(delivery_items),
        )

    logger.info("Step %d: 任务完成申请被接受", step)
    state.done = True
    state.answer = final_answer_text
    state.final_artifact = state.final_artifact_requested
    return observation
