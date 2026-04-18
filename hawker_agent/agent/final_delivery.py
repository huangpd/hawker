from __future__ import annotations

import logging

from hawker_agent.agent.artifact import normalize_final_artifact, recover_items_from_artifact
from hawker_agent.agent.evaluator import evaluate_final_delivery, extract_task_requirements
from hawker_agent.models.history import CodeAgentHistoryList
from hawker_agent.models.state import CodeAgentState
from hawker_agent.models.step import CodeAgentStepMetadata
from hawker_agent.tools.data_tools import normalize_items

logger = logging.getLogger(__name__)


def recover_items_from_final_answer(answer: str) -> list[dict]:
    """从 final_answer 文本中兜底恢复结构化 items。"""
    artifact = normalize_final_artifact(answer, expected_output_format="json")
    return recover_items_from_artifact(artifact)

def replace_state_items(state: CodeAgentState, items: list[dict]) -> None:
    """用最终交付结果覆盖运行态 items。

    这是系统内部一致性收敛，不暴露成模型工具。仅在 inline JSON 交付通过
    最终验收后调用，确保 result.json / items_count / final_answer 口径一致。
    """
    state.items.clear()
    state.items.append(normalize_items(items))


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
    inline JSON items 收敛，并在放行后把候选结果晋升为正式结果。
    """
    if not state.final_answer_requested:
        return observation

    if step_meta.error:
        logger.warning("Step %d: final_answer 被拒绝，因为代码执行报错", step)
        state.final_answer_requested = None
        state.final_artifact_requested = None
        return f"{observation}\n[final_answer已拒绝] 本步有执行错误"

    final_answer_text = state.final_answer_requested or ""
    recovered_items = recover_items_from_final_answer(final_answer_text)
    delivery_items = recovered_items or state.items.to_list()
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
    if requirements.delivery_mode == "inline_json":
        if recovered_items:
            # 评估通过后，把最终交付结果回写成正式 items，避免 answer 与落盘 items 分裂。
            replace_state_items(state, recovered_items)
            logger.info(
                "Step %d: inline_json 交付已覆盖运行态 items，最终条数=%d",
                step,
                len(recovered_items),
            )

    logger.info("Step %d: 任务完成申请被接受", step)
    state.done = True
    state.answer = final_answer_text
    state.final_artifact = state.final_artifact_requested
    return observation
