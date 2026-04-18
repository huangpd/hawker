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


def resolve_final_delivery_items(task: str, final_answer: str, state: CodeAgentState) -> list[dict]:
    """确定最终交付评估应使用哪一份 items。

    默认使用运行时累积的 ``state.items``。当任务要求 inline JSON 且
    ``final_answer`` 中携带了合法 ``items`` 时，优先把这份内联结果视为
    最终交付候选，避免早期探索阶段 append 的脏数据绑架最终验收。
    """
    requirements = extract_task_requirements(task)
    if requirements.delivery_mode != "inline_json":
        return state.items.to_list()

    # inline_json 任务的最终答案更接近用户要的交付物，运行时缓存可能包含探索脏样本。
    recovered_items = recover_items_from_final_answer(final_answer)
    return recovered_items or state.items.to_list()


def replace_state_items(state: CodeAgentState, items: list[dict]) -> None:
    """用最终交付结果覆盖运行态 items。

    这是系统内部一致性收敛，不暴露成模型工具。仅在 inline JSON 交付通过
    最终验收后调用，确保 result.json / items_count / final_answer 口径一致。
    """
    state.items.clear()
    state.items.append(normalize_items(items))


def validate_final_answer_request(
    step: int,
    state: CodeAgentState,
    step_meta: CodeAgentStepMetadata,
) -> str | None:
    """校验本步是否允许接受 final_answer。

    返回拒绝原因；返回 None 表示允许继续走最终交付评估。
    """
    if step <= 1:
        return "首步禁止直接完成。请先观察样本并确认提取策略。"

    first_collection_step = (
        step_meta.activity_before == 0 and state.activity_marker > step_meta.activity_before
    )
    if first_collection_step:
        return "这是首次采集到数据的步骤。请下一步先检查样本、清洗字段或验证去重后再完成任务。"

    return None


def collect_recent_observations(state: CodeAgentState, limit: int = 2) -> list[str]:
    """提取最近少量 observation，供最终交付评估使用。"""
    observations: list[str] = []
    for record in reversed(state.llm_records):
        execution = record.get("execution") or {}
        obs = str(execution.get("observation") or "").strip()
        if not obs:
            continue
        # 最终交付评估只关心最近成功采集证据，不把旧的执行错误噪音带进去。
        if "[执行错误]" in obs or "未找到 ```python``` 代码块" in obs:
            continue
        observations.append(obs[:500])
        if len(observations) >= limit:
            break
    return list(reversed(observations))


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

    ``final_answer()`` 只是提交候选结果；本函数负责执行拒绝规则、最终评估、
    inline JSON items 收敛，并在放行后把候选结果晋升为正式结果。
    """
    if not state.final_answer_requested:
        return observation

    if step_meta.error:
        logger.warning("Step %d: final_answer 被拒绝，因为代码执行报错", step)
        state.final_answer_requested = None
        state.final_artifact_requested = None
        return f"{observation}\n[final_answer已拒绝] 本步有执行错误"

    reject_reason = validate_final_answer_request(step, state, step_meta)
    if reject_reason:
        logger.warning("Step %d: final_answer 被拒绝: %s", step, reject_reason)
        state.final_answer_requested = None
        state.final_artifact_requested = None
        history.add_user(
            "[System 提示] 本步 final_answer 已被拒绝。\n"
            f"原因: {reject_reason}\n"
            "请优先检查样本数据、关键字段是否为空/为0、以及选择器是否正确，"
            "必要时重新提取后再提交最终结果。"
        )
        return f"{observation}\n[final_answer已拒绝] {reject_reason}"

    # 最终交付评估优先看“这次准备交付什么”，而不是无条件看历史缓存。
    # 尤其 inline_json 任务里，final_answer.items 可能是在纠正早期误采的脏数据。
    final_answer_text = state.final_answer_requested or ""
    delivery_items = resolve_final_delivery_items(task, final_answer_text, state)
    evaluation = await evaluate_final_delivery(
        task=task,
        final_answer=final_answer_text,
        items=delivery_items,
        recent_observations=collect_recent_observations(state),
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
        recovered_items = recover_items_from_final_answer(final_answer_text)
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
