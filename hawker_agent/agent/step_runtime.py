from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from hawker_agent.agent.executor import execute
from hawker_agent.agent.final_delivery import process_final_answer_request
from hawker_agent.agent.parser import parse_response
from hawker_agent.models.cell import CodeCell
from hawker_agent.models.history import CodeAgentHistoryList
from hawker_agent.models.output import CodeAgentModelOutput
from hawker_agent.models.state import CodeAgentState
from hawker_agent.models.step import CodeAgentStepMetadata
from hawker_agent.storage.logger import log_step

logger = logging.getLogger(__name__)


@dataclass
class StepRunResult:
    """单步执行结果。

    该对象是 step runtime 与 runner orchestration 之间的边界。
    runner 只需要关心是否应继续下一步、当前无进展计数，以及是否触发终止原因。
    """

    no_progress_steps: int
    stop_reason: Literal["done", "token_budget", "no_progress"] | None = None
    skipped: bool = False


async def run_agent_step(
    *,
    step: int,
    task: str,
    max_steps: int,
    cfg: Any,
    llm: Any,
    history: CodeAgentHistoryList,
    namespace: Any,
    state: CodeAgentState,
    log_path: Any,
    cells: list[CodeCell],
    no_progress_steps: int,
    inject_reflection_prompts: Any,
) -> StepRunResult:
    """执行一次完整 step 生命周期。

    包含：
    1. 组装 prompt 并调用模型
    2. 解析并执行代码
    3. 处理 final_answer 申请
    4. 更新 token/历史/日志
    5. 判定是否触发 done/token_budget/no_progress 终止
    """
    prompt_package = history.build_prompt_package()
    prompt_msgs = prompt_package["messages"]
    llm_response = await llm.complete(prompt_msgs)
    model_output = parse_response(llm_response.text)

    if llm_response.is_truncated:
        logger.warning("Step %d: 响应异常: %s", step, llm_response.truncate_reason)
        if not model_output.has_code:
            state.llm_records.append(
                {
                    "step": step,
                    "prompt": prompt_package,
                    "llm_response": {
                        "text": llm_response.text,
                        "input_tokens": llm_response.input_tokens,
                        "output_tokens": llm_response.output_tokens,
                        "cached_tokens": llm_response.cached_tokens,
                        "total_tokens": llm_response.total_tokens,
                        "cost": llm_response.cost,
                        "is_truncated": llm_response.is_truncated,
                        "truncate_reason": llm_response.truncate_reason,
                        "raw": llm_response.raw,
                    },
                    "parsed_output": None,
                    "execution": None,
                }
            )
            history.add_user(
                f"[System] 上一次响应异常: {llm_response.truncate_reason}\n"
                "请写一个简短计划(1-2句)，然后尝试执行一个简单的单步操作。"
            )
            return StepRunResult(no_progress_steps=no_progress_steps, skipped=True)
        logger.warning("Step %d: 截断响应包含可执行代码，继续执行已解析代码块", step)

    activity_before, progress_before = state.snapshot_markers()
    step_meta = CodeAgentStepMetadata(
        step_no=step,
        activity_before=activity_before,
        progress_before=progress_before,
    )

    logger.info(
        "Thought: %s",
        model_output.thought[:150] + "..." if len(model_output.thought) > 150 else model_output.thought,
    )

    observation = await _execute_step_code(step, model_output, namespace, state)

    if state.pending_dom:
        history.inject_dom(state.pending_dom)
        state.pending_dom = None

    step_meta.output = observation
    if "[执行错误]" in observation:
        step_meta.error = observation

    if state.final_answer_requested:
        observation = await process_final_answer_request(
            task=task,
            step=step,
            state=state,
            step_meta=step_meta,
            history=history,
            observation=observation,
        )

    state.token_stats.add(
        llm_response.input_tokens,
        llm_response.output_tokens,
        llm_response.cached_tokens,
        llm_response.cost,
    )

    progress_made, next_no_progress_steps = _update_progress_counters(
        state=state,
        step_meta=step_meta,
        no_progress_steps=no_progress_steps,
    )

    cell = step_meta.to_cell(model_output, state.token_stats, len(state.items))
    cells.append(cell)
    log_step(
        log_path,
        step,
        step_meta.elapsed(),
        state.token_stats,
        model_output.thought,
        model_output.code,
        observation,
    )

    _record_step_history(
        step=step,
        max_steps=max_steps,
        max_total_tokens=cfg.max_total_tokens,
        prompt_package=prompt_package,
        llm_response=llm_response,
        model_output=model_output,
        observation=observation,
        had_error=bool(step_meta.error),
        progress_made=progress_made,
        no_progress_steps=next_no_progress_steps,
        history=history,
        namespace=namespace,
        state=state,
    )

    stop_reason = _resolve_step_stop_reason(
        step=step,
        cfg=cfg,
        state=state,
        no_progress_steps=next_no_progress_steps,
    )
    if stop_reason:
        return StepRunResult(no_progress_steps=next_no_progress_steps, stop_reason=stop_reason)

    inject_reflection_prompts(history, step, state, step_meta, max_steps, next_no_progress_steps)

    remaining_steps = max_steps - step
    if remaining_steps == 1 or (state.token_stats.total_tokens >= cfg.max_total_tokens * 0.9):
        history.add_user(
            "[System 警告] 即将达到步数/token上限！\n"
            "请优先准备最终交付：\n"
            "- 下一步应以 `final_answer()` 收尾，除非还缺最后一个必要动作\n"
            "- 说明哪些已完成、哪些未完成\n"
            "- 即使任务未完成，也要返回已收集的部分数据。"
        )

    return StepRunResult(no_progress_steps=next_no_progress_steps)


async def _execute_step_code(
    step: int,
    model_output: CodeAgentModelOutput,
    namespace: Any,
    state: CodeAgentState,
) -> str:
    """执行模型产出的代码块。"""
    if model_output.has_code:
        return await execute(model_output.code, namespace, state=state, step=step)
    return "[错误] 未找到 ```python``` 代码块"


def _update_progress_counters(
    *,
    state: CodeAgentState,
    step_meta: CodeAgentStepMetadata,
    no_progress_steps: int,
) -> tuple[bool, int]:
    """根据本步执行结果更新进度计数。"""
    progress_made = step_meta.has_progress(state)
    next_no_progress_steps = 0 if progress_made else no_progress_steps + 1
    state.no_progress_streak = next_no_progress_steps
    if state.sop_guided_dom_steps_remaining > 0:
        state.sop_guided_dom_steps_remaining -= 1
        logger.info(
            "SOP 引导 DOM 护栏剩余步数: %d",
            state.sop_guided_dom_steps_remaining,
        )
    return progress_made, next_no_progress_steps


def _record_step_history(
    *,
    step: int,
    max_steps: int,
    max_total_tokens: int,
    prompt_package: dict[str, Any],
    llm_response: Any,
    model_output: CodeAgentModelOutput,
    observation: str,
    had_error: bool,
    progress_made: bool,
    no_progress_steps: int,
    history: CodeAgentHistoryList,
    namespace: Any,
    state: CodeAgentState,
) -> None:
    """把本步执行结果写入 history 和 llm_records。"""
    session_vars = namespace.get_llm_view()
    history.record_step(
        step=step,
        max_steps=max_steps,
        assistant_content=llm_response.text,
        observation=observation,
        namespace_view=session_vars,
        items_count=len(state.items),
        total_tokens=state.token_stats.total_tokens,
        max_total_tokens=max_total_tokens,
        progress=progress_made,
        had_error=had_error,
        no_progress_steps=no_progress_steps,
    )

    state.llm_records.append(
        {
            "step": step,
            "prompt": prompt_package,
            "llm_response": {
                "text": llm_response.text,
                "input_tokens": llm_response.input_tokens,
                "output_tokens": llm_response.output_tokens,
                "cached_tokens": llm_response.cached_tokens,
                "total_tokens": llm_response.total_tokens,
                "cost": llm_response.cost,
                "is_truncated": llm_response.is_truncated,
                "truncate_reason": llm_response.truncate_reason,
                "raw": llm_response.raw,
            },
            "parsed_output": {
                "thought": model_output.thought,
                "code": model_output.code,
            },
            "execution": {
                "observation": observation,
                "error": had_error,
                "progress_made": progress_made,
                "items_count": len(state.items),
                "no_progress_streak": no_progress_steps,
            },
        }
    )


def _resolve_step_stop_reason(
    *,
    step: int,
    cfg: Any,
    state: CodeAgentState,
    no_progress_steps: int,
) -> Literal["done", "token_budget", "no_progress"] | None:
    """根据运行状态判断是否需要在本步后终止。"""
    if state.done:
        return "done"
    if state.is_over_budget(cfg.max_total_tokens):
        logger.warning("Step %d: Token 预算耗尽", step)
        return "token_budget"
    if no_progress_steps >= cfg.max_no_progress_steps:
        logger.warning("Step %d: 连续 %d 步无进展，终止运行", step, no_progress_steps)
        return "no_progress"
    return None
