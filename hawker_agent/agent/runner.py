from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Literal

from hawker_agent.agent.artifact import (
    artifact_to_answer_text,
    normalize_final_artifact,
    recover_items_from_artifact,
)
from hawker_agent.agent.executor import execute
from hawker_agent.agent.evaluator import evaluate_final_delivery, extract_task_requirements
from hawker_agent.agent.namespace import HawkerNamespace, build_namespace
from hawker_agent.agent.parser import parse_response
from hawker_agent.agent.prompts import build_system_prompt
from hawker_agent.browser.session import BrowserSession
from hawker_agent.config import get_settings
from hawker_agent.llm.client import LLMClient
from hawker_agent.llm.tokenizer import count_tokens
from hawker_agent.langfuse_client import flush_langfuse
from hawker_agent.memory.store import MemoryStore, build_raw_code_memories
from hawker_agent.models.cell import CodeCell
from hawker_agent.models.history import CodeAgentHistoryList
from hawker_agent.models.result import CodeAgentResult
from hawker_agent.models.state import CodeAgentState, TokenStats
from hawker_agent.models.step import CodeAgentStepMetadata
from hawker_agent.observability import trace, ToolStatsProcessor, add_trace_processor, remove_trace_processor
from hawker_agent.storage.exporter import export_notebook, save_llm_io_json, save_result_json
from hawker_agent.storage.logger import init_run_dir, log_step, log_summary
from hawker_agent.tools.data_tools import normalize_items
from hawker_agent.tools.http_tools import close_http_clients
from hawker_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _build_memory_workspace_entries(matches: list) -> list[str]:
    """构建注入给模型的 Memory Workspace 条目。

    对 raw_* 记忆附带 detail 片段，帮助模型直接复用可执行上下文。
    """
    entries: list[str] = []
    for match in matches:
        rendered = match.render()
        entries.append(rendered)

        memory_type = getattr(match.entry, "memory_type", "")
        detail = str(getattr(match.entry, "detail", "") or "").strip()
        if not memory_type.startswith("raw_") or not detail:
            continue

        # 控制 token：每条 detail 限长，避免 Memory Workspace 膨胀。
        detail_snippet = detail[:1500]
        if len(detail) > 1500:
            detail_snippet += "\n... [detail 已截断]"
        entries.append(f"{rendered}\n[Memory Detail]\n{detail_snippet}")
    return entries[:6]


def _recover_items_from_final_answer(answer: str) -> list[dict]:
    """从 final_answer 文本中兜底恢复结构化 items。"""
    artifact = normalize_final_artifact(answer)
    return recover_items_from_artifact(artifact)


def _resolve_final_delivery_items(task: str, final_answer: str, state: CodeAgentState) -> list[dict]:
    """确定最终交付评估应使用哪一份 items。

    默认使用运行时累积的 ``state.items``。当任务要求 inline JSON 且
    ``final_answer`` 中携带了合法 ``items`` 时，优先把这份内联结果视为
    最终交付候选，避免早期探索阶段 append 的脏数据绑架最终验收。
    """
    requirements = extract_task_requirements(task)
    if requirements.delivery_mode != "inline_json":
        return state.items.to_list()

    # 对 inline_json 任务，最终交付里的 items 才是最接近“用户要什么”的候选真值。
    # 运行过程中 append_items() 累积的是工作缓存，里面可能混入早期探索阶段的误采样本。
    # 因此 evaluator 不应被旧缓存绑死，而应优先检查 final_answer 内联提交的结果。
    recovered_items = _recover_items_from_final_answer(final_answer)
    return recovered_items or state.items.to_list()


def _replace_state_items(state: CodeAgentState, items: list[dict]) -> None:
    """用最终交付结果覆盖运行态 items。

    这里是系统内部的一致性收敛，不暴露成模型工具。仅在 inline JSON
    交付通过最终验收后调用，用来把运行过程中的临时/错误采集结果
    收敛为最终可落盘的结构化数据。
    """
    # 这里不是给模型增加一个“replace_items”工具，而是系统在最终放行后做内部收敛。
    # 只有 inline_json 交付通过评估时，才允许最终交付覆盖运行时缓存。
    # 这样 result.json / items_count / 最终 answer 才能保持一致。
    state.items.clear()
    state.items.append(normalize_items(items))


def _validate_final_answer_request(
    step: int,
    state: CodeAgentState,
    step_meta: CodeAgentStepMetadata,
) -> str | None:
    """校验本步是否允许接受 final_answer。

    返回拒绝原因；返回 None 表示允许完成。
    """
    if step <= 1:
        return "首步禁止直接完成。请先观察样本并确认提取策略。"

    first_collection_step = (
        step_meta.activity_before == 0 and state.activity_marker > step_meta.activity_before
    )
    if first_collection_step:
        return "这是首次采集到数据的步骤。请下一步先检查样本、清洗字段或验证去重后再完成任务。"

    return None


def _collect_recent_observations(state: CodeAgentState, limit: int = 2) -> list[str]:
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


def _inject_reflection_prompts(
    history: CodeAgentHistoryList,
    step: int,
    state: CodeAgentState,
    step_meta: CodeAgentStepMetadata,
    max_steps: int,
    no_progress_steps: int,
) -> None:
    """在关键间隔注入系统反思 Prompt 以引导代理。

    Args:
        history (CodeAgentHistoryList): 当前对话历史。
        step (int): 当前步骤编号。
        state (CodeAgentState): 当前全局代理状态。
        step_meta (CodeAgentStepMetadata): 当前步骤的元数据。
        max_steps (int): 任务允许的最大总步数。
        no_progress_steps (int): 连续无进展步骤的计数器。
    """
    # 3a. 首步完成 — 策略确认
    if step == 1:
        history.add_user(
            "[System 提示] 第一步完成。下一步请先确认：\n"
            "1. 页面类型：SSR 还是 SPA？有没有发现数据 API？\n"
            "2. 数据获取策略：API 重放 / DOM 提取 / 混合？\n"
            "3. 预估总数据量和翻页方式\n"
            "确认策略后再开始提取。"
        )

    # 3b. 首次成功采集 — 数据质量检查
    if step_meta.activity_before == 0 and state.activity_marker > 0:
        history.add_user(
            "[System 提示] 首批数据已采集。请检查：\n"
            "1. 字段是否完整，有无缺失关键字段？\n"
            "2. 数据格式是否干净（HTML 残留、编码问题）？\n"
            "3. 去重是否正常，有无重复条目？\n"
            "如有问题，先修复提取逻辑再继续。"
        )

    # 3c. 进度过半 — 进展回顾
    if step == max_steps // 2:
        history.add_user(
            f"[System 提示] 已用 {step}/{max_steps} 步，采集 {len(state.items)} 条。\n"
            "请评估：\n"
            "1. 当前进度是否符合预期？\n"
            "2. 剩余步数是否足够完成任务？\n"
            "3. 是否需要调整策略（如加大翻页步长、简化字段）？"
        )

    # 3d. 连续 2 步无进展 — 诊断提示
    if no_progress_steps == 2:
        history.add_user(
            "[System 警告] 连续 2 步无进展。请停下来诊断：\n"
            "1. 是选择器失效？用 dom_state() 重新获取页面状态\n"
            "2. 是 API 返回异常？检查 HTTP 状态码和响应内容\n"
            "3. 是翻页逻辑有误？检查页码是否超出范围\n"
            "必须切换策略，不要重复同一种方法。"
        )


def _build_namespace_skip_names(namespace: HawkerNamespace) -> set[str]:
    """基于当前 namespace 动态生成保留名集合。"""
    # 这里不再维护手写 skip_names 黑名单。
    # 直接以当前 system namespace 为准，这样工具/helper/预装模块增删时，
    # 结果恢复逻辑会自动同步，避免再次出现“新增工具后忘记更新 skip_names”的漂移。
    reserved = set(namespace.system.keys())
    # run_dir 在 session 层，不在 system 中，但它属于系统注入的运行上下文，
    # 不应被误判为模型产出的业务结果变量。
    reserved.update({"run_dir"})
    return reserved


def _build_result(
    state: CodeAgentState,
    cells: list[CodeCell],
    stop_reason: Literal["done", "token_budget", "no_progress", "max_steps"],
    total_steps: int,
    log_path: Path,
    task: str,
    model_name: str,
    namespace: HawkerNamespace | None = None,
) -> CodeAgentResult:
    """构建最终结果对象并执行数据持久化。

    Args:
        state (CodeAgentState): 最终的全局代理状态。
        cells (list[CodeCell]): 已执行的笔记本单元格列表。
        stop_reason (Literal): 代理停止的原因。
        total_steps (int): 已执行的总步数。
        log_path (Path): 日志文件路径。
        task (str): 原始任务描述。
        model_name (str): 所使用的 LLM 模型名称。
        namespace (HawkerNamespace | None): 最终执行的命名空间。

    Returns:
        CodeAgentResult: 代理运行的结构化结果。
    """
    total_duration = time.time() - state.started_at
    final_answer = state.answer
    final_artifact = state.final_artifact

    if not final_artifact and final_answer:
        final_artifact = normalize_final_artifact(final_answer)
        state.final_artifact = final_artifact

    if not state.items and final_artifact:
        recovered_items = recover_items_from_artifact(final_artifact)
        if recovered_items:
            added, skipped = state.items.append(recovered_items)
            logger.warning(
                "final_answer artifact 未显式 append_items，已自动回填 %d 条数据 (skipped=%d)",
                added,
                skipped,
            )

    # 结果恢复逻辑（当任务非正常结束时生成总结）
    if stop_reason != "done" and not final_answer:
        parts = [f"停止原因: {stop_reason}。执行了 {total_steps} 步。"]
        if state.items:
            parts.insert(0, "[任务部分完成]")
            parts.append(f"已成功采集 {len(state.items)} 条数据。")
        else:
            parts.insert(0, "[任务未完成]")
            parts.append("未采集到有效数据。")

        if namespace:
            # 只展示 session 中真正像“业务数据”的变量。
            # system namespace 里的工具、helper、标准库等统一动态跳过。
            skip_names = _build_namespace_skip_names(namespace)
            data_vars = []
            # 扫描持久化层 (session)
            for name, val in sorted(namespace.session.items()):
                if name.startswith("_") or name in skip_names or callable(val):
                    continue
                if isinstance(val, (list, dict)) and val:
                    data_vars.append(f"  - {name}: {type(val).__name__}, {len(val)} 项")

            if data_vars:
                parts.append("\nnamespace 中可能包含部分数据的变量:")
                parts.extend(data_vars)

        # 因为 items 字段已经包含了完整数据。保持 answer 作为纯语义总结。
        final_answer = " ".join(parts[:3]) + ("\n" + "\n".join(parts[3:]) if len(parts) > 3 else "")
        final_artifact = normalize_final_artifact(final_answer)
        state.final_artifact = final_artifact

    # 执行持久化
    log_summary(log_path, state.token_stats, total_duration, total_steps, final_answer)
    nb_path = export_notebook(cells, task, state.run_dir)
    json_path = save_result_json(
        state.run_dir,
        state.items.to_list(),
        final_answer,
        final_artifact=final_artifact,
        checkpoint_files=state.checkpoint_files,
    )
    llm_io_path = save_llm_io_json(
        state.run_dir,
        task,
        state.llm_records,
        healing_records=state.healing_records,
        evaluator_records=state.evaluator_records,
    )

    return CodeAgentResult(
        answer=final_answer,
        success=stop_reason == "done",
        artifact=final_artifact,
        items=state.items.to_list(),
        run_id=state.run_id,
        model_name=model_name,
        total_steps=total_steps,
        total_duration=total_duration,
        token_stats=state.token_stats,
        stop_reason=stop_reason,
        run_dir=state.run_dir,
        log_path=log_path,
        notebook_path=nb_path,
        result_json_path=json_path,
        llm_io_path=llm_io_path,
    )


async def run(
    task: str,
    max_steps: int = 25,
    browser: BrowserSession | None = None,
    registry: ToolRegistry | None = None,
) -> CodeAgentResult:
    """Hawker Agent 循环的主入口。

    Args:
        task (str): 要执行任务的自然语言描述。
        max_steps (int): 允许的最大迭代步数。默认为 25。
        browser (BrowserSession | None): 可选的现有浏览器会话，用于复用。
        registry (ToolRegistry | None): 可选的自定义工具注册表。

    Returns:
        CodeAgentResult: 代理执行的最终结果。
    """
    cfg = get_settings()
    state = CodeAgentState()
    # 初始化运行目录和日志
    run_dir, log_dir, log_path = init_run_dir(task, cfg, run_id=state.run_id, trace_id=state.trace_id)
    state.run_dir = run_dir
    state.log_dir = log_dir

    with trace("agent_run", task=task, run_id=state.run_id):
        # 注册工具统计处理器
        stats_proc = ToolStatsProcessor()
        add_trace_processor(stats_proc)
        try:
            # 初始化组件
            llm = LLMClient(cfg)
            memory_store = MemoryStore(cfg.memory_db_path)
            reg = registry or ToolRegistry()
        
            # 加载指令 (默认指令 + 用户自定义)
            tpl_dir = Path(__file__).parent.parent / "templates"
            default_path = tpl_dir / "default_instructions.txt"
            custom_path = tpl_dir / "custom_instructions.txt"
            
            instructions = default_path.read_text(encoding="utf-8") if default_path.exists() else ""
            if custom_path.exists():
                custom_content = custom_path.read_text(encoding="utf-8").strip()
                if custom_content:
                    instructions += "\n\n" + custom_content

            history = CodeAgentHistoryList.from_task(
                task,
                system_prompt="", # 稍后在工具注册完后再填充
                compression_threshold=cfg.message_compression_tokens,
            )
            # 为 history 注入 token 计数函数
            history._count_tokens_fn = lambda msgs: count_tokens(msgs, cfg.model_name)

            recalled_memories = memory_store.search(task, limit=5)
            if recalled_memories:
                history.set_memory_workspace(_build_memory_workspace_entries(recalled_memories))
                logger.info("启动时命中 %d 条站点记忆", len(recalled_memories))
                for idx, match in enumerate(recalled_memories, start=1):
                    logger.info(
                        "记忆召回 #%d: site=%s score=%.1f negative=%s summary=%s",
                        idx,
                        match.entry.site_key,
                        match.score,
                        match.entry.negative,
                        match.entry.summary,
                    )
                top_match = recalled_memories[0]
                if top_match.score >= 130:
                    state.memory_guided_dom_steps_remaining = 2
                    state.memory_guided_reason = top_match.entry.summary
                    logger.info(
                        "启用记忆引导DOM护栏: 前 %d 步压制主动 full DOM | reason=%s",
                        state.memory_guided_dom_steps_remaining,
                        state.memory_guided_reason,
                    )

            cells: list[CodeCell] = []
            no_progress_steps = 0

            async with (browser or BrowserSession()) as br:
                # 设置产物归档目录
                if hasattr(br, "target_dir"):
                    br.target_dir = run_dir
                    
                # 1. 注册标准工具集
                from hawker_agent.tools.browser_tools import register_browser_tools
                from hawker_agent.tools.http_tools import register_http_tools
                from hawker_agent.tools.data_tools import register_data_tools
                from hawker_agent.agent.namespace import register_core_actions
                
                register_core_actions(reg, state, str(run_dir))
                register_browser_tools(reg, br, history, state)
                register_http_tools(reg)
                register_data_tools(reg)

                # 2. 生成并注入最终提示词 (此时分类生成才准确)
                full_system_prompt = build_system_prompt(
                    async_capabilities=reg.build_capabilities_list("async"),
                    sync_capabilities=reg.build_capabilities_list("sync"),
                    instructions=instructions
                )
                
                history.system_prompt = full_system_prompt

                # 构建代码执行命名空间
                system_dict = build_namespace(state, reg.as_namespace_dict(), str(run_dir))
                namespace = HawkerNamespace(system_dict, str(run_dir))

                async def _finish(stop_reason: Literal["done", "token_budget", "no_progress", "max_steps"], step: int) -> CodeAgentResult:
                    try:
                        entries = build_raw_code_memories(task, state)
                        saved = memory_store.upsert_entries(entries)
                        if saved:
                            logger.info("本次运行写入/更新 %d 条记忆", len(saved))
                            for idx, entry in enumerate(saved, start=1):
                                logger.info(
                                    "记忆写回 #%d: site=%s intent=%s type=%s negative=%s step=%d summary=%s",
                                    idx,
                                    entry.site_key,
                                    entry.task_intent,
                                    entry.memory_type,
                                    entry.negative,
                                    entry.source_step,
                                    entry.summary,
                                )
                    except Exception as exc:
                        logger.warning("记忆总结失败，已跳过写入: %s", exc)
                    return _build_result(state, cells, stop_reason, step, log_path, task, cfg.model_name, namespace)

                for step in range(1, max_steps + 1):
                    with trace(f"agent_step_{step}", step=step):
                        # 1. 准备并调用 LLM
                        prompt_package = history.build_prompt_package()
                        prompt_msgs = prompt_package["messages"]
                        llm_response = await llm.complete(prompt_msgs)
                        model_output = parse_response(llm_response.text)

                        # 2. 处理截断异常
                        if llm_response.is_truncated:
                            logger.warning("Step %d: 响应异常: %s", step, llm_response.truncate_reason)
                            if model_output.has_code:
                                logger.warning("Step %d: 截断响应包含可执行代码，继续执行已解析代码块", step)
                            else:
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
                                continue

                        # 3. 解析响应
                        activity_before, progress_before = state.snapshot_markers()
                        step_meta = CodeAgentStepMetadata(
                            step_no=step,
                            activity_before=activity_before,
                            progress_before=progress_before,
                        )

                        logger.info("Thought: %s", model_output.thought[:150] + "..." if len(model_output.thought) > 150 else model_output.thought)

                        # 4. 执行代码
                        if model_output.has_code:
                            observation = await execute(model_output.code, namespace, state=state, step=step)
                        else:
                            observation = "[错误] 未找到 ```python``` 代码块"
                        
                        # 处理侧道注入的 DOM 状态
                        if state.pending_dom:
                            history.inject_dom(state.pending_dom)
                            state.pending_dom = None
                        
                        step_meta.output = observation
                        if "[执行错误]" in observation:
                            step_meta.error = observation
                        
                        # 5. 处理 final_answer 申请
                        if state.final_answer_requested:
                            if step_meta.error:
                                # 报错了则拒绝本步的完成申请
                                logger.warning("Step %d: final_answer 被拒绝，因为代码执行报错", step)
                                state.final_answer_requested = None
                                state.final_artifact_requested = None
                                observation = f"{observation}\n[final_answer已拒绝] 本步有执行错误"
                            else:
                                reject_reason = _validate_final_answer_request(step, state, step_meta)
                                if reject_reason:
                                    logger.warning("Step %d: final_answer 被拒绝: %s", step, reject_reason)
                                    state.final_answer_requested = None
                                    state.final_artifact_requested = None
                                    observation = (
                                        f"{observation}\n[final_answer已拒绝] {reject_reason}"
                                    )
                                    history.add_user(
                                        "[System 提示] 本步 final_answer 已被拒绝。\n"
                                        f"原因: {reject_reason}\n"
                                        "请优先检查样本数据、关键字段是否为空/为0、以及选择器是否正确，"
                                        "必要时重新提取后再提交最终结果。"
                                    )
                                else:
                                    # 最终交付评估优先看“这次准备交付什么”，而不是无条件看历史缓存。
                                    # 尤其 inline_json 任务里，final_answer.items 可能是在纠正早期误采的脏数据。
                                    final_answer_text = state.final_answer_requested or ""
                                    delivery_items = _resolve_final_delivery_items(
                                        task,
                                        final_answer_text,
                                        state,
                                    )
                                    evaluation = await evaluate_final_delivery(
                                        task=task,
                                        final_answer=final_answer_text,
                                        items=delivery_items,
                                        recent_observations=_collect_recent_observations(state),
                                        state=state,
                                    )
                                    if evaluation and not evaluation.accept:
                                        logger.warning("Step %d: final_answer 被评估器拒绝: %s", step, evaluation.reason)
                                        observation = (
                                            f"{observation}\n[final_answer已拒绝] {evaluation.reason}"
                                        )
                                        history.add_user(
                                            "[System 提示] 最终交付已被评估器拒绝。\n"
                                            f"原因: {evaluation.reason}\n"
                                            "请基于当前样本、字段完整性和最近 observation 修正后再重新提交 final_answer。"
                                        )
                                        state.final_answer_requested = None
                                        state.final_artifact_requested = None
                                    else:
                                        requirements = extract_task_requirements(task)
                                        if requirements.delivery_mode == "inline_json":
                                            recovered_items = _recover_items_from_final_answer(final_answer_text)
                                            if recovered_items:
                                                # 评估通过后，把最终交付结果回写成正式 items。
                                                # 否则 result.json 仍会落盘旧缓存，出现“answer 是 4 条、产物是 5 条”的分裂状态。
                                                _replace_state_items(state, recovered_items)
                                                logger.info(
                                                    "Step %d: inline_json 交付已覆盖运行态 items，最终条数=%d",
                                                    step,
                                                    len(recovered_items),
                                                )
                                        logger.info("Step %d: 任务完成申请被接受", step)
                                        state.done = True
                                        state.answer = final_answer_text
                                        state.final_artifact = state.final_artifact_requested

                        # 6. 更新状态与统计
                        state.token_stats.add(
                            llm_response.input_tokens,
                            llm_response.output_tokens,
                            llm_response.cached_tokens,
                            llm_response.cost
                        )

                        # 判断进度
                        progress_made = step_meta.has_progress(state)
                        if progress_made:
                            no_progress_steps = 0
                        else:
                            no_progress_steps += 1
                        state.no_progress_streak = no_progress_steps
                        if state.memory_guided_dom_steps_remaining > 0:
                            state.memory_guided_dom_steps_remaining -= 1
                            logger.info(
                                "记忆引导DOM护栏剩余步数: %d",
                                state.memory_guided_dom_steps_remaining,
                            )

                        # 7. 固化 CodeCell 并记录日志
                        cell = step_meta.to_cell(model_output, state.token_stats, len(state.items))
                        cells.append(cell)
                        log_step(log_path, step, step_meta.elapsed(), state.token_stats, model_output.thought, model_output.code, observation)

                        # 8. 更新对话历史
                        session_vars = namespace.get_llm_view()
                        history.record_step(
                            step=step,
                            max_steps=max_steps,
                            assistant_content=llm_response.text,
                            observation=observation,
                            namespace_view=session_vars,
                            items_count=len(state.items),
                            total_tokens=state.token_stats.total_tokens,
                            max_total_tokens=cfg.max_total_tokens,
                            progress=progress_made,
                            had_error=bool(step_meta.error),
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
                                    "error": step_meta.error,
                                    "progress_made": progress_made,
                                    "items_count": len(state.items),
                                    "no_progress_streak": no_progress_steps,
                                },
                            }
                        )

                        # 9. 终止判断
                        if state.done:
                            logger.info(stats_proc.get_summary())
                            result = await _finish("done", step)
                            flush_langfuse()
                            return result

                        # 10. 关键节点反思注入
                        _inject_reflection_prompts(history, step, state, step_meta, max_steps, no_progress_steps)
                        
                        if state.is_over_budget(cfg.max_total_tokens):
                            logger.warning("Step %d: Token 预算耗尽", step)
                            logger.info(stats_proc.get_summary())
                            result = await _finish("token_budget", step)
                            flush_langfuse()
                            return result
                        
                        if no_progress_steps >= cfg.max_no_progress_steps:
                            logger.warning("Step %d: 连续 %d 步无进展，终止运行", step, no_progress_steps)
                            logger.info(stats_proc.get_summary())
                            result = await _finish("no_progress", step)
                            flush_langfuse()
                            return result
                        
                        # 接近上限警告（迁移自 main.py 逻辑）
                        remaining_steps = max_steps - step
                        if remaining_steps == 1 or (state.token_stats.total_tokens >= cfg.max_total_tokens * 0.9):
                            history.add_user(
                                "[System 警告] 即将达到步数/token上限！\n"
                                "你必须在下一步调用 final_answer() 返回结果：\n"
                                "- 说明哪些已完成，哪些未完成\n"
                                "- 即使任务未完成，也请返回已收集的部分数据。"
                            )

                # 步数上限
                logger.warning("任务在达到最大步数 (%d) 后终止", max_steps)
                logger.info(stats_proc.get_summary())
                result = await _finish("max_steps", max_steps)
                flush_langfuse()
                return result
        finally:
            remove_trace_processor(stats_proc)
            await close_http_clients()
