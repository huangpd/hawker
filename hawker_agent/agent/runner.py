from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Literal

from hawker_agent.agent.executor import execute
from hawker_agent.agent.namespace import HawkerNamespace, build_system_dict
from hawker_agent.agent.parser import parse_response
from hawker_agent.agent.prompts import build_system_prompt
from hawker_agent.browser.session import BrowserSession
from hawker_agent.config import get_settings
from hawker_agent.llm.client import LLMClient
from hawker_agent.llm.tokenizer import count_tokens
from hawker_agent.models.cell import CodeCell
from hawker_agent.models.history import CodeAgentHistoryList
from hawker_agent.models.result import CodeAgentResult
from hawker_agent.models.state import CodeAgentState, TokenStats
from hawker_agent.models.step import CodeAgentStepMetadata
from hawker_agent.storage.exporter import export_notebook, save_result_json
from hawker_agent.storage.logger import init_run_dir, log_step, log_summary
from hawker_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _inject_reflection_prompts(
    history: CodeAgentHistoryList,
    step: int,
    state: CodeAgentState,
    step_meta: CodeAgentStepMetadata,
    max_steps: int,
    no_progress_steps: int,
) -> None:
    """在关键节点注入反思提示。迁移自 main.py 逻辑 3a/3b/3c/3d。"""
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


def _build_result(
    state: CodeAgentState,
    cells: list[CodeCell],
    stop_reason: Literal["done", "token_budget", "no_progress", "max_steps"],
    total_steps: int,
    log_path: Path,
    task: str,
    namespace: HawkerNamespace | None = None,
) -> CodeAgentResult:
    """构建最终结果对象并执行持久化。"""
    total_duration = time.time() - state.started_at
    final_answer = state.answer

    # 结果恢复逻辑（迁移自 _recover_partial_result）
    if stop_reason != "done" and not final_answer:
        parts = [f"停止原因: {stop_reason}。执行了 {total_steps} 步。"]
        if state.items:
            parts.insert(0, "[任务部分完成]")
            parts.append(f"已成功采集 {len(state.items)} 条数据。")
        else:
            parts.insert(0, "[任务未完成]")
            parts.append("未采集到有效数据。")

        if namespace:
            llm_vars = namespace.get_llm_view()
            data_vars = []
            for name, val in sorted(llm_vars.items()):
                if isinstance(val, (list, dict)) and val:
                    data_vars.append(f"  - {name}: {type(val).__name__}, {len(val)} 项")

            if data_vars:
                parts.append("\nnamespace 中可能包含部分数据的变量:")
                parts.extend(data_vars)
                
        final_answer = " ".join(parts[:3]) + ("\n" + "\n".join(parts[3:]) if len(parts) > 3 else "")

    # 执行持久化
    log_summary(log_path, state.token_stats, total_duration, total_steps, final_answer)
    nb_path = export_notebook(cells, task, state.run_dir)
    json_path = save_result_json(state.run_dir, state.items.to_list(), final_answer, state.checkpoint_files)

    return CodeAgentResult(
        answer=final_answer,
        success=stop_reason == "done",
        items=state.items.to_list(),
        run_id=state.run_id,
        total_steps=total_steps,
        total_duration=total_duration,
        token_stats=state.token_stats,
        stop_reason=stop_reason,
        run_dir=state.run_dir,
        log_path=log_path,
        notebook_path=nb_path,
        result_json_path=json_path,
    )


async def run(
    task: str,
    max_steps: int = 25,
    browser: BrowserSession | None = None,
    registry: ToolRegistry | None = None,
) -> CodeAgentResult:
    """
    Agent 主循环。

    参数:
        task: 自然语言描述的任务
        max_steps: 最大允许迭代步数
        browser: 可选，传入已存在的 BrowserSession 以复用
        registry: 可选，传入自定义 ToolRegistry
    """
    cfg = get_settings()
    state = CodeAgentState()
    # 初始化运行目录和日志
    run_dir, log_path = init_run_dir(task, cfg, run_id=state.run_id, trace_id=state.trace_id)
    state.run_dir = run_dir

    # 初始化组件
    llm = LLMClient(cfg)
    reg = registry or ToolRegistry()
    
    # 加载默认指令
    instructions_path = Path(__file__).parent.parent / "templates" / "default_instructions.txt"
    instructions = instructions_path.read_text(encoding="utf-8") if instructions_path.exists() else ""

    history = CodeAgentHistoryList.from_task(
        task,
        system_prompt=build_system_prompt(reg.build_description(), instructions=instructions),
        compression_threshold=cfg.message_compression_tokens,
    )
    # 为 history 注入 token 计数函数
    history._count_tokens_fn = lambda msgs: count_tokens(msgs, cfg.model_name)

    cells: list[CodeCell] = []
    no_progress_steps = 0

    async with (browser or BrowserSession()) as br:
        # 设置产物归档目录
        if hasattr(br, "target_dir"):
            br.target_dir = run_dir
            
        # 注册标准工具集
        from hawker_agent.tools.browser_tools import register_browser_tools
        from hawker_agent.tools.http_tools import register_http_tools
        
        register_browser_tools(reg, br, history)
        register_http_tools(reg)

        # 构建分层代码执行命名空间
        system_dict = build_system_dict(state, reg.as_namespace_dict(), str(run_dir))
        namespace = HawkerNamespace(system_dict, str(run_dir))

        for step in range(1, max_steps + 1):
            with state.bind_log_context(step):
                # 1. 准备并调用 LLM
                prompt_messages = history.to_prompt_messages()
                llm_response = llm.complete(prompt_messages)

                # 2. 处理截断异常
                if llm_response.is_truncated:
                    logger.warning("Step %d: 响应异常: %s", step, llm_response.truncate_reason)
                    history.add_user(
                        f"[System] 上一次响应异常: {llm_response.truncate_reason}\n"
                        "请写一个简短计划(1-2句)，然后尝试执行一个简单的单步操作。"
                    )
                    continue

                # 3. 解析响应
                model_output = parse_response(llm_response.text)
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
                        observation = f"{observation}\n[final_answer已拒绝] 本步有执行错误"
                    else:
                        logger.info("Step %d: 任务完成申请被接受", step)
                        state.done = True
                        state.answer = state.final_answer_requested

                # 6. 更新状态与统计
                state.token_stats.add(
                    llm_response.input_tokens,
                    llm_response.output_tokens,
                    llm_response.cached_tokens,
                    llm_response.cost
                )

                # 判断进度
                if step_meta.has_progress(state):
                    no_progress_steps = 0
                else:
                    no_progress_steps += 1
                
                # 7. 固化 CodeCell 并记录日志
                cell = step_meta.to_cell(model_output, state.token_stats, len(state.items))
                cells.append(cell)
                log_step(log_path, step, step_meta.elapsed(), state.token_stats, model_output.thought, model_output.code, observation)

                # 8. 更新对话历史
                history.add_assistant(llm_response.text)
                
                # 获取持久化变量摘要
                session_vars = namespace.get_llm_view()
                var_summary = []
                for k, v in sorted(session_vars.items()):
                    if k == "run_dir": continue
                    v_str = str(len(v)) + " 项" if isinstance(v, (list, dict)) else str(v)[:30]
                    var_summary.append(f"{k}: {v_str}")
                
                var_line = f" | 变量: {', '.join(var_summary)}" if var_summary else ""
                
                # 注入当前状态摘要和执行输出
                status_line = (
                    f"[状态] 已采集: {len(state.items)}条"
                    f" | 步骤: {step}/{max_steps}"
                    f" | token: {state.token_stats.total_tokens:,}/{cfg.max_total_tokens:,}"
                    f"{var_line}"
                )
                # observation 进历史前强制截断至 1500 字符，防止长列表拖垮上下文
                from hawker_agent.agent.compressor import truncate_output
                obs_for_history = truncate_output(observation, 1500)
                history.add_user(f"{status_line}\nObservation:\n{obs_for_history}")

                # 9. 关键节点反思注入
                _inject_reflection_prompts(history, step, state, step_meta, max_steps, no_progress_steps)

                # 10. 终止判断
                if state.done:
                    return _build_result(state, cells, "done", step, log_path, task, namespace)
                
                if state.is_over_budget(cfg.max_total_tokens):
                    logger.warning("Step %d: Token 预算耗尽", step)
                    return _build_result(state, cells, "token_budget", step, log_path, task, namespace)
                
                if no_progress_steps >= cfg.max_no_progress_steps:
                    logger.warning("Step %d: 连续 %d 步无进展，终止运行", step, no_progress_steps)
                    return _build_result(state, cells, "no_progress", step, log_path, task, namespace)
                
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
        return _build_result(state, cells, "max_steps", max_steps, log_path, task, namespace)
