from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

from hawker_agent.agent.artifact import normalize_final_artifact
from hawker_agent.agent.evaluator import extract_task_requirements
from hawker_agent.agent.final_delivery import resolve_final_items
from hawker_agent.agent.namespace import HawkerNamespace, build_namespace
from hawker_agent.agent.prompts import build_system_prompt
from hawker_agent.agent.step_runtime import run_agent_step
from hawker_agent.browser.session import BrowserSession
from hawker_agent.config import get_settings
from hawker_agent.knowledge.observer import maybe_generate_and_store_site_sop
from hawker_agent.knowledge.store import SiteSOP, SiteSOPStore
from hawker_agent.llm.client import LLMClient
from hawker_agent.llm.tokenizer import count_tokens
from hawker_agent.langfuse_client import flush_langfuse
from hawker_agent.models.cell import CodeCell
from hawker_agent.models.history import CodeAgentHistoryList
from hawker_agent.models.result import CodeAgentResult
from hawker_agent.models.state import CodeAgentState
from hawker_agent.models.step import CodeAgentStepMetadata
from hawker_agent.observability import trace, ToolStatsProcessor, add_trace_processor, remove_trace_processor
from hawker_agent.storage.exporter import export_notebook, save_llm_io_json, save_result_json
from hawker_agent.storage.logger import init_run_dir, log_summary
from hawker_agent.tools.http_tools import close_http_clients
from hawker_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _build_site_sop_workspace(sop: SiteSOP, *, max_chars: int = 5000) -> str:
    """构建注入给模型的站点 SOP 摘要。"""
    body = sop.sop_markdown.strip()
    if len(body) > max_chars:
        body = body[: max_chars - 20].rstrip() + "\n... [SOP 已截断]"
    return (
        f"Domain: {sop.domain}\n"
        f"Page Pattern: {sop.page_pattern or '(generic)'}\n"
        f"Golden Rule: {sop.golden_rule}\n"
        f"Workflow Kind: {sop.workflow_kind}\n"
        f"Should Inspect First: {sop.should_inspect_first}\n"
        f"Preferred Entry: {sop.preferred_entry or '(none)'}\n"
        f"Field Contract: {', '.join(sop.field_contract or []) or '(none)'}\n"
        f"Version: {sop.version}\n\n"
        f"{body}"
    )


def _build_site_sop_execution_instruction(sop: SiteSOP, task: str) -> str:
    """在页面模式精确命中时注入强执行指令。"""
    if not sop.page_pattern or sop.should_inspect_first:
        return ""
    fields = ", ".join(sop.field_contract or [])
    field_text = f"目标字段为: {fields}。" if fields else ""
    return (
        "已命中一条高置信站点 SOP，且页面模式已对齐当前任务。"
        f"Domain={sop.domain}; page_pattern={sop.page_pattern}; workflow={sop.workflow_kind}; preferred_entry={sop.preferred_entry}. "
        "下一步应直接使用已验证的提取 workflow，除非提取失败，否则不要先做页面侦察、不要先调用 inspect_page()。"
        f"{field_text}"
    )


def _build_output_format_instruction(task: str) -> str:
    """仅为显式 JSON 任务补充一条最终交付指令。"""
    requirements = extract_task_requirements(task)
    output_format = requirements.expected_output_format
    if output_format == "json":
        return (
            "最终交付格式要求：用户要求 JSON。调用 final_answer() 时，"
            "请优先提交合法 JSON，或显式使用 {'type': 'json', 'content': ...}。"
        )
    return ""


def _inject_reflection_prompts(
    history: CodeAgentHistoryList,
    step: int,
    state: CodeAgentState,
    step_meta: CodeAgentStepMetadata,
    max_steps: int,
    no_progress_steps: int,
) -> None:
    """只在明显卡住时注入一条简短诊断提示。"""
    if no_progress_steps == 2:
        history.add_user(
            "[System 警告] 连续 2 步无进展。请立即切换方法："
            "重新侦察 DOM / 检查 API 响应 / 更换翻页策略，不要重复同一操作。"
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
    requirements = extract_task_requirements(task)

    if not final_artifact and final_answer:
        final_artifact = normalize_final_artifact(
            final_answer,
            expected_output_format=requirements.expected_output_format,
        )
        state.final_artifact = final_artifact

    export_items = resolve_final_items(
        final_artifact=final_artifact,
        fallback_items=state.items.to_list(),
    )
    if export_items and export_items != state.items.to_list():
        logger.info(
            "导出前按最终交付收敛 items：runtime=%d -> export=%d",
            len(state.items),
            len(export_items),
        )
        state.items.clear()
        state.items.append(export_items)

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
        final_artifact = normalize_final_artifact(
            final_answer,
            expected_output_format=requirements.expected_output_format,
        )
        state.final_artifact = final_artifact

    # 执行持久化
    log_summary(log_path, state.token_stats, total_duration, total_steps, final_answer)
    nb_path = export_notebook(cells, task, state.run_dir)
    json_path = save_result_json(
        state.run_dir,
        export_items or state.items.to_list(),
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
        items=export_items or state.items.to_list(),
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
    requirements = extract_task_requirements(task)
    state.expected_output_format = requirements.expected_output_format
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
            sop_store = SiteSOPStore(cfg.knowledge_db_path)
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

            active_sop = sop_store.find_for_task(task)
            output_format_instruction = _build_output_format_instruction(task)
            if output_format_instruction:
                instructions += "\n\n" + output_format_instruction
            if active_sop:
                history.set_site_sop(_build_site_sop_workspace(active_sop))
                strong_execution_instruction = _build_site_sop_execution_instruction(active_sop, task)
                if strong_execution_instruction:
                    instructions += "\n\n" + strong_execution_instruction
                logger.info(
                    "启动时命中站点 SOP: domain=%s page_pattern=%s version=%d reason=%s",
                    active_sop.domain,
                    active_sop.page_pattern,
                    active_sop.version,
                    active_sop.update_reason,
                )
                state.sop_guided_dom_steps_remaining = 2
                state.sop_guided_reason = active_sop.golden_rule
                logger.info(
                    "启用 SOP 引导 DOM 护栏: 前 %d 步压制主动 full DOM | reason=%s",
                    state.sop_guided_dom_steps_remaining,
                    state.sop_guided_reason,
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
                    if cfg.observer_enabled:
                        try:
                            await maybe_generate_and_store_site_sop(
                                task=task,
                                state=state,
                                cells=cells,
                                browser=br,
                                sop_store=sop_store,
                                cfg=cfg,
                            )
                        except Exception as exc:
                            logger.warning("Observer 旁路生成失败，已忽略: %s", exc)
                    return _build_result(state, cells, stop_reason, step, log_path, task, cfg.model_name, namespace)

                for step in range(1, max_steps + 1):
                    with trace(f"agent_step_{step}", step=step):
                        step_result = await run_agent_step(
                            step=step,
                            task=task,
                            max_steps=max_steps,
                            cfg=cfg,
                            llm=llm,
                            history=history,
                            namespace=namespace,
                            state=state,
                            log_path=log_path,
                            cells=cells,
                            no_progress_steps=no_progress_steps,
                            inject_reflection_prompts=_inject_reflection_prompts,
                        )
                        no_progress_steps = step_result.no_progress_steps
                        if step_result.skipped:
                            continue
                        if step_result.stop_reason:
                            logger.info(stats_proc.get_summary())
                            result = await _finish(step_result.stop_reason, step)
                            flush_langfuse()
                            return result

                # 步数上限
                logger.warning("任务在达到最大步数 (%d) 后终止", max_steps)
                logger.info(stats_proc.get_summary())
                result = await _finish("max_steps", max_steps)
                flush_langfuse()
                return result
        finally:
            remove_trace_processor(stats_proc)
            await close_http_clients()
