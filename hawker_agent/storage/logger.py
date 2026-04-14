from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from hawker_agent.config import Settings
from hawker_agent.models.state import TokenStats
from hawker_agent.observability import configure_logging, generate_trace_id, set_log_context

logger = logging.getLogger(__name__)


def init_run_dir(
    task: str,
    cfg: Settings,
    *,
    run_id: str | None = None,
    trace_id: str | None = None,
) -> tuple[Path, Path]:
    """初始化任务运行目录及日志配置。

    创建唯一的运行目录，初始化应用日志，并生成运行摘要文件的头部信息。

    Args:
        task (str): 任务描述文本。
        cfg (Settings): 全局配置对象。
        run_id (str | None, optional): 运行 ID，若不提供则自动生成。
        trace_id (str | None, optional): 追踪 ID，若不提供则自动生成。

    Returns:
        tuple[Path, Path]: 包含 (运行目录路径, 运行日志文件路径) 的元组。
    """
    resolved_run_id = run_id or uuid.uuid4().hex[:12]
    resolved_trace_id = trace_id or generate_trace_id()
    run_dir = cfg.scrape_dir / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 统一配置系统日志（app.log 会包含 trace_id）
    # 注意：这里的 level 应该从配置读取，默认 INFO
    configure_logging(level="INFO", log_path=run_dir / "app.log")
    
    # 初始化业务追踪上下文
    set_log_context(trace_id=resolved_trace_id, run_id=resolved_run_id)
    
    log_path = run_dir / "run.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run ID: {resolved_run_id}\n")
        f.write(f"Trace ID: {resolved_trace_id}\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {cfg.model_name}\n")
        if cfg.reasoning_effort:
            f.write(f"Reasoning effort: {cfg.reasoning_effort}\n")
        f.write("\nTask:\n")
        f.write(task.strip() + "\n")
    
    logger.info("✅ 运行目录就绪: %s [trace_id=%s]", run_dir, resolved_trace_id)
    return run_dir, log_path


def log_step(
    log_path: Path,
    step: int,
    duration: float,
    usage: TokenStats,
    thought: str,
    code: str,
    observation: str,
) -> None:
    """将单个步骤的详细执行记录追加到运行日志中。

    Args:
        log_path (Path): 运行日志文件的路径。
        step (int): 当前步骤序号。
        duration (float): 步骤执行耗时。
        usage (TokenStats): 本步骤的 token 消耗统计。
        thought (str): 模型生成的思考内容。
        code (str): 执行的 Python 代码。
        observation (str): 执行后的观察输出。
    """
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "━" * 80 + "\n")
        f.write(
            f"Step {step} | duration={duration:.1f}s | input={usage.input_tokens:,} | "
            f"output={usage.output_tokens:,} | cached={usage.cached_tokens:,}"
            f" | cost=${usage.cost:.4f}\n\n"
        )
        f.write("Thought:\n")
        f.write((thought or "[空]") + "\n\n")
        f.write("Action:\n```python\n")
        f.write((code or "[未提供]") + "\n```\n\n")
        f.write("Observation:\n")
        f.write(observation + "\n")


def log_summary(
    log_path: Path,
    token_stats: TokenStats,
    total_duration: float,
    total_steps: int,
    final_result: str,
) -> None:
    """将任务运行的最终摘要统计信息追加到运行日志中。

    Args:
        log_path (Path): 运行日志文件的路径。
        token_stats (TokenStats): 累计的 token 消耗统计。
        total_duration (float): 任务总耗时。
        total_steps (int): 执行的总步数。
        final_result (str): 任务的最终结果描述。
    """
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("Summary\n")
        f.write(f"Steps: {total_steps}\n")
        f.write(f"Duration: {total_duration:.1f}s\n")
        f.write(f"Input tokens: {token_stats.input_tokens:,}\n")
        f.write(f"Output tokens: {token_stats.output_tokens:,}\n")
        f.write(f"Cached tokens: {token_stats.cached_tokens:,}\n")
        total = token_stats.input_tokens + token_stats.output_tokens
        f.write(f"Total tokens: {total:,}\n")
        f.write(f"Total cost: ${token_stats.cost:.4f}\n")
        f.write(f"\nFinal result:\n{final_result}\n")
