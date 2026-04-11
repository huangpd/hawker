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
    """创建本次运行的目录和日志文件，返回 (run_dir, log_path)。"""
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
    """将单步详情追加到 run.log。"""
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
    """将运行摘要追加到 run.log。"""
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
