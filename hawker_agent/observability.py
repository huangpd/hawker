from __future__ import annotations

import contextlib
import contextvars
import logging
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class LogContext:
    """日志上下文，挂载到每条 LogRecord 上。"""
    trace_id: str = "-"
    run_id: str = "-"
    step: str = "-"


_LOG_CONTEXT: contextvars.ContextVar[LogContext] = contextvars.ContextVar(
    "hawker_log_context",
    default=LogContext(),
)
_OBSERVATION_SINK: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "hawker_observation_sink",
    default=None,
)
_BASE_RECORD_FACTORY = logging.getLogRecordFactory()
_RECORD_FACTORY_INSTALLED = False


def generate_trace_id() -> str:
    """生成 32 位十六进制 trace_id。"""
    return uuid.uuid4().hex


def get_log_context() -> LogContext:
    """获取当前协程/线程的日志上下文。"""
    return _LOG_CONTEXT.get()


def clear_log_context() -> None:
    """清空日志上下文，恢复默认值。"""
    _LOG_CONTEXT.set(LogContext())


def set_log_context(
    *,
    trace_id: str | None = None,
    run_id: str | None = None,
    step: int | str | None = None,
) -> LogContext:
    """直接设置日志上下文。"""
    current = get_log_context()
    updated = LogContext(
        trace_id=trace_id or current.trace_id,
        run_id=run_id or current.run_id,
        step=current.step if step is None else str(step),
    )
    _LOG_CONTEXT.set(updated)
    return updated


@contextlib.contextmanager
def bind_log_context(
    *,
    trace_id: str | None = None,
    run_id: str | None = None,
    step: int | str | None = None,
) -> Iterator[LogContext]:
    """临时绑定日志上下文。"""
    current = get_log_context()
    updated = LogContext(
        trace_id=trace_id or current.trace_id,
        run_id=run_id or current.run_id,
        step=current.step if step is None else str(step),
    )
    token = _LOG_CONTEXT.set(updated)
    try:
        yield updated
    finally:
        _LOG_CONTEXT.reset(token)


def _install_log_record_factory() -> None:
    global _RECORD_FACTORY_INSTALLED
    if _RECORD_FACTORY_INSTALLED:
        return

    def record_factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = _BASE_RECORD_FACTORY(*args, **kwargs)
        ctx = get_log_context()
        record.trace_id = ctx.trace_id
        record.run_id = ctx.run_id
        record.step = ctx.step
        return record

    logging.setLogRecordFactory(record_factory)
    _RECORD_FACTORY_INSTALLED = True


def _normalize_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    value = logging.getLevelName(level.upper())
    if isinstance(value, int):
        return value
    return logging.INFO


def _build_formatter(with_color: bool = False, for_rich: bool = False) -> logging.Formatter:
    """构建日志格式器。"""
    if for_rich:
        # RichHandler 已经处理了时间、级别和颜色，我们只需提供上下文信息
        return logging.Formatter("[%(trace_id).8s] [%(step)s] %(message)s")
    
    if with_color:
        return logging.Formatter("\033[2m%(trace_id).8s\033[0m [%(step)s] %(message)s")
    
    return logging.Formatter(
        fmt=(
            "%(asctime)s %(levelname)s %(name)s "
            "[trace_id=%(trace_id)s run_id=%(run_id)s step=%(step)s] %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@contextlib.contextmanager
def collect_observations() -> Iterator[list[str]]:
    """在当前上下文中收集显式 observation，避免与 stdout 混流。"""
    buffer: list[str] = []
    token = _OBSERVATION_SINK.set(buffer)
    try:
        yield buffer
    finally:
        _OBSERVATION_SINK.reset(token)


def configure_logging(
    *,
    level: int | str = logging.INFO,
    log_path: Path | None = None,
    force: bool = False,
) -> None:
    """初始化统一 logging，默认使用 INFO 级别。"""
    _install_log_record_factory()
    root = logging.getLogger()
    
    target_level = _normalize_level(level)
    root.setLevel(target_level)

    # 尝试引入 Rich
    try:
        from rich.logging import RichHandler
        use_rich = True
    except ImportError:
        use_rich = False

    if force:
        for handler in root.handlers[:]:
            if type(handler).__name__ == "LogCaptureHandler":
                continue
            root.removeHandler(handler)
            if hasattr(handler, "close"):
                handler.close()
        managed_handlers = []
    else:
        for handler in root.handlers[:]:
            if not getattr(handler, "_hawker_managed", False) and isinstance(handler, logging.StreamHandler):
                if type(handler).__name__ == "LogCaptureHandler":
                    continue
                root.removeHandler(handler)
        managed_handlers = [h for h in root.handlers if getattr(h, "_hawker_managed", False)]

    # 1. 终端 Handler
    console_handler = next(
        (h for h in managed_handlers if getattr(h, "_hawker_handler_kind", "") == "console"),
        None,
    )
    if console_handler is None:
        if use_rich:
            console_handler = RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                log_time_format="[%X]",
            )
            console_handler.setFormatter(_build_formatter(for_rich=True))
        else:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(_build_formatter(with_color=False))
            
        console_handler._hawker_managed = True  # type: ignore
        console_handler._hawker_handler_kind = "console"  # type: ignore
        root.addHandler(console_handler)
    
    console_handler.setLevel(target_level)

    # 2. 文件 Handler
    existing_file_handlers = [
        h for h in root.handlers if getattr(h, "_hawker_handler_kind", "") == "file"
    ]
    desired_path = log_path.resolve() if log_path else None
    if desired_path is None:
        for handler in existing_file_handlers:
            root.removeHandler(handler)
            handler.close()
    else:
        desired_path.parent.mkdir(parents=True, exist_ok=True)
        active_file_handler: logging.Handler | None = None
        for handler in existing_file_handlers:
            handler_path = Path(getattr(handler, "baseFilename", "")).resolve()
            if handler_path == desired_path and active_file_handler is None:
                active_file_handler = handler
                continue
            root.removeHandler(handler)
            handler.close()

        if active_file_handler is None:
            active_file_handler = logging.FileHandler(desired_path, encoding="utf-8")
            active_file_handler._hawker_managed = True  # type: ignore
            active_file_handler._hawker_handler_kind = "file"  # type: ignore
            root.addHandler(active_file_handler)

        active_file_handler.setFormatter(_build_formatter(with_color=False))
        active_file_handler.setLevel(target_level)

    # 抑制三方库噪音
    noisy_loggers = [
        "litellm", "LiteLLM", "httpx", "httpcore", "urllib3", "browser_use", "playwright"
    ]
    for l_name in noisy_loggers:
        logger_obj = logging.getLogger(l_name)
        logger_obj.setLevel(logging.WARNING)
        logger_obj.propagate = False


def emit_observation(message: str) -> None:
    """
    专业的观测摘要发送器。
    1. 直接通过 sys.stdout 输出（被执行器拦截后成为大模型的 Observation）。
    2. 这种方式不带 logging 前缀，保持 Observation 纯净。
    """
    sink = _OBSERVATION_SINK.get()
    if sink is not None:
        sink.append(message)
        return
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def emit_tool_observation(tool_name: str, status: str, metrics: str = "", summary: str = "") -> None:
    """
    标准化工具观测摘要。
    格式: [{tool_name}] {status} | {metrics} | {summary}
    """
    parts = [f"[{tool_name}] {status}"]
    if metrics:
        parts.append(metrics)
    if summary:
        parts.append(summary)
    
    emit_observation(" | ".join(parts))
