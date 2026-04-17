from __future__ import annotations

import contextlib
import contextvars
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from hawker_agent.langfuse_client import end_observation, start_observation, update_observation
from hawker_agent.models.trace import LogContext, Span, TraceProcessor

_LOG_CONTEXT: contextvars.ContextVar[LogContext] = contextvars.ContextVar(
    "hawker_log_context",
    default=LogContext(),
)
_CURRENT_SPAN: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "hawker_current_span",
    default=None,
)
_TRACE_PROCESSORS: list[TraceProcessor] = []
_OBSERVATION_SINK: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "hawker_observation_sink",
    default=None,
)
_BASE_RECORD_FACTORY = logging.getLogRecordFactory()
_RECORD_FACTORY_INSTALLED = False


def generate_trace_id() -> str:
    """生成一个 32 位的十六进制追踪 ID (trace ID)。

    Returns:
        str: 随机生成的 32 位十六进制字符串。
    """
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """生成一个 16 位的十六进制跨度 ID (span ID)。

    Returns:
        str: 随机生成的 16 位十六进制字符串。
    """
    return uuid.uuid4().hex[:16]


def get_log_context() -> LogContext:
    """获取当前协程或线程的日志上下文。

    Returns:
        LogContext: 当前的日志上下文。
    """
    return _LOG_CONTEXT.get()


def get_current_span() -> Span | None:
    """获取当前活动的跨度 (span)。

    Returns:
        Span | None: 如果存在活动跨度，则返回该跨度；否则返回 None。
    """
    return _CURRENT_SPAN.get()


def add_trace_processor(processor: TraceProcessor) -> None:
    """注册一个追踪处理器 (trace processor)。

    Args:
        processor (TraceProcessor): 要添加到注册表中的处理器。
    """
    processor_type = type(processor)
    for existing in list(_TRACE_PROCESSORS):
        if type(existing) is processor_type:
            return
    _TRACE_PROCESSORS.append(processor)


def remove_trace_processor(processor: TraceProcessor) -> None:
    """移除一个追踪处理器。"""
    try:
        _TRACE_PROCESSORS.remove(processor)
    except ValueError:
        pass


@contextlib.contextmanager
def trace(name: str, **metadata: Any) -> Iterator[Span]:
    """跨度的核心追踪上下文管理器。

    处理跨度的自动嵌套、计时和异常捕获。

    Args:
        name (str): 跨度的名称。
        **metadata (Any): 附加到跨度的额外元数据。

    Yields:
        Iterator[Span]: 创建的跨度对象。

    Raises:
        Exception: 捕获并重新抛出上下文中发生的任何异常，并在抛出前更新跨度状态。
    """
    parent = get_current_span()
    
    # 优先沿用：1. 父 Span 的 ID; 2. 现有日志上下文中的 ID; 3. 生成新 ID
    existing_ctx = get_log_context()
    if parent:
        trace_id = parent.trace_id
    elif existing_ctx.trace_id and existing_ctx.trace_id != "-":
        trace_id = existing_ctx.trace_id
    else:
        trace_id = generate_trace_id()
    
    span = Span(
        trace_id=trace_id,
        span_id=generate_span_id(),
        parent_id=parent.span_id if parent else None,
        name=name,
        metadata=metadata,
    )

    observation_type = "tool" if metadata.get("is_tool") else metadata.get("as_type", "span")
    observation_input = metadata.get("input")
    observation_model = metadata.get("model")
    observation_metadata = {
        "hawker_trace_id": trace_id,
        "hawker_span_id": span.span_id,
        "hawker_parent_id": span.parent_id,
        **metadata,
    }
    obs, obs_ctx = start_observation(
        name=name,
        input=observation_input,
        metadata=observation_metadata,
        as_type=observation_type,
        model=observation_model,
        parent_observation=parent.external_observation if parent else None,
    )
    span.external_observation = obs
    span.external_context_manager = obs_ctx
    
    # 兼容现有日志系统：绑定 trace_id 和 step
    log_ctx_token = _LOG_CONTEXT.set(LogContext(
        trace_id=trace_id,
        run_id=get_log_context().run_id,
        step=name
    ))
    
    span_token = _CURRENT_SPAN.set(span)
    
    for proc in _TRACE_PROCESSORS:
        try:
            proc.on_span_start(span)
        except Exception:
            logging.getLogger(__name__).exception("Error in trace processor on_span_start")

    try:
        yield span
        span.status = "success"
    except Exception as e:
        span.status = "error"
        span.data["error"] = str(e)
        span.data["error_type"] = type(e).__name__
        update_observation(span.external_observation, level="ERROR", status_message=str(e))
        raise
    finally:
        span.end_time = time.time()
        update_observation(
            span.external_observation,
            output=span.data if span.data else None,
            metadata={**span.metadata, "status": span.status, "duration_s": round(span.elapsed(), 4)},
        )
        for proc in _TRACE_PROCESSORS:
            try:
                proc.on_span_end(span)
            except Exception:
                logging.getLogger(__name__).exception("Error in trace processor on_span_end")
        
        end_observation(span.external_context_manager)
        _CURRENT_SPAN.reset(span_token)
        _LOG_CONTEXT.reset(log_ctx_token)


class LoggingTraceProcessor:
    """默认的追踪处理器，将跨度事件转换为结构化日志。"""

    def __init__(self, logger: logging.Logger | None = None):
        """初始化日志追踪处理器。

        Args:
            logger (logging.Logger | None): 要使用的日志记录器。默认为 "hawker.trace"。
        """
        self.logger = logger or logging.getLogger("hawker.trace")

    def on_span_start(self, span: Span) -> None:
        """当跨度开始时调用。

        Args:
            span (Span): 已启动的跨度。
        """
        pass

    def on_span_end(self, span: Span) -> None:
        """当跨度结束时调用，记录其状态和持续时间。

        Args:
            span (Span): 已结束的跨度。
        """
        level = logging.INFO if span.status == "success" else logging.ERROR
        
        # 针对工具调用，输出更精简的日志
        if span.metadata.get("is_tool"):
            msg = f"  [Tool] {span.name} -> {span.status} ({span.elapsed():.3f}s)"
        else:
            msg = f"Span Finished: {span.name} | Status: {span.status} | Duration: {span.elapsed():.3f}s"
            if span.parent_id:
                msg = f"  {msg} (parent: {span.parent_id})"
        
        self.logger.log(level, msg, extra={
            "span_id": span.span_id,
            "parent_id": span.parent_id,
            "span_data": span.data,
            "span_metadata": span.metadata
        })


class ToolStatsProcessor:
    """聚合工具调用的统计信息。"""

    def __init__(self):
        """初始化工具统计处理器。"""
        self.stats: dict[str, dict[str, Any]] = {}

    def on_span_start(self, span: Span) -> None:
        """当跨度开始时调用。

        Args:
            span (Span): 已启动的跨度。
        """
        pass

    def on_span_end(self, span: Span) -> None:
        """当跨度结束时调用，更新工具调用的统计信息。

        Args:
            span (Span): 已结束的跨度。
        """
        if not span.metadata.get("is_tool"):
            return
        
        name = span.name.replace("tool_", "")
        if name not in self.stats:
            self.stats[name] = {"calls": 0, "success": 0, "error": 0, "total_time": 0.0}
        
        s = self.stats[name]
        s["calls"] += 1
        s["total_time"] += span.elapsed()
        if span.status == "success":
            s["success"] += 1
        else:
            s["error"] += 1

    def get_summary(self) -> str:
        """生成所有工具调用的摘要字符串。

        Returns:
            str: 格式化的表格，总结工具调用次数、成功、失败和耗时。
        """
        if not self.stats:
            return "没有工具调用记录。"
        
        lines = ["\n🛠️  工具调用汇总统计:"]
        lines.append(f"{'工具名':<20} | {'次数':<5} | {'成功':<5} | {'失败':<5} | {'平均耗时':<8}")
        lines.append("-" * 65)
        
        for name, s in sorted(self.stats.items(), key=lambda x: x[1]["calls"], reverse=True):
            avg = s["total_time"] / s["calls"]
            lines.append(f"{name:<20} | {s['calls']:<5} | {s['success']:<5} | {s['error']:<5} | {avg:.3f}s")
        
        return "\n".join(lines)


def clear_log_context() -> None:
    """清除当前日志上下文并重置为默认值。"""
    _LOG_CONTEXT.set(LogContext())
    _CURRENT_SPAN.set(None)


def set_log_context(
    *,
    trace_id: str | None = None,
    run_id: str | None = None,
    step: int | str | None = None,
) -> LogContext:
    """直接更新当前日志上下文。

    Args:
        trace_id (str | None): 可选的追踪 ID。
        run_id (str | None): 可选的运行 ID。
        step (int | str | None): 可选的步骤标识符。

    Returns:
        LogContext: 更新后的日志上下文。
    """
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
    """在作用域内临时绑定日志上下文。

    Args:
        trace_id (str | None): 可选的追踪 ID。
        run_id (str | None): 可选的运行 ID。
        step (int | str | None): 可选的步骤标识符。

    Yields:
        Iterator[LogContext]: 临时生效的日志上下文。
    """
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
    """安装自定义日志记录工厂，将追踪上下文注入到日志记录中。"""
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
    """将日志级别规范化为其整数表示。

    Args:
        level (int | str): 整数或字符串形式的日志级别。

    Returns:
        int: 整数形式的日志级别。
    """
    if isinstance(level, int):
        return level
    value = logging.getLevelName(level.upper())
    if isinstance(value, int):
        return value
    return logging.INFO


def _build_formatter(with_color: bool = False, for_rich: bool = False) -> logging.Formatter:
    """构建日志格式化器。

    Args:
        with_color (bool): 是否包含 ANSI 颜色代码。默认为 False。
        for_rich (bool): 是否构建针对 RichHandler 优化的格式化器。默认为 False。

    Returns:
        logging.Formatter: 配置好的格式化器。
    """
    if for_rich:
        # RichHandler 已经处理了时间、级别和颜色，我们只需提供上下文信息
        return logging.Formatter("[%(trace_id).8s] [%(step)s] %(message)s")
    
    if with_color:
        return logging.Formatter("\033[2m%(trace_id).8s\033[0m [%(step)s] %(message)s")
    
    # 文件日志：精简格式，保持 ID 长度一致 (8位)
    return logging.Formatter(
        fmt="%(asctime)s %(levelname).1s [%(trace_id).8s] [%(step)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


@contextlib.contextmanager
def collect_observations() -> Iterator[list[str]]:
    """收集观察结果到缓冲区中的上下文管理器。

    这可以防止观察结果被直接写入标准输出。

    Yields:
        Iterator[list[str]]: 将接收观察结果的列表缓冲区。
    """
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
    """初始化并配置日志系统。

    设置具有追踪上下文注入功能的终端和文件日志处理器。

    Args:
        level (int | str): 日志级别。默认为 logging.INFO。
        log_path (Path | None): 日志文件路径。如果为 None，则禁用文件日志。
        force (bool): 如果为 True，则替换现有的处理器。默认为 False。
    """
    _install_log_record_factory()
    
    # 注册默认 Tracing 处理器
    add_trace_processor(LoggingTraceProcessor())
    
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
                console=None,  # Rich 默认使用 stderr，但我们显式指定
            )
            from rich.console import Console
            console_handler.console = Console(stderr=True)
            console_handler.setFormatter(_build_formatter(for_rich=True))
        else:
            # 显式使用 sys.stderr，避免被 executor 的 redirect_stdout 捕获
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
    """发出一条观察消息。

    观察结果要么缓存在观察接收器中，要么写入标准输出。
    这绕过了标准日志系统，为 LLM 提供整洁的输出。

    Args:
        message (str): 要发出的观察消息。
    """
    sink = _OBSERVATION_SINK.get()
    if sink is not None:
        sink.append(message)
        return
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def emit_tool_observation(tool_name: str, status: str, metrics: str = "", summary: str = "") -> None:
    """发出标准化的工具观察消息。

    格式：[{tool_name}] {status} | {metrics} | {summary}

    Args:
        tool_name (str): 工具名称。
        status (str): 工具执行状态。
        metrics (str): 性能指标或其他数字数据。默认为 ""。
        summary (str): 结果的简短摘要。默认为 ""。
    """
    parts = [f"[{tool_name}] {status}"]
    if metrics:
        parts.append(metrics)
    if summary:
        parts.append(summary)
    
    emit_observation(" | ".join(parts))
