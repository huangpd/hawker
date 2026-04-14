from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LogContext:
    """日志上下文数据模型。"""
    trace_id: str = "-"
    run_id: str = "-"
    step: str = "-"


@dataclass
class Span:
    """
    结构化追踪单元模型。
    对应 OpenAI Agents SDK 中的 Span 概念。
    """
    trace_id: str
    span_id: str
    name: str
    parent_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: str = "running"  # running, success, error
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def elapsed(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class TraceProcessor(Protocol):
    """追踪处理器接口协议。"""
    def on_span_start(self, span: Span) -> None: ...
    def on_span_end(self, span: Span) -> None: ...
