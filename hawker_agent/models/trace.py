from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LogContext:
    """日志上下文数据模型。

    用于在日志记录中携带追踪和标识信息，方便日志的聚合与检索。

    Attributes:
        trace_id (str): 全链路追踪 ID。
        run_id (str): 运行实例 ID。
        step (str): 当前步骤标识。
    """
    trace_id: str = "-"
    run_id: str = "-"
    step: str = "-"


@dataclass
class Span:
    """结构化追踪单元模型。

    对应追踪系统中的一个基本操作单元（Span），记录其起始时间、耗时、状态及相关元数据。

    Attributes:
        trace_id (str): 所属链路的追踪 ID。
        span_id (str): 本单元的唯一标识 ID。
        name (str): 追踪单元的名称（如方法名或操作名）。
        parent_id (str | None): 父级追踪单元的 ID。
        start_time (float): 单元启动的时间戳（秒）。
        end_time (float | None): 单元结束的时间戳（秒）。
        status (str): 当前状态（如 running, success, error）。
        data (dict[str, Any]): 业务相关的数据载荷。
        metadata (dict[str, Any]): 追踪相关的辅助元数据。
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
    external_observation: Any | None = None
    external_context_manager: Any | None = None

    def elapsed(self) -> float:
        """计算该追踪单元的持续时长。

        Returns:
            float: 持续秒数。
        """
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class TraceProcessor(Protocol):
    """追踪处理器接口协议。"""
    def on_span_start(self, span: Span) -> None: ...
    def on_span_end(self, span: Span) -> None: ...
