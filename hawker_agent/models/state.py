from __future__ import annotations

import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path

from hawker_agent.models.item import ItemStore
from hawker_agent.observability import LogContext, bind_log_context, generate_trace_id


@dataclass
class TokenStats:
    """Token 用量和费用统计，按步骤累加。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def add(self, input_t: int | None, output_t: int | None, cached_t: int | None, cost: float | None) -> None:
        i = input_t or 0
        o = output_t or 0
        c = cached_t or 0
        f = cost or 0.0
        
        self.input_tokens += i
        self.output_tokens += o
        self.cached_tokens += c
        self.total_tokens += i + o
        self.cost += f

    def is_over_budget(self, limit: int) -> bool:
        return self.total_tokens >= limit


@dataclass
class CodeAgentState:
    """
    替换 run() 函数的 state dict（原 9 个无类型 key）。
    包含 Agent 生命周期内的所有可变状态。
    """

    # 终止标志
    done: bool = False
    answer: str = ""
    final_answer_requested: str | None = None

    # 数据采集
    items: ItemStore = field(default_factory=ItemStore)
    pending_dom: str | None = None

    # Token 预算
    token_stats: TokenStats = field(default_factory=TokenStats)

    # 进度追踪（用于无进展检测）
    activity_marker: int = 0
    progress_marker: int = 0

    # 运行元数据
    trace_id: str = field(default_factory=generate_trace_id)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    run_dir: Path | None = None
    started_at: float = field(default_factory=time.time)

    # 检查点管理
    checkpoint_files: set[str] = field(default_factory=set)

    def mark_activity(self) -> None:
        """采集到新数据时调用。"""
        self.activity_marker += 1
        self.progress_marker += 1

    def snapshot_markers(self) -> tuple[int, int]:
        """步骤开始前快照，用于步骤后对比进度。"""
        return self.activity_marker, self.progress_marker

    def is_over_budget(self, limit: int) -> bool:
        return self.token_stats.is_over_budget(limit)

    def bind_log_context(self, step: int | str | None = None) -> AbstractContextManager[LogContext]:
        """将 state 上的 trace_id/run_id 绑定到当前日志上下文。"""
        return bind_log_context(trace_id=self.trace_id, run_id=self.run_id, step=step)
