from __future__ import annotations

import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from hawker_agent.models.item import ItemStore
from hawker_agent.observability import LogContext, bind_log_context, generate_trace_id


@dataclass
class TokenStats:
    """Token 消耗及费用统计管理器。

    用于实时统计模型交互中的输入、输出、缓存 token 数量以及产生的预估费用。

    Attributes:
        input_tokens (int): 累计输入的 token 数量。
        output_tokens (int): 累计输出的 token 数量。
        cached_tokens (int): 累计命中的缓存 token 数量。
        total_tokens (int): 累计的总 token 数量。
        cost (float): 累计产生的预估费用（单位通常为美元）。
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def add(self, input_t: int | None, output_t: int | None, cached_t: int | None, cost: float | None) -> None:
        """累加单次交互的 token 和费用。

        Args:
            input_t (int | None): 本次输入的 token 数。
            output_t (int | None): 本次输出的 token 数。
            cached_t (int | None): 本次缓存命中的 token 数。
            cost (float | None): 本次交互产生的费用。
        """
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
        """检查当前总 token 消耗是否已超过指定预算。

        Args:
            limit (int): 允许的 token 预算上限。

        Returns:
            bool: 若已超标则返回 True，否则返回 False。
        """
        return self.total_tokens >= limit


@dataclass
class CodeAgentState:
    """代理生命周期内的可变状态管理器。

    该类统一管理代理运行期间的所有状态，包括数据采集进度、token 预算、生命周期标志以及运行元数据等。

    Attributes:
        done (bool): 代理是否已判定任务完成。
        answer (str): 代理生成的最终回复文本。
        final_answer_requested (str | None): 标志模型是否已请求输出最终答案。
        items (ItemStore): 采集到的数据项存储库。
        pending_dom (str | None): 下一轮模型交互待注入的 DOM 状态文本。
        last_dom_snapshot (dict[str, Any] | None): 最近一次页面的结构化快照。
        llm_records (list[dict[str, Any]]): 详细的模型交互过程记录。
        token_stats (TokenStats): 当前运行的 token 统计信息。
        activity_marker (int): 基础活跃度计数器，用于追踪采集项的变动。
        progress_marker (int): 实质进展计数器，反映任务目标的推进。
        no_progress_streak (int): 连续无进展的步数累计。
        memory_guided_dom_steps_remaining (int): 命中高置信度记忆后，剩余多少步压制主动 full DOM。
        memory_guided_reason (str): 当前记忆引导策略的说明。
        trace_id (str): 全链路追踪 ID。
        run_id (str): 本次运行的短 ID（12位）。
        run_dir (Path | None): 运行产物的输出目录。
        started_at (float): 任务启动的时间戳（秒）。
        checkpoint_files (set[str]): 已创建的检查点文件集合。
    """

    # 终止标志
    done: bool = False
    answer: str = ""
    final_answer_requested: str | None = None
    final_artifact_requested: dict[str, Any] | None = None
    final_artifact: dict[str, Any] | None = None
    expected_output_format: Literal["text", "json", "markdown"] | None = None

    # 数据采集
    items: ItemStore = field(default_factory=ItemStore)
    pending_dom: str | None = None
    last_dom_snapshot: dict[str, Any] | None = None
    llm_records: list[dict[str, Any]] = field(default_factory=list)
    healing_records: list[dict[str, Any]] = field(default_factory=list)
    evaluator_records: list[dict[str, Any]] = field(default_factory=list)

    # Token 预算
    token_stats: TokenStats = field(default_factory=TokenStats)

    # 进度追踪（用于无进展检测）
    activity_marker: int = 0
    progress_marker: int = 0
    no_progress_streak: int = 0
    memory_guided_dom_steps_remaining: int = 0
    memory_guided_reason: str = ""

    # 运行元数据
    trace_id: str = field(default_factory=generate_trace_id)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    run_dir: Path | None = None
    log_dir: Path | None = None
    started_at: float = field(default_factory=time.time)

    # 检查点管理
    checkpoint_files: set[str] = field(default_factory=set)

    def mark_activity(self) -> None:
        """记录发生了有效活动（如采集到新数据）。

        该操作将推进 activity_marker 和 progress_marker。
        """
        self.activity_marker += 1
        self.progress_marker += 1

    def snapshot_markers(self) -> tuple[int, int]:
        """获取当前活跃度和进展计数器的快照。

        常在步骤开始前调用，以便在步骤结束后判断该步是否有实质进展。

        Returns:
            tuple[int, int]: 包含 (activity_marker, progress_marker) 的元组。
        """
        return self.activity_marker, self.progress_marker

    def is_over_budget(self, limit: int) -> bool:
        """检查当前运行是否已超出 token 预算。

        Args:
            limit (int): 预算上限值。

        Returns:
            bool: 是否超限。
        """
        return self.token_stats.is_over_budget(limit)

    def bind_log_context(self, step: int | str | None = None) -> AbstractContextManager[LogContext]:
        """将当前状态的追踪 ID 和运行 ID 绑定到日志上下文中。

        Args:
            step (int | str | None, optional): 当前执行的步骤标识。

        Returns:
            AbstractContextManager[LogContext]: 一个上下文管理器，用于在 with 语句块中应用日志绑定。
        """
        return bind_log_context(trace_id=self.trace_id, run_id=self.run_id, step=step)
