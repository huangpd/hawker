from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from hawker_agent.models.state import TokenStats


@dataclass
class CodeAgentResult:
    """
    run() 的返回值，替换原来的裸 str。
    __str__ 返回 answer 以保持向后兼容（旧代码 print(result) 仍可用）。
    """

    # 核心输出
    answer: str
    success: bool

    # 采集数据
    items: list[dict] = field(default_factory=list)

    # 运行元数据
    run_id: str = ""
    total_steps: int = 0
    total_duration: float = 0.0
    token_stats: TokenStats = field(default_factory=TokenStats)
    stop_reason: Literal["done", "token_budget", "no_progress", "max_steps"] = "done"

    # 产物路径（任务失败时可能为 None）
    run_dir: Path | None = None
    log_path: Path | None = None
    notebook_path: Path | None = None
    result_json_path: Path | None = None

    @property
    def items_count(self) -> int:
        return len(self.items)

    def __str__(self) -> str:
        """向后兼容：str(result) 或 print(result) 仍输出 answer。"""
        return self.answer
