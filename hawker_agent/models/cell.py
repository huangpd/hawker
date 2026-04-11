from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hawker_agent.models.state import TokenStats


class CellStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


@dataclass(frozen=True)
class CodeCell:
    """
    单步的不可变事后记录，由 CodeAgentStepMetadata.to_cell() 生成。
    唯一消费者：storage/exporter.py 的 Jupyter Notebook 导出。
    不要在此添加任何运行时逻辑。
    """

    step: int
    thought: str
    source: str
    output: str | None
    error: str | None
    status: CellStatus
    duration: float
    usage: TokenStats
    url: str = ""
    items_count: int = 0
