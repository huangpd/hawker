from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hawker_agent.models.state import TokenStats


class CellStatus(str, Enum):
    """单元格执行状态枚举。"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


@dataclass(frozen=True)
class CodeCell:
    """单步的不可变事后记录，由 CodeAgentStepMetadata.to_cell() 生成。

    该类仅用于存储步骤的执行结果，作为 Jupyter Notebook 导出的唯一数据源。
    请勿在此类中添加任何运行时逻辑。

    Attributes:
        step (int): 步骤序号。
        thought (str): 模型在执行该步骤时的思考过程。
        source (str): 执行的源代码。
        output (str | None): 代码执行的输出内容。
        error (str | None): 执行过程中产生的错误信息（如有）。
        status (CellStatus): 单元格执行状态。
        duration (float): 步骤执行耗时（秒）。
        usage (TokenStats): 该步骤消耗的 token 统计。
        url (str): 执行步骤时的页面 URL（可选）。
        items_count (int): 该步骤采集到的项目数量。
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
