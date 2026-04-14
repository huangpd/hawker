from __future__ import annotations

import time
from dataclasses import dataclass, field

from hawker_agent.models.cell import CellStatus, CodeCell
from hawker_agent.models.output import CodeAgentModelOutput
from hawker_agent.models.state import CodeAgentState, TokenStats


@dataclass
class CodeAgentStepMetadata:
    """单步执行的运行时上下文记录。

    该类记录了单个步骤执行期间的动态状态，并包含用于判断该步骤是否取得实质进展的逻辑。
    步骤结束后，该上下文可被固化为不可变的 CodeCell 以供持久化存储。

    Attributes:
        step_no (int): 当前步骤序号。
        activity_before (int): 步骤开始前的基础活跃度快照。
        progress_before (int): 步骤开始前的实质进展快照。
        started_at (float): 步骤启动的时间戳（秒）。
        output (str): 本步骤代码执行的输出内容。
        error (str | None): 本步骤代码执行过程中产生的错误信息（如有）。
    """

    step_no: int
    activity_before: int = 0
    progress_before: int = 0
    started_at: float = field(default_factory=time.time)

    output: str = ""
    error: str | None = None

    def elapsed(self) -> float:
        """计算自步骤开始以来经过的时间。

        Returns:
            float: 耗时秒数。
        """
        return time.time() - self.started_at

    def has_progress(self, state: CodeAgentState) -> bool:
        """判断本步骤是否取得了实质性进展。

        判断标准包括：采集到了新数据、任务被标记为完成、或者代码执行未报错。

        Args:
            state (CodeAgentState): 当前代理的全局状态。

        Returns:
            bool: 若有进展则返回 True，否则返回 False。
        """
        return (
            state.activity_marker > self.activity_before
            or state.progress_marker > self.progress_before
            or state.done
            or not self.error
        )

    def to_cell(
        self,
        model_output: CodeAgentModelOutput,
        usage: TokenStats,
        items_count: int,
    ) -> CodeCell:
        """在步骤结束后，将当前的运行时上下文转化为不可变的 CodeCell 对象。

        Args:
            model_output (CodeAgentModelOutput): 模型在该步骤生成的思考和代码。
            usage (TokenStats): 该步骤消耗的 token 统计。
            items_count (int): 该步骤累计采集到的项目数量。

        Returns:
            CodeCell: 固化后的步骤记录对象，用于后续导出。
        """
        return CodeCell(
            step=self.step_no,
            thought=model_output.thought,
            source=model_output.code,
            output=self.output or None,
            error=self.error,
            status=CellStatus.ERROR if self.error else CellStatus.SUCCESS,
            duration=self.elapsed(),
            usage=usage,
            items_count=items_count,
        )
