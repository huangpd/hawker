from __future__ import annotations

import time
from dataclasses import dataclass, field

from hawker_agent.models.cell import CellStatus, CodeCell
from hawker_agent.models.output import CodeAgentModelOutput
from hawker_agent.models.state import CodeAgentState, TokenStats


@dataclass
class CodeAgentStepMetadata:
    """
    单步的运行时上下文，执行期间存活，结束后固化为 CodeCell。

    与 CodeCell 的职责区分：
    - CodeAgentStepMetadata：可变，运行时，包含进度判断逻辑
    - CodeCell：不可变，事后记录，仅用于 Notebook 导出

    不要合并这两个类。
    """

    step_no: int
    activity_before: int = 0
    progress_before: int = 0
    started_at: float = field(default_factory=time.time)

    output: str = ""
    error: str | None = None

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def has_progress(self, state: CodeAgentState) -> bool:
        """判断本步是否有实际进展（新数据 / 任务完成 / 代码成功执行）。"""
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
        """步骤结束后将运行时上下文固化为不可变的 CodeCell 记录。"""
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
