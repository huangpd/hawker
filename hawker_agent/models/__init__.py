"""代理数据模型的懒加载导出。

避免包初始化阶段一次性导入所有 model，减少环依赖风险。
"""
from __future__ import annotations

__all__ = [
    "CellStatus",
    "CodeAgentHistoryList",
    "CodeAgentModelOutput",
    "CodeAgentResult",
    "CodeAgentState",
    "CodeAgentStepMetadata",
    "CodeCell",
    "ItemStore",
    "TokenStats",
    "LogContext",
    "Span",
]


def __getattr__(name: str):
    if name in {"CellStatus", "CodeCell"}:
        from hawker_agent.models.cell import CellStatus, CodeCell

        return {"CellStatus": CellStatus, "CodeCell": CodeCell}[name]
    if name == "CodeAgentHistoryList":
        from hawker_agent.models.history import CodeAgentHistoryList

        return CodeAgentHistoryList
    if name == "ItemStore":
        from hawker_agent.models.item import ItemStore

        return ItemStore
    if name == "CodeAgentModelOutput":
        from hawker_agent.models.output import CodeAgentModelOutput

        return CodeAgentModelOutput
    if name == "CodeAgentResult":
        from hawker_agent.models.result import CodeAgentResult

        return CodeAgentResult
    if name in {"CodeAgentState", "TokenStats"}:
        from hawker_agent.models.state import CodeAgentState, TokenStats

        return {"CodeAgentState": CodeAgentState, "TokenStats": TokenStats}[name]
    if name == "CodeAgentStepMetadata":
        from hawker_agent.models.step import CodeAgentStepMetadata

        return CodeAgentStepMetadata
    if name in {"LogContext", "Span"}:
        from hawker_agent.models.trace import LogContext, Span

        return {"LogContext": LogContext, "Span": Span}[name]
    raise AttributeError(name)
