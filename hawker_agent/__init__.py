"""hawker_agent — LLM 驱动的自主网络爬虫 Agent。"""
from __future__ import annotations

from hawker_agent.exceptions import CrawlerAgentError

__all__ = [
    "run",
    "CodeAgentResult",
    "CodeAgentState",
    "CrawlerAgentError",
]


def __getattr__(name: str):
    """延迟导出重量级对象，避免包导入阶段形成环依赖。"""
    if name == "run":
        from hawker_agent.agent.runner import run as _run

        return _run
    if name == "CodeAgentResult":
        from hawker_agent.models.result import CodeAgentResult as _CodeAgentResult

        return _CodeAgentResult
    if name == "CodeAgentState":
        from hawker_agent.models.state import CodeAgentState as _CodeAgentState

        return _CodeAgentState
    raise AttributeError(name)
