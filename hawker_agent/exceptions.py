from __future__ import annotations


class CrawlerAgentError(Exception):
    """所有自定义异常的基类。"""


class BrowserError(CrawlerAgentError):
    """浏览器操作失败（导航、点击、CDP 错误等）。"""


class LLMError(CrawlerAgentError):
    """LLM 调用失败或响应异常。"""


class LLMResponseTruncated(LLMError):
    """LLM 响应被截断（status=incomplete 或输出接近上限）。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"LLM 响应截断: {reason}")


class ExecutionError(CrawlerAgentError):
    """代码执行沙箱内的异常。"""


class TokenBudgetExceeded(CrawlerAgentError):
    """Token 预算耗尽。"""


class NoProgressError(CrawlerAgentError):
    """连续多步无进展，触发终止。"""


class ConfigurationError(CrawlerAgentError):
    """配置缺失或无效（如 MODEL_NAME 未设置）。"""
