from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hawker_agent.browser.session import BrowserSession

logger = logging.getLogger(__name__)


async def get_cdp(session: BrowserSession):
    """获取当前 focus 页面的 CDP 会话。

    Args:
        session (BrowserSession): 浏览器会话对象。

    Returns:
        CDPSession: 当前 focus 页面的 CDP 会话对象。
    """
    raw = session.raw
    target_id = raw.agent_focus_target_id
    cdp = await raw.get_or_create_cdp_session(target_id=target_id, focus=False)
    return cdp


async def run_js(session: BrowserSession, expression: str) -> Any:
    """通过 CDP Runtime.evaluate 执行任意 JavaScript 代码，并返回原生 Python 对象。

    支持自动处理 Illegal return statement 异常，通过包裹异步 IIFE 进行重试。

    Args:
        session (BrowserSession): 浏览器会话对象。
        expression (str): 要执行的 JavaScript 表达式或代码块。

    Returns:
        Any: 执行结果的原生 Python 对象。如果执行出错，返回以 "[JS错误]" 开头的错误描述。
    """
    clean_expr = expression.strip()
    cdp = await get_cdp(session)

    async def _eval(expr: str) -> dict:
        return await cdp.cdp_client.send.Runtime.evaluate(
            params={
                "expression": expr,
                "returnByValue": True,
                "awaitPromise": True,
                "replMode": True,  # 允许重复声明 let/const，解决 Already declared 报错
            },
            session_id=cdp.session_id,
        )

    result = await _eval(clean_expr)

    # 异常处理与启发式重试
    if "exceptionDetails" in result:
        detail = result["exceptionDetails"]
        text = detail.get("text", "")
        exc = detail.get("exception", {})
        desc = exc.get("description", "") if isinstance(exc, dict) else ""
        
        # 如果大模型习惯性地写了顶级 return (Illegal return statement)
        if "Illegal return statement" in text or "Illegal return statement" in desc:
            logger.debug("检测到非法的顶级 return，自动包裹异步 IIFE 进行重试")
            # 既然有 return，包进函数里就一定能拿到正确的返回值
            wrapped_expr = f"(async () => {{\n{clean_expr}\n}})()"
            result = await _eval(wrapped_expr)
            
            if "exceptionDetails" in result:
                detail = result["exceptionDetails"]
                text = detail.get("text", "")
                exc = detail.get("exception", {})
                desc = exc.get("description", "") if isinstance(exc, dict) else ""
                msg = f"[JS错误] {text} {desc}"
                logger.debug("JS 自动包裹后执行异常: %s", msg)
                return msg
        else:
            msg = f"[JS错误] {text} {desc}"
            logger.debug("JS 执行异常: %s", msg)
            return msg

    value = result.get("result", {}).get("value")
    return "" if value is None else value
