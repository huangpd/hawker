from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hawker_agent.browser.session import BrowserSession

logger = logging.getLogger(__name__)


async def get_cdp(session: BrowserSession):
    """获取当前 focus 页面的 CDP session。"""
    raw = session.raw
    target_id = raw.agent_focus_target_id
    cdp = await raw.get_or_create_cdp_session(target_id=target_id, focus=False)
    return cdp


async def run_js(session: BrowserSession, expression: str) -> str:
    """通过 CDP Runtime.evaluate 执行任意 JS，返回字符串结果。"""
    cdp = await get_cdp(session)
    result = await cdp.cdp_client.send.Runtime.evaluate(
        params={"expression": expression, "returnByValue": True, "awaitPromise": True},
        session_id=cdp.session_id,
    )
    if "exceptionDetails" in result:
        detail = result["exceptionDetails"]
        text = detail.get("text", "")
        exc = detail.get("exception", {})
        desc = exc.get("description", "") if isinstance(exc, dict) else ""
        msg = f"[JS错误] {text} {desc}"
        logger.debug("JS 执行异常: %s", msg)
        return msg

    value = result.get("result", {}).get("value")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
