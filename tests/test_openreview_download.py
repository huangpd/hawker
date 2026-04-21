from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hawker_agent.browser.session import BrowserSession
from hawker_agent.tools.browser_tools import register_browser_tools
from hawker_agent.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_openreview_pdf_browser_download(tmp_path: Path) -> None:
    """验证 OpenReview PDF 链接可通过 browser_download 落地到本地文件。"""
    run_dir = tmp_path / "openreview_download"
    run_dir.mkdir()

    pdf_url = "https://openreview.net/pdf?id=ybA4EcMmUZ"
    filename = "86.pdf"

    registry = ToolRegistry()
    history = MagicMock()

    async with BrowserSession(headless=True) as session:
        register_browser_tools(registry, session, history)
        browser_download = registry.as_namespace_dict()["browser_download"]

        # 先打开论文页，确保浏览器会话和 Cookie 状态就绪。
        await session.raw.navigate_to("https://openreview.net/forum?id=ybA4EcMmUZ")

        result = await browser_download(pdf_url, filename=filename, run_dir=str(run_dir))

    saved_path = run_dir / filename
    assert result["ok"] is True
    assert result["filename"] == filename
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0
    with open(saved_path, "rb") as f:
        assert f.read(4) == b"%PDF"
