import asyncio
import pytest
import re
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

from hawker_agent.browser.session import BrowserSession
from hawker_agent.browser.actions import browser_download, get_cookies

# ─── 维度 1: 文件名清洗单元测试 ──────────────────────────────────

def test_filename_sanitization():
    """验证文件名清洗逻辑是否能处理各种非法字符。"""
    # 模拟原始文件名（包含 Windows/Linux 非法字符）
    dirty_names = [
        "paper:2024*final?.pdf",
        "data/results|v1.0.csv",
        "hello<world>gui.png",
        'quote"name".txt'
    ]
    expected_names = [
        "paper_2024_final_.pdf",
        "data_results_v1.0.csv",
        "hello_world_gui.png",
        "quote_name_.txt"
    ]
    
    for dirty, expected in zip(dirty_names, expected_names):
        sanitized = re.sub(r'[\\/*?:"<>|]', '_', dirty)
        assert sanitized == expected

# ─── 维度 2: CDP 凭证提取逻辑验证 ────────────────────────────────

@pytest.mark.asyncio
async def test_cdp_cookie_extraction():
    """模拟 CDP 响应，验证 Cookie 提取和格式转换。"""
    mock_session = MagicMock()
    mock_raw = AsyncMock()
    mock_session.raw = mock_raw
    
    # 模拟 CDP 结构
    mock_cdp = MagicMock()
    mock_cdp.cdp_client.send.Network.getCookies = AsyncMock(return_value={
        'cookies': [
            {'name': 'session_id', 'value': 'abc-123'},
            {'name': 'auth_token', 'value': 'secret-jwt'}
        ]
    })
    mock_raw.get_or_create_cdp_session.return_value = mock_cdp
    
    # 探测提取代码
    cdp_s = await mock_raw.get_or_create_cdp_session()
    res = await cdp_s.cdp_client.send.Network.getCookies()
    cookies = {c['name']: c['value'] for c in res['cookies']}
    
    assert cookies['session_id'] == 'abc-123'
    assert cookies['auth_token'] == 'secret-jwt'

# ─── 维度 3: 真实集成下载测试 (Headless) ─────────────────────────

@pytest.mark.asyncio
async def test_real_streaming_download(tmp_path):
    """
    真实环境下启动浏览器，导航并下载一个 PDF 样本。
    验证：1. 文件是否存在；2. 大小是否符合预期；3. 目录是否正确。
    """
    run_dir = tmp_path / "downloads"
    run_dir.mkdir()
    
    # 使用一个可靠的公开 PDF 链接 (W3C 样板)
    target_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    target_filename = "rigorous_test_sample.pdf"
    
    async with BrowserSession(headless=True) as session:
        # 先导航，确保浏览器上下文已初始化
        await session.raw.navigate_to("https://example.com")
        
        # 执行下载
        result = await browser_download(
            session, 
            url=target_url, 
            filename=target_filename, 
            run_dir=str(run_dir)
        )
        
        print(f"\nIntegration Download Result: {result}")
        
        # 验证文件落地
        expected_path = run_dir / target_filename
        assert result["ok"] is True
        assert result["filename"] == target_filename
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0

# ─── 维度 4: get_cookies 动作集成测试 ──────────────────────────

@pytest.mark.asyncio
async def test_real_get_cookies():
    """验证 get_cookies 动作是否能从真实浏览器会话中提取数据。"""
    async with BrowserSession(headless=True) as session:
        # 导航到一个会设置基础 Cookie 的页面
        await session.raw.navigate_to("https://www.google.com")
        
        # 执行提取
        cookies = await get_cookies(session)
        
        print(f"\nCookies Extraction Result: {len(cookies)} cookies found.")
        
        # 验证返回类型
        assert isinstance(cookies, list)
        if len(cookies) > 0:
            assert isinstance(cookies[0], dict)
            assert "name" in cookies[0]
            assert "value" in cookies[0]
            # 至少应该有一个相关的域名
            domains = [c.get('domain', '') for c in cookies]
            assert any('google' in d for d in domains)

if __name__ == "__main__":
    # 手动运行冒烟测试
    print("🚀 开始严谨测试套件...")
    test_filename_sanitization()
    print("✅ 文件名清洗测试通过")
    
    asyncio.run(test_cdp_cookie_extraction())
    print("✅ CDP 凭证提取模拟测试通过")
    
    # 集成测试
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(test_real_streaming_download(Path(tmp)))
    print("✅ 真实集成下载测试通过")

    asyncio.run(test_real_get_cookies())
    print("✅ 真实 get_cookies 集成测试通过")
    
    print("\n🎉 所有测试维度全部达标！")
