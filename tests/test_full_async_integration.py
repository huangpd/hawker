import asyncio
import pytest
import json
from pathlib import Path
from hawker_agent.agent.executor import execute
from hawker_agent.agent.namespace import build_namespace
from hawker_agent.models.state import CodeAgentState
from hawker_agent.tools.registry import ToolRegistry

@pytest.mark.asyncio
async def test_full_async_tools_execution(tmp_path):
    """测试所有核心工具在全异步模式下的执行。"""
    state = CodeAgentState()
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    
    # 模拟工具集
    registry = ToolRegistry()
    
    # 模拟一个异步浏览器工具
    async def mock_nav(url: str):
        return f"Visited {url}"
    
    async def mock_download(url: str):
        return f"Downloaded {url}"
        
    registry.register(mock_nav, name="nav")
    registry.register(mock_download, name="browser_download")
    
    # 构建命名空间 (此时 append_items 等已是异步)
    ns = build_namespace(state, registry.as_namespace_dict(), str(run_dir))
    
    # 场景：Agent 执行一段全异步代码
    code = """
# 1. 异步导航
res = await nav("https://example.com")
# 2. 异步提交数据
items = [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}]
await append_items(items)
# 3. 异步保存检查点
await save_checkpoint("test.json")
# 4. 异步下载
await browser_download("https://file.pdf")
# 5. 定义变量并结束
msg = f"Got {len(all_items)} items"
await final_answer(msg)
"""
    
    output = await execute(code, ns, state=state)
    
    # 验证数据是否提交成功 (说明 append_items 成功)
    assert len(state.items) == 2
    # 验证 final_answer 是否被触发 (说明执行到了最后一行且变量持久化成功)
    assert state.final_answer_requested == "Got 2 items"
    
    # 验证检查点文件是否生成 (说明 save_checkpoint 成功)
    checkpoint_file = run_dir / "test.json"
    assert checkpoint_file.exists()
    
    # 验证 namespace 变量持久化
    assert ns["res"] == "Visited https://example.com"
    assert ns["msg"] == "Got 2 items"

@pytest.mark.asyncio
async def test_async_exception_handling():
    """测试异步工具报错时的捕获。"""
    state = CodeAgentState()
    registry = ToolRegistry()
    
    async def broken_tool():
        raise ValueError("Tool Failed")
    
    registry.register(broken_tool, name="break_it")
    ns = build_namespace(state, registry.as_namespace_dict(), "tmp")
    
    code = "await break_it()"
    output = await execute(code, ns)
    
    assert "[执行错误]" in output
    assert "ValueError: Tool Failed" in output
