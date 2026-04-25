import pytest
from hawker_agent.agent.executor import execute
from hawker_agent.agent.namespace import HawkerNamespace, build_system_dict
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
        return {"ok": True, "url": url, "filename": "file.pdf", "path": "/tmp/file.pdf", "size": 3}
        
    registry.register(mock_nav, name="nav")
    registry.register(mock_download, name="browser_download")
    
    # 构建分层命名空间
    sys_dict = build_system_dict(state, registry.as_namespace_dict(), str(run_dir))
    ns = HawkerNamespace(sys_dict, str(run_dir))
    
    # 场景：Agent 执行一段全异步代码
    code = """
# 1. 异步导航
res_visited = await nav("https://example.com")
# 2. 异步提交数据
my_items = [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}]
i = 0 # 临时循环索引
await append_items(my_items)
# 3. 异步保存检查点
await save_checkpoint("test.json")
# 4. 异步下载
await browser_download("https://file.pdf")
# 5. 定义变量并结束
final_msg = f"Got {len(all_items)} items"
await final_answer(final_msg)
"""
    
    await execute(code, ns, state=state)
    
    # 验证业务数据与下载证据是否提交成功
    assert len(state.items) == 3
    assert state.items.to_list()[-1]["download"]["status"] == "success"
    # 验证 final_answer 是否被触发
    assert state.final_answer_requested == "Got 3 items"
    
    # 验证检查点文件是否生成
    checkpoint_file = run_dir / "test.json"
    assert checkpoint_file.exists()
    
    # 验证 namespace 变量持久化 (符合协议的变量提升到 session)
    assert ns.session["res_visited"] == "Visited https://example.com"
    assert ns.session["final_msg"] == "Got 3 items"
    assert ns.session["my_items"] == [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}]
    
    # 不符合协议的临时变量 (i) 应该在 session 中找不到
    assert "i" not in ns.session

@pytest.mark.asyncio
async def test_async_exception_handling():
    """测试异步工具报错时的捕获。"""
    state = CodeAgentState()
    registry = ToolRegistry()
    
    async def broken_tool():
        raise ValueError("Tool Failed")
    
    registry.register(broken_tool, name="break_it")
    sys_dict = build_system_dict(state, registry.as_namespace_dict(), "tmp")
    ns = HawkerNamespace(sys_dict, "tmp")
    
    code = "await break_it()"
    output = await execute(code, ns)
    
    assert "[执行错误]" in output
    assert "ValueError: Tool Failed" in output
