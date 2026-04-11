import asyncio
import pytest
from hawker_agent.agent.executor import execute

@pytest.mark.asyncio
async def test_execute_async_variable_persistence():
    """测试异步执行中的变量是否能正确同步回 namespace。"""
    ns = {"asyncio": asyncio}
    
    # 模拟一个异步操作
    async def mock_async_get_data():
        await asyncio.sleep(0.01)
        return {"id": 123, "name": "test"}
    
    ns["get_data"] = mock_async_get_data
    
    # 执行代码：定义一个新变量 data，并 await 异步函数
    code = """
data = await get_data()
x = 100
"""
    output = await execute(code, ns)
    
    assert "success=true" in output.lower() or output == "[无输出]"
    assert "data" in ns
    assert ns["data"] == {"id": 123, "name": "test"}
    assert ns["x"] == 100

@pytest.mark.asyncio
async def test_execute_async_global_modification():
    """测试异步代码是否能修改 namespace 中已有的变量。"""
    ns = {"counter": 0, "asyncio": asyncio}
    
    code = """
await asyncio.sleep(0.01)
counter += 1
"""
    # 第一次执行
    await execute(code, ns)
    assert ns["counter"] == 1
    
    # 第二次执行
    await execute(code, ns)
    assert ns["counter"] == 2

@pytest.mark.asyncio
async def test_execute_async_and_sync_mix():
    """测试异步和同步代码混合时，变量的持久化。"""
    ns = {"asyncio": asyncio}
    
    # 第一步：异步定义 a
    await execute("import time; a = 1; await asyncio.sleep(0.01)", ns)
    # 第二步：同步定义 b，引用 a
    await execute("b = a + 1", ns)
    # 第三步：异步修改 a 和 b
    await execute("await asyncio.sleep(0.01); a += 10; b *= 10", ns)
    
    assert ns["a"] == 11
    assert ns["b"] == 20

@pytest.mark.asyncio
async def test_execute_async_syntax_error():
    """测试异步语法错误的处理。"""
    ns = {"asyncio": asyncio}
    # 缺少冒号的语法错误
    code = "async with asyncio.sleep(1)\n    pass"
    output = await execute(code, ns)
    
    assert "[执行错误]" in output
    assert "SyntaxError" in output

@pytest.mark.asyncio
async def test_execute_async_runtime_error_cleanup():
    """测试异步运行时错误是否能正确清理 Traceback。"""
    ns = {"asyncio": asyncio}
    
    code = """
async def fail():
    raise ValueError("Async Crash")

await fail()
"""
    output = await execute(code, ns)
    
    assert "[执行错误]" in output
    assert "ValueError: Async Crash" in output
    # 确保没有暴露我们内部的 __code_exec__ 包装行
    assert "__code_exec__" not in output
    assert "exec(compile(" not in output
