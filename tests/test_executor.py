from __future__ import annotations

import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, patch

from hawker_agent.agent.executor import (
    _clean_traceback,
    execute,
)
from hawker_agent.agent.namespace import HawkerNamespace
from hawker_agent.models.state import CodeAgentState
from hawker_agent.observability import clear_log_context, configure_logging, emit_observation


# ─── helpers ────────────────────────────────────────────────────


class TestCleanTraceback:
    def test_removes_internal_frames(self) -> None:
        tb = (
            'Traceback (most recent call last):\n'
            '  File "executor.py", line 100, in execute\n'
            '    result = exec(compiled, view)\n'
            '  File "<hawker-cell>", line 1, in <module>\n'
            '    raise ValueError("boom")\n'
            'ValueError: boom'
        )
        cleaned = _clean_traceback(tb)
        assert "executor.py" not in cleaned
        assert "<hawker-cell>" in cleaned
        assert "ValueError: boom" in cleaned

    def test_preserves_cell_frames(self) -> None:
        tb = (
            'Traceback (most recent call last):\n'
            '  File "<hawker-cell>", line 3, in foo\n'
            '    x = 1/0\n'
            'ZeroDivisionError: division by zero'
        )
        cleaned = _clean_traceback(tb)
        assert "<hawker-cell>" in cleaned
        assert "ZeroDivisionError" in cleaned


# ─── execute ────────────────────────────────────────────────────


class TestExecute:
    def teardown_method(self) -> None:
        clear_log_context()

    def _make_ns(self) -> HawkerNamespace:
        return HawkerNamespace(system_dict={"asyncio": asyncio, "observe": emit_observation}, run_dir="tmp")

    @pytest.mark.asyncio
    async def test_print_does_not_become_observation(self) -> None:
        ns = self._make_ns()
        result = await execute("print('hello')", ns)
        assert result == "[无输出]"

    @pytest.mark.asyncio
    async def test_explicit_observation(self) -> None:
        ns = self._make_ns()
        result = await execute("observe('hello')", ns)
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_variable_persistence_protocol(self) -> None:
        ns = self._make_ns()
        # 符合协议的变量会被提升到 session
        await execute("products_list = [1, 2, 3]", ns)
        assert "products_list" in ns.session
        assert ns.session["products_list"] == [1, 2, 3]
        
        # 不符合协议的变量 (过短) 会被丢弃
        await execute("i = 42", ns)
        assert "i" not in ns.session
        
        # 验证持久化变量在下一步可用
        result = await execute("observe(str(len(products_list)))", ns)
        assert result == "3"

    @pytest.mark.asyncio
    async def test_transactional_rollback(self) -> None:
        ns = self._make_ns()
        await execute("my_data = {'count': 10}", ns)
        
        # 尝试修改变量但发生错误
        code = "my_data['count'] = 20\nraise ValueError('fail')"
        result = await execute(code, ns)
        
        assert "[执行错误]" in result
        # 验证回滚：my_data 的修改被撤销了
        assert ns.session["my_data"]["count"] == 10
        # 验证临时变量被清理
        assert ns.cell_local == {}

    @pytest.mark.asyncio
    async def test_rejects_execution_when_session_snapshot_is_not_safe(self) -> None:
        class NonCopyable:
            def __deepcopy__(self, memo):
                raise TypeError("no deepcopy")

        ns = self._make_ns()
        ns.session["bad_state"] = NonCopyable()

        result = await execute("new_value = 1\nobserve('should not run')", ns)

        assert "[执行错误]" in result
        assert "session 快照失败" in result
        assert "bad_state" in result
        assert "new_value" not in ns.session

    @pytest.mark.asyncio
    async def test_no_output(self) -> None:
        ns = self._make_ns()
        result = await execute("x_val = 1", ns)
        assert result == "[无输出]"

    @pytest.mark.asyncio
    async def test_syntax_error(self) -> None:
        ns = self._make_ns()
        result = await execute("if True print('bad')", ns)
        assert "[执行错误]" in result
        assert "SyntaxError" in result

    @pytest.mark.asyncio
    async def test_name_error_with_hint(self) -> None:
        ns = self._make_ns()
        result = await execute("print(undefined_var)", ns)
        assert "[执行错误]" in result
        assert "NameError" in result
        assert "提示" in result

    @pytest.mark.asyncio
    async def test_async_code_native(self) -> None:
        ns = self._make_ns()
        # 原生支持顶层 await
        result = await execute("await asyncio.sleep(0.01)\nobserve('async success')", ns)
        assert "async success" in result

    @pytest.mark.asyncio
    async def test_async_variable_persistence(self) -> None:
        ns = self._make_ns()
        await execute("total_count = 10", ns)
        # 异步环境下的变量修改与持久化
        result = await execute(
            "await asyncio.sleep(0)\ntotal_count += 5\nobserve(str(total_count))", ns
        )
        assert "15" in result
        assert ns.session["total_count"] == 15

    @pytest.mark.asyncio
    async def test_layered_isolation(self) -> None:
        ns = self._make_ns()
        # 尝试覆盖系统工具 (asyncio)
        await execute("asyncio = 'hacked'", ns)
        # 检查是否由于 commit 逻辑被拦截
        assert ns.session.get("asyncio") != "hacked"
        assert ns.system["asyncio"] == asyncio

    @pytest.mark.asyncio
    async def test_output_truncation(self) -> None:
        ns = self._make_ns()
        code = "observe('x' * 10000)"
        result = await execute(code, ns)
        assert len(result) < 10000
        assert "截断" in result

    @pytest.mark.asyncio
    async def test_execute_binds_log_context(self, caplog: pytest.LogCaptureFixture) -> None:
        configure_logging(force=True)
        state = CodeAgentState(trace_id="trace-exec", run_id="run-exec")
        ns = self._make_ns()

        with caplog.at_level(logging.INFO, logger="hawker_agent.agent.executor"):
            result = await execute("observe('hello')", ns, state=state, step=7)

        assert result == "hello"
        assert any(record.trace_id == "trace-exec" for record in caplog.records)
        assert any(record.run_id == "run-exec" for record in caplog.records)
        assert any(record.step == "7" for record in caplog.records)

    @pytest.mark.asyncio
    async def test_healing_retry_can_fix_current_cell(self) -> None:
        state = CodeAgentState(trace_id="trace-heal", run_id="run-heal")
        ns = self._make_ns()

        with patch(
            "hawker_agent.agent.executor.try_heal_code",
            new=AsyncMock(return_value="safe_items = []\nobserve(str(len(safe_items)))"),
        ) as mock_heal:
            result = await execute("observe(str(len(safe_items)))", ns, state=state, step=3)

        assert result == "0"
        assert mock_heal.await_count == 1

    @pytest.mark.asyncio
    async def test_healing_falls_back_to_original_error_when_no_fix(self) -> None:
        state = CodeAgentState(trace_id="trace-heal-fail", run_id="run-heal-fail")
        ns = self._make_ns()

        with patch(
            "hawker_agent.agent.executor.try_heal_code",
            new=AsyncMock(return_value=None),
        ) as mock_heal:
            result = await execute("observe(str(len(missing_items)))", ns, state=state, step=4)

        assert "[执行错误]" in result
        assert "NameError" in result
        assert mock_heal.await_count == 1
