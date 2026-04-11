from __future__ import annotations

import asyncio
import logging

import pytest

from hawker_agent.agent.executor import (
    _clean_traceback,
    _collect_assigned_names,
    _has_async_constructs,
    execute,
)
from hawker_agent.models.state import CodeAgentState
from hawker_agent.observability import clear_log_context, configure_logging


# ─── helpers ────────────────────────────────────────────────────


class TestCleanTraceback:
    def test_removes_exec_frames(self) -> None:
        tb = (
            'Traceback (most recent call last):\n'
            '  File "<code>", line 1, in <module>\n'
            '    exec(compile(wrapped, "<code>", "exec"), namespace)\n'
            '    some detail\n'
            '  File "<code>", line 3, in foo\n'
            'NameError: name \'x\' is not defined'
        )
        cleaned = _clean_traceback(tb)
        assert "exec(compile(" not in cleaned
        assert "NameError" in cleaned

    def test_preserves_normal_frames(self) -> None:
        tb = (
            'Traceback (most recent call last):\n'
            '  File "my_script.py", line 10, in func\n'
            '    raise ValueError("bad")\n'
            'ValueError: bad'
        )
        cleaned = _clean_traceback(tb)
        assert cleaned == tb


class TestCollectAssignedNames:
    def test_simple_assignment(self) -> None:
        import ast

        tree = ast.parse("x = 1\ny = 2")
        names = _collect_assigned_names(tree)
        assert names == {"x", "y"}

    def test_augmented_assignment(self) -> None:
        import ast

        tree = ast.parse("x += 1")
        names = _collect_assigned_names(tree)
        assert "x" in names

    def test_global_declaration(self) -> None:
        import ast

        tree = ast.parse("global x, y")
        names = _collect_assigned_names(tree)
        assert names == {"x", "y"}


class TestHasAsyncConstructs:
    def test_sync_code(self) -> None:
        import ast

        tree = ast.parse("x = 1")
        assert not _has_async_constructs(tree)

    def test_await(self) -> None:
        import ast

        tree = ast.parse("async def f():\n    await something()", mode="exec")
        assert _has_async_constructs(tree)


# ─── execute ────────────────────────────────────────────────────


class TestExecute:
    def teardown_method(self) -> None:
        clear_log_context()

    @pytest.mark.asyncio
    async def test_simple_print(self) -> None:
        ns: dict = {}
        result = await execute("print('hello')", ns)
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_variable_persistence(self) -> None:
        ns: dict = {}
        await execute("x = 42", ns)
        assert ns["x"] == 42
        result = await execute("print(x + 1)", ns)
        assert result == "43"

    @pytest.mark.asyncio
    async def test_no_output(self) -> None:
        ns: dict = {}
        result = await execute("x = 1", ns)
        assert result == "[无输出]"

    @pytest.mark.asyncio
    async def test_syntax_error(self) -> None:
        ns: dict = {}
        result = await execute("if True print('bad')", ns)
        assert "[执行错误]" in result
        assert "SyntaxError" in result

    @pytest.mark.asyncio
    async def test_name_error_with_hint(self) -> None:
        ns: dict = {}
        result = await execute("print(undefined_var)", ns)
        assert "[执行错误]" in result
        assert "NameError" in result
        assert "提示" in result

    @pytest.mark.asyncio
    async def test_runtime_error(self) -> None:
        ns: dict = {}
        result = await execute("raise ValueError('test error')", ns)
        assert "[执行错误]" in result
        assert "ValueError" in result

    @pytest.mark.asyncio
    async def test_async_code(self) -> None:
        ns: dict = {"asyncio": asyncio}
        result = await execute("result = await asyncio.sleep(0); print('async done')", ns)
        assert "async done" in result

    @pytest.mark.asyncio
    async def test_async_variable_persistence(self) -> None:
        ns: dict = {"asyncio": asyncio}
        await execute("x = 10", ns)
        result = await execute(
            "await asyncio.sleep(0)\nx = x + 5\nprint(x)", ns
        )
        assert "15" in result
        assert ns["x"] == 15

    @pytest.mark.asyncio
    async def test_multiline_code(self) -> None:
        ns: dict = {}
        code = "for i in range(3):\n    print(i)"
        result = await execute(code, ns)
        assert "0" in result
        assert "1" in result
        assert "2" in result

    @pytest.mark.asyncio
    async def test_output_truncation(self) -> None:
        ns: dict = {}
        code = "print('x' * 10000)"
        result = await execute(code, ns)
        assert len(result) < 10000
        assert "截断" in result

    @pytest.mark.asyncio
    async def test_exception_preserves_partial_output(self) -> None:
        ns: dict = {}
        code = "print('before error')\nraise RuntimeError('boom')"
        result = await execute(code, ns)
        assert "before error" in result
        assert "[执行错误]" in result
        assert "RuntimeError" in result

    @pytest.mark.asyncio
    async def test_execute_binds_log_context(self, caplog: pytest.LogCaptureFixture) -> None:
        configure_logging(force=True)
        state = CodeAgentState(trace_id="trace-exec", run_id="run-exec")

        with caplog.at_level(logging.INFO, logger="hawker_agent.agent.executor"):
            result = await execute("print('hello')", {}, state=state, step=7)

        assert result == "hello"
        assert any(record.trace_id == "trace-exec" for record in caplog.records)
        assert any(record.run_id == "run-exec" for record in caplog.records)
        assert any(record.step == "7" for record in caplog.records)
