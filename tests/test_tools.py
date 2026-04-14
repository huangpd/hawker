from __future__ import annotations

import json as json_mod
import logging
import os
import tempfile
from pathlib import Path

import pytest

from hawker_agent.agent.namespace import build_namespace, register_core_actions
from hawker_agent.agent.prompts import build_system_prompt
from hawker_agent.models.state import CodeAgentState
from hawker_agent.observability import clear_log_context, configure_logging
from hawker_agent.storage.exporter import save_result_json
from hawker_agent.tools.data_tools import (
    clean_items,
    ensure,
    normalize_items,
    parse_http_response,
    save_file,
    summarize_json,
    register_data_tools,
)
from hawker_agent.tools.registry import ToolRegistry


# ─── data_tools ────────────────────────────────────────────────


class TestDataTools:
    def test_get_type_signature(self) -> None:
        from hawker_agent.tools.data_tools import get_type_signature

        d = {"a": 1, "b": "str", "c": [1, 2], "d": {"x": 1}}
        sig = get_type_signature(d)
        assert "a: int" in sig
        assert "b: str" in sig
        assert "c: list[int]" in sig
        assert "d: dict{x}" in sig

    def test_parse_http_response_success(self) -> None:
        raw = "[200]\nhello world"
        status, body = parse_http_response(raw)
        assert status == 200
        assert body == "hello world"

    def test_parse_http_response_error(self) -> None:
        raw = "[错误] connection failed"
        with pytest.raises(RuntimeError, match="connection failed"):
            parse_http_response(raw)

    def test_clean_items(self) -> None:
        items = [
            {"id": 1},
            {"_truncated": "too long"},
            "not a dict",
            {"id": 2},
        ]
        cleaned = clean_items(items)
        assert len(cleaned) == 2
        assert cleaned[0]["id"] == 1
        assert cleaned[1]["id"] == 2

    def test_normalize_items(self) -> None:
        assert len(normalize_items({"a": 1})) == 1
        assert len(normalize_items([{"a": 1}, {"b": 2}])) == 2
        assert len(normalize_items('[{"a": 1}]')) == 1

    def test_save_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = json_mod.dumps([{"id": 1}])
            result = save_file(data, "test.json", tmpdir)
            assert "[OK] 已保存 1 条记录" in result
            path = os.path.join(tmpdir, "test.json")
            with open(path) as f:
                data = json_mod.load(f)
                assert data == [{"id": 1}]

    def test_save_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_file("plain text", "test.txt", tmpdir)
            assert "[OK]" in result
            path = os.path.join(tmpdir, "test.txt")
            with open(path) as f:
                assert f.read() == "plain text"


# ─── prompts ────────────────────────────────────────────────────


class TestBuildSystemPrompt:
    def test_renders_tool_desc(self) -> None:
        result = build_system_prompt(
            async_capabilities="await nav()",
            sync_capabilities="clean_items()",
        )
        assert "await nav()" in result
        assert "clean_items()" in result
        assert "爬虫智能体" in result

    def test_renders_instructions(self) -> None:
        result = build_system_prompt(
            async_capabilities="caps_a",
            sync_capabilities="caps_s",
            instructions="自定义指令",
        )
        assert "自定义指令" in result

    def test_default_no_instructions(self) -> None:
        result = build_system_prompt(
            async_capabilities="caps_a",
            sync_capabilities="caps_s",
        )
        assert "工具列表" in result


# ─── namespace ──────────────────────────────────────────────────


class TestBuildNamespace:
    def teardown_method(self) -> None:
        clear_log_context()

    def _get_ns(self, state, run_dir, tools=None):
        reg = ToolRegistry()
        register_core_actions(reg, state, run_dir)
        register_data_tools(reg)
        if tools:
            for name, fn in tools.items():
                reg.register(fn, name=name)
        return build_namespace(state, reg.as_namespace_dict(), run_dir)

    def test_contains_tools(self) -> None:
        state = CodeAgentState()
        ns = self._get_ns(state, "/tmp/test", tools={"my_tool": lambda: "result"})
        assert ns["my_tool"]() == "result"

    def test_contains_helpers(self) -> None:
        state = CodeAgentState()
        ns = self._get_ns(state, "/tmp/test")
        # Core actions
        assert callable(ns["append_items"])
        assert callable(ns["save_checkpoint"])
        assert callable(ns["observe"])
        assert callable(ns["final_answer"])
        # Data tools
        assert callable(ns["clean_items"])
        assert callable(ns["ensure"])
        assert callable(ns["summarize_json"])

    def test_contains_stdlib(self) -> None:
        state = CodeAgentState()
        ns = self._get_ns(state, "/tmp/test")
        import json as json_mod

        assert ns["json"] is json_mod
        assert ns["re"] is not None
        assert ns["asyncio"] is not None
        assert ns["time"] is not None

    @pytest.mark.asyncio
    async def test_append_items_updates_state(self) -> None:
        state = CodeAgentState()
        ns = self._get_ns(state, "/tmp/test")
        await ns["append_items"]([{"url": "a"}, {"url": "b"}])
        assert len(state.items) == 2
        assert state.activity_marker == 1

    @pytest.mark.asyncio
    async def test_final_answer_sets_state(self) -> None:
        state = CodeAgentState()
        ns = self._get_ns(state, "/tmp/test")
        await ns["final_answer"]("任务完成")
        assert state.final_answer_requested == "任务完成"

    @pytest.mark.asyncio
    async def test_save_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = CodeAgentState()
            state.items.append([{"id": 1}])
            ns = self._get_ns(state, tmpdir)
            result = await ns["save_checkpoint"]()
            assert "[OK]" in result
            assert state.checkpoint_files

    @pytest.mark.asyncio
    async def test_tools_bind_log_context(self, caplog: pytest.LogCaptureFixture) -> None:
        configure_logging(force=True)
        state = CodeAgentState(trace_id="trace-tool", run_id="run-tool")

        async def my_tool() -> str:
            import logging
            logging.getLogger("hawker_agent.test.tool").info("tool-called")
            return "ok"

        ns = self._get_ns(state, "/tmp/test", tools={"my_tool": my_tool})

        with caplog.at_level(logging.INFO, logger="hawker_agent.test.tool"):
            result = await ns["my_tool"]()

        assert result == "ok"
        record = caplog.records[-1]
        assert record.trace_id == "trace-tool"
        assert record.run_id == "run-tool"


# ─── Data Tools Registration ───────────────────────────────────

def test_register_data_tools() -> None:
    reg = ToolRegistry()
    register_data_tools(reg)
    assert "ensure" in reg
    assert "parse_http_response" in reg
    
    # Check sync/async split
    sync_caps = reg.build_capabilities_list("sync")
    async_caps = reg.build_capabilities_list("async")
    assert "ensure" in sync_caps
    assert "await" not in sync_caps
    assert "asyncio.sleep" in async_caps


def test_save_result_json_does_not_delete_result_when_checkpoint_has_same_name(tmp_path: Path) -> None:
    result_path = save_result_json(
        tmp_path,
        [{"id": 1}],
        "done",
        checkpoint_files={"result.json"},
    )

    assert result_path.exists()
    data = json_mod.loads(result_path.read_text(encoding="utf-8"))
    assert data["items_count"] == 1
