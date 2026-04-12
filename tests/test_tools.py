from __future__ import annotations

import json
import logging
import os
import tempfile

import pytest

from hawker_agent.agent.namespace import build_namespace
from hawker_agent.agent.prompts import build_system_prompt
from hawker_agent.models.state import CodeAgentState
from hawker_agent.tools.data_tools import (
    clean_items,
    ensure,
    get_type_signature,
    normalize_items,
    parse_http_response,
    save_file,
    summarize_json,
)
from hawker_agent.tools.registry import ToolRegistry
from hawker_agent.observability import clear_log_context, configure_logging


# ─── ToolRegistry ───────────────────────────────────────────────


class TestToolRegistry:
    def test_register_and_description(self) -> None:
        registry = ToolRegistry()

        def my_tool(url: str) -> str:
            """导航到指定 URL。"""
            return url

        registry.register(my_tool)
        desc = registry.build_description()
        assert "my_tool" in desc
        assert "导航到指定 URL" in desc
        assert "url: str" in desc

    def test_register_with_custom_name(self) -> None:
        registry = ToolRegistry()
        registry.register(lambda: None, name="custom_name")
        assert "custom_name" in registry

    def test_as_namespace_dict(self) -> None:
        registry = ToolRegistry()

        def tool_a() -> str:
            """Tool A."""
            return "a"

        registry.register(tool_a)
        ns = registry.as_namespace_dict()
        assert ns["tool_a"] is tool_a

    def test_len_and_contains(self) -> None:
        registry = ToolRegistry()
        assert len(registry) == 0
        registry.register(lambda: None, name="test")
        assert len(registry) == 1
        assert "test" in registry
        assert "other" not in registry


# ─── data_tools ─────────────────────────────────────────────────


class TestParseHttpResponse:
    def test_success(self) -> None:
        status, body = parse_http_response("[200]\n{\"key\": \"value\"}")
        assert status == 200
        assert body == '{"key": "value"}'

    def test_error_prefix(self) -> None:
        with pytest.raises(RuntimeError, match="错误"):
            parse_http_response("[错误] Connection refused")

    def test_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="无法解析"):
            parse_http_response("not a valid response")


class TestCleanItems:
    def test_filters_non_dict(self) -> None:
        result = clean_items([{"a": 1}, "string", 42, {"b": 2}])
        assert result == [{"a": 1}, {"b": 2}]

    def test_filters_truncated(self) -> None:
        result = clean_items([{"a": 1}, {"_truncated": "yes"}, {"b": 2}])
        assert result == [{"a": 1}, {"b": 2}]

    def test_type_error(self) -> None:
        with pytest.raises(TypeError):
            clean_items("not a list")  # type: ignore[arg-type]


class TestEnsure:
    def test_passes(self) -> None:
        ensure(True, "should not raise")

    def test_fails(self) -> None:
        with pytest.raises(RuntimeError, match="条件不满足"):
            ensure(False, "条件不满足")


class TestNormalizeItems:
    def test_list_input(self) -> None:
        result = normalize_items([{"a": 1}])
        assert result == [{"a": 1}]

    def test_dict_input(self) -> None:
        result = normalize_items({"a": 1})
        assert result == [{"a": 1}]

    def test_json_string_input(self) -> None:
        result = normalize_items('[{"a": 1}]')
        assert result == [{"a": 1}]

    def test_invalid_type(self) -> None:
        with pytest.raises(TypeError):
            normalize_items(42)


class TestGetTypeSignature:
    def test_basic(self) -> None:
        result = get_type_signature({"name": "test", "count": 42})
        assert "name: str" in result
        assert "count: int" in result

    def test_nested(self) -> None:
        result = get_type_signature({"items": [1, 2, 3]})
        assert "items: list[int]" in result


class TestSummarizeJson:
    def test_list(self) -> None:
        result = summarize_json([{"id": 1}, {"id": 2}])
        assert "[http_json] 2 条" in result

    def test_empty_list(self) -> None:
        result = summarize_json([])
        assert "空列表" in result

    def test_dict(self) -> None:
        result = summarize_json({"key": "value"})
        assert "[http_json] dict" in result


class TestSaveFile:
    def test_save_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_file('[{"id": 1}]', "test.json", tmpdir)
            assert "[OK]" in result
            path = os.path.join(tmpdir, "test.json")
            with open(path) as f:
                data = json.load(f)
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
        result = build_system_prompt(tool_desc="- nav(url) -> str: 导航")
        assert "- nav(url) -> str: 导航" in result
        assert "爬虫智能体" in result

    def test_renders_instructions(self) -> None:
        result = build_system_prompt(
            tool_desc="tools here",
            instructions="自定义指令",
        )
        assert "自定义指令" in result

    def test_default_no_instructions(self) -> None:
        result = build_system_prompt(tool_desc="tools")
        assert "执行策略" in result


# ─── namespace ──────────────────────────────────────────────────


class TestBuildNamespace:
    def teardown_method(self) -> None:
        clear_log_context()

    def test_contains_tools(self) -> None:
        state = CodeAgentState()
        tools = {"my_tool": lambda: "result"}
        ns = build_namespace(state, tools, "/tmp/test")
        assert ns["my_tool"]() == "result"

    def test_contains_helpers(self) -> None:
        state = CodeAgentState()
        ns = build_namespace(state, {}, "/tmp/test")
        assert callable(ns["append_items"])
        assert callable(ns["save_checkpoint"])
        assert callable(ns["final_answer"])
        assert callable(ns["clean_items"])
        assert callable(ns["ensure"])

    def test_contains_stdlib(self) -> None:
        state = CodeAgentState()
        ns = build_namespace(state, {}, "/tmp/test")
        import json as json_mod

        assert ns["json"] is json_mod
        assert ns["re"] is not None
        assert ns["asyncio"] is not None
        assert ns["time"] is not None

    @pytest.mark.asyncio
    async def test_append_items_updates_state(self) -> None:
        state = CodeAgentState()
        ns = build_namespace(state, {}, "/tmp/test")
        await ns["append_items"]([{"url": "a"}, {"url": "b"}])
        assert len(state.items) == 2
        assert state.activity_marker == 1

    @pytest.mark.asyncio
    async def test_final_answer_sets_state(self) -> None:
        state = CodeAgentState()
        ns = build_namespace(state, {}, "/tmp/test")
        await ns["final_answer"]("任务完成")
        assert state.final_answer_requested == "任务完成"

    @pytest.mark.asyncio
    async def test_save_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = CodeAgentState()
            state.items.append([{"id": 1}])
            ns = build_namespace(state, {}, tmpdir)
            result = await ns["save_checkpoint"]()
            assert "[OK]" in result
            assert state.checkpoint_files

    @pytest.mark.asyncio
    async def test_tools_bind_log_context(self, caplog: pytest.LogCaptureFixture) -> None:
        configure_logging(force=True)
        state = CodeAgentState(trace_id="trace-tool", run_id="run-tool")

        async def my_tool() -> str:
            logging.getLogger("hawker_agent.test.tool").info("tool-called")
            return "ok"

        ns = build_namespace(state, {"my_tool": my_tool}, "/tmp/test")

        with caplog.at_level(logging.INFO, logger="hawker_agent.test.tool"):
            result = await ns["my_tool"]()

        assert result == "ok"
        record = caplog.records[-1]
        assert record.trace_id == "trace-tool"
        assert record.run_id == "run-tool"
