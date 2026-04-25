from __future__ import annotations

import asyncio
import json as json_mod
import logging
import os
import tempfile
from pathlib import Path

import pytest

from hawker_agent.agent.namespace import build_namespace, build_system_dict, register_core_actions
from hawker_agent.agent.artifact import normalize_final_artifact, recover_items_from_artifact
from hawker_agent.agent.prompts import build_system_prompt
from hawker_agent.models.history import CodeAgentHistoryList
from hawker_agent.models.state import CodeAgentState
from hawker_agent.observability import clear_log_context, configure_logging
from hawker_agent.storage.exporter import save_llm_io_json, save_result_json
from hawker_agent.tools.browser_tools import register_browser_tools
from hawker_agent.tools.data_tools import (
    analyze_json_structure,
    clean_items,
    normalize_items,
    parse_http_response,
    save_file,
    register_data_tools,
)
from hawker_agent.tools.http_tools import fetch, http_json, http_request
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

    def test_analyze_json_structure_reports_generic_paths(self) -> None:
        data = {
            "organic": [
                {
                    "position": 1,
                    "title": "Paper",
                    "url": "https://example.com/paper",
                    "metadata": {"source": "search"},
                }
            ],
            "totalResults": 20,
        }

        structure = analyze_json_structure(data)

        assert structure["root_type"] == "dict"
        assert structure["root_keys"] == ["organic", "totalResults"]
        assert {
            "path": "$.organic",
            "type": "list",
            "length": 1,
            "sample_item_types": ["dict"],
            "sample_item_keys": ["metadata", "position", "title", "url"],
            "truncated_item_keys": False,
        } in structure["paths"]
        assert any(
            path["path"] == "$.organic[].metadata" and path["keys"] == ["source"]
            for path in structure["paths"]
        )

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

    def test_final_answer_prompt_does_not_teach_result_json_boilerplate(self) -> None:
        instructions = (Path(__file__).resolve().parents[1] / "hawker_agent" / "templates" / "default_instructions.txt").read_text(
            encoding="utf-8"
        )
        result = build_system_prompt(
            async_capabilities="caps_a",
            sync_capabilities="caps_s",
            instructions=instructions,
        )
        assert "不要提 `result.json`" in result
        assert "正式结果会在任务结束后由系统自动写入 `result.json`" not in result


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
        assert callable(ns["sys_clean_items"])
        assert callable(ns["ensure"])
        assert callable(ns["sys_summarize_json"])
        assert callable(ns["sys_analyze_json_structure"])

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
    async def test_final_answer_accepts_structured_artifact(self) -> None:
        state = CodeAgentState()
        ns = self._get_ns(state, "/tmp/test")
        await ns["final_answer"](
            {
                "type": "text",
                "content": "# 总结\n已完成",
                "items": [{"url": "https://example.com"}],
            }
        )
        assert state.final_artifact_requested is not None
        assert state.final_artifact_requested["type"] == "text"
        assert state.final_artifact_requested["items"] == [{"url": "https://example.com"}]
        assert state.final_answer_requested == "# 总结\n已完成"

    @pytest.mark.asyncio
    async def test_final_answer_json_artifact_uses_summary_text_not_json_dump(self) -> None:
        state = CodeAgentState()
        ns = self._get_ns(state, "/tmp/test")
        await ns["final_answer"](
            {
                "type": "json",
                "content": {"summary": "中文总结：模型能力继续增强。"},
                "items": [{"url": "https://example.com"}],
            }
        )
        assert state.final_artifact_requested is not None
        assert state.final_answer_requested == "中文总结：模型能力继续增强。"

    @pytest.mark.asyncio
    async def test_browser_js_accepts_args_keyword(self) -> None:
        state = CodeAgentState()

        async def fake_js(_session, code: str):
            return code

        reg = ToolRegistry()
        register_core_actions(reg, state, "/tmp/test")
        history = CodeAgentHistoryList()
        session = type("Session", (), {})()
        from hawker_agent.tools import browser_tools as browser_tools_module

        original = browser_tools_module.actions.js
        browser_tools_module.actions.js = fake_js
        try:
            register_browser_tools(reg, session, history, state)
            ns = build_namespace(state, reg.as_namespace_dict(), "/tmp/test")
            result = await ns["js"]("(selector) => selector", args=[".repo"])
        finally:
            browser_tools_module.actions.js = original

        assert "__hawker_args" in result
        assert '".repo"' in result

    @pytest.mark.asyncio
    async def test_browser_js_accepts_positional_function_args(self) -> None:
        state = CodeAgentState()

        async def fake_js(_session, code: str):
            return code

        reg = ToolRegistry()
        register_core_actions(reg, state, "/tmp/test")
        history = CodeAgentHistoryList()
        session = type("Session", (), {})()
        from hawker_agent.tools import browser_tools as browser_tools_module

        original = browser_tools_module.actions.js
        browser_tools_module.actions.js = fake_js
        try:
            register_browser_tools(reg, session, history, state)
            ns = build_namespace(state, reg.as_namespace_dict(), "/tmp/test")
            result = await ns["js"]("(a, b) => [a, b]", "x", "y")
        finally:
            browser_tools_module.actions.js = original

        assert "__hawker_args" in result
        assert '"x"' in result
        assert '"y"' in result

    @pytest.mark.asyncio
    async def test_browser_js_auto_invokes_zero_arg_function_expression(self) -> None:
        state = CodeAgentState()

        async def fake_js(_session, code: str):
            return code

        reg = ToolRegistry()
        register_core_actions(reg, state, "/tmp/test")
        history = CodeAgentHistoryList()
        session = type("Session", (), {})()
        from hawker_agent.tools import browser_tools as browser_tools_module

        original = browser_tools_module.actions.js
        browser_tools_module.actions.js = fake_js
        try:
            register_browser_tools(reg, session, history, state)
            ns = build_namespace(state, reg.as_namespace_dict(), "/tmp/test")
            result = await ns["js"]("() => document.title")
        finally:
            browser_tools_module.actions.js = original

        assert result == "(() => document.title)()"

    @pytest.mark.asyncio
    async def test_browser_js_keeps_plain_expression_unchanged(self) -> None:
        state = CodeAgentState()

        async def fake_js(_session, code: str):
            return code

        reg = ToolRegistry()
        register_core_actions(reg, state, "/tmp/test")
        history = CodeAgentHistoryList()
        session = type("Session", (), {})()
        from hawker_agent.tools import browser_tools as browser_tools_module

        original = browser_tools_module.actions.js
        browser_tools_module.actions.js = fake_js
        try:
            register_browser_tools(reg, session, history, state)
            ns = build_namespace(state, reg.as_namespace_dict(), "/tmp/test")
            result = await ns["js"]("document.title")
        finally:
            browser_tools_module.actions.js = original

        assert result == "document.title"

    @pytest.mark.asyncio
    async def test_browser_js_does_not_rewrap_iife(self) -> None:
        state = CodeAgentState()

        async def fake_js(_session, code: str):
            return code

        reg = ToolRegistry()
        register_core_actions(reg, state, "/tmp/test")
        history = CodeAgentHistoryList()
        session = type("Session", (), {})()
        from hawker_agent.tools import browser_tools as browser_tools_module

        original = browser_tools_module.actions.js
        browser_tools_module.actions.js = fake_js
        try:
            register_browser_tools(reg, session, history, state)
            ns = build_namespace(state, reg.as_namespace_dict(), "/tmp/test")
            result = await ns["js"]("(() => document.title)()")
        finally:
            browser_tools_module.actions.js = original

        assert result == "(() => document.title)()"

    @pytest.mark.asyncio
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
    assert "analyze_json_structure" in reg
    sync_caps = reg.build_capabilities_list("sync")
    async_caps = reg.build_capabilities_list("async")
    assert "ensure" not in sync_caps
    assert "parse_http_response" not in sync_caps
    assert "analyze_json_structure" in sync_caps
    assert "await" not in sync_caps
    assert "asyncio.sleep" in async_caps


@pytest.mark.asyncio
async def test_search_web_uses_searlo_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from hawker_agent.tools import http_tools as http_tools_module
    from hawker_agent.tools.http_tools import search_web

    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "success": True,
                "searchInformation": {"query": "machine learning"},
                "items": [
                    {
                        "rank": 1,
                        "title": "Intro",
                        "link": "https://example.com/ml",
                        "snippet": "Machine learning is a subset of AI...",
                    }
                ],
            }

    class DummyClient:
        async def get(self, url: str, **kwargs: object) -> DummyResponse:
            captured["url"] = url
            captured["kwargs"] = kwargs
            return DummyResponse()

    monkeypatch.setenv("SEARLO_API_KEY", "searlo-test-key")

    async def fake_get_client() -> DummyClient:
        return DummyClient()

    http_tools_module._search_web_cache.clear()
    monkeypatch.setattr("hawker_agent.tools.http_tools._get_client", fake_get_client)
    from hawker_agent.config import get_settings

    get_settings.cache_clear()
    try:
        result = await search_web("machine learning", limit=3, page=2, gl="us")
        full_result = await search_web("machine learning", limit=3, page=2, gl="us", full=True)
    finally:
        get_settings.cache_clear()

    assert result == [
        {
            "rank": 1,
            "title": "Intro",
            "link": "https://example.com/ml",
            "snippet": "Machine learning is a subset of AI...",
        }
    ]
    assert full_result["success"] is True
    assert captured["url"] == "https://api.searlo.tech/api/v1/search/web"
    kwargs = captured["kwargs"]
    assert kwargs["headers"] == {"x-api-key": "searlo-test-key"}
    assert kwargs["params"] == {"q": "machine learning", "limit": 3, "page": 2, "gl": "us"}


@pytest.mark.asyncio
async def test_search_web_returns_raw_detected_result_items_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    from hawker_agent.tools import http_tools as http_tools_module
    from hawker_agent.tools.http_tools import search_web

    calls = 0

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "searchParameters": {"q": "paper", "type": "search", "gl": "us"},
                "page": 1,
                "nextPage": 2,
                "previousPage": None,
                "totalResults": 20,
                "searchTime": 3.551,
                "organic": [
                    {
                        "position": 1,
                        "title": "Paper",
                        "url": "https://example.com/paper",
                        "description": "Abstract snippet",
                        "displayedLink": "example.com › paper",
                        "domain": "example.com",
                    }
                ],
                "source": "searlo.tech(brave)",
                "credits": 1,
            }

    class DummyClient:
        async def get(self, _url: str, **_kwargs: object) -> DummyResponse:
            nonlocal calls
            calls += 1
            return DummyResponse()

    monkeypatch.setenv("SEARLO_API_KEY", "searlo-test-key")
    http_tools_module._search_web_cache.clear()

    async def fake_get_client() -> DummyClient:
        return DummyClient()

    monkeypatch.setattr("hawker_agent.tools.http_tools._get_client", fake_get_client)
    from hawker_agent.config import get_settings

    get_settings.cache_clear()
    try:
        first = await search_web("paper", limit=5)
        second = await search_web("paper", limit=5)
        full = await search_web("paper", limit=5, full=True)
    finally:
        get_settings.cache_clear()
        http_tools_module._search_web_cache.clear()

    assert calls == 1
    assert first == second
    assert first == [
        {
            "position": 1,
            "title": "Paper",
            "url": "https://example.com/paper",
            "description": "Abstract snippet",
            "displayedLink": "example.com › paper",
            "domain": "example.com",
        }
    ]
    assert full["items"] == first
    assert full["searchInformation"] == {
        "totalResults": 20,
        "searchTime": 3.551,
        "query": "paper",
        "currentPage": 1,
        "nextPage": 2,
        "previousPage": None,
        "hasNextPage": True,
        "source": "searlo.tech(brave)",
        "credits": 1,
    }
    assert full["schema"]["root_type"] == "dict"
    assert full["schema"]["root_keys"] == [
        "searchParameters",
        "page",
        "nextPage",
        "previousPage",
        "totalResults",
        "searchTime",
        "organic",
        "source",
        "credits",
    ]
    assert full["schema"]["selected_record_path"] == "$.organic"
    assert full["schema"]["candidate_record_lists"] == [
        {
            "path": "$.organic",
            "length": 1,
            "dict_items": 1,
            "sample_item_keys": ["description", "displayedLink", "domain", "position", "title", "url"],
        }
    ]
    assert any(
        path["path"] == "$.organic"
        and path["type"] == "list"
        and path["sample_item_keys"] == ["description", "displayedLink", "domain", "position", "title", "url"]
        for path in full["schema"]["paths"]
    )


def test_hidden_tools_do_not_appear_in_prompt_capabilities() -> None:
    reg = ToolRegistry()

    def visible_tool() -> None:
        """可见工具"""

    def hidden_tool() -> None:
        """隐藏工具"""

    reg.register(visible_tool, category="同步工具")
    reg.register(hidden_tool, category="同步工具", expose_in_prompt=False)

    sync_caps = reg.build_capabilities_list("sync")
    assert "visible_tool" in sync_caps
    assert "hidden_tool" not in sync_caps


@pytest.mark.asyncio
async def test_fetch_dispatches_to_http_json() -> None:
    from hawker_agent.tools import http_tools as http_tools_module

    async def fake_http_json(*args, **kwargs):
        return {"ok": True, "mode": "json"}

    original = http_tools_module.http_json
    http_tools_module.http_json = fake_http_json
    try:
        result = await fetch("https://example.com/api", parse="json")
    finally:
        http_tools_module.http_json = original

    assert result == {"ok": True, "mode": "json"}


@pytest.mark.asyncio
async def test_fetch_json_accepts_preview_chars_alias() -> None:
    from hawker_agent.tools import http_tools as http_tools_module

    captured_kwargs: dict = {}

    async def fake_http_request(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return "[200]\n{\"ok\": true}"

    original = http_tools_module.http_request
    http_tools_module.http_request = fake_http_request
    try:
        result = await fetch("https://example.com/api", parse="json", preview_chars=321)
    finally:
        http_tools_module.http_request = original

    assert result == {"ok": True}
    assert captured_kwargs["preview_chars"] == 321


@pytest.mark.asyncio
async def test_fetch_dispatches_to_http_request() -> None:
    from hawker_agent.tools import http_tools as http_tools_module

    async def fake_http_request(*args, **kwargs):
        return "[200]\\nplain text"

    original = http_tools_module.http_request
    http_tools_module.http_request = fake_http_request
    try:
        result = await fetch("https://example.com/api", parse="text")
    finally:
        http_tools_module.http_request = original

    assert result == "[200]\\nplain text"


@pytest.mark.asyncio
async def test_fetch_body_returns_clean_response_body() -> None:
    from hawker_agent.tools import http_tools as http_tools_module

    async def fake_http_request(*args, **kwargs):
        return "[200]\n<feed><title>ok</title></feed>"

    original = http_tools_module.http_request
    http_tools_module.http_request = fake_http_request
    try:
        result = await fetch("https://example.com/feed.xml", parse="body")
    finally:
        http_tools_module.http_request = original

    assert result == "<feed><title>ok</title></feed>"


def test_save_result_json_does_not_delete_result_when_checkpoint_has_same_name(tmp_path: Path) -> None:
    result_path = save_result_json(
        tmp_path,
        [{"id": 1}],
        "done",
        checkpoint_files={"result.json"},
    )

    assert result_path.exists()
    assert result_path.parent.name == "result"
    data = json_mod.loads(result_path.read_text(encoding="utf-8"))
    assert data["items_count"] == 1


def test_save_result_json_persists_answer_text(tmp_path: Path) -> None:
    result_path = save_result_json(
        tmp_path,
        [{"id": 1}],
        "done",
    )

    data = json_mod.loads(result_path.read_text(encoding="utf-8"))
    assert data["result"] == "done"


def test_save_result_json_does_not_duplicate_json_items_artifact(tmp_path: Path) -> None:
    items = [{"id": 1}, {"id": 2}]
    result_path = save_result_json(
        tmp_path,
        items,
        "中文总结：共 2 条，详见 items 字段。",
    )

    data = json_mod.loads(result_path.read_text(encoding="utf-8"))
    assert data["result"] == "中文总结：共 2 条，详见 items 字段。"
    assert data["items"] == items


def test_save_result_json_result_matches_terminal_answer(tmp_path: Path) -> None:
    answer = (
        "已查看 @sama 和 @claudeai 各最新 10 条可见动态，共 20 条。\n\n"
        "@sama 近 10 条要点：\n"
        "- 2025-02-10: very long raw item\n"
        "- 2025-03-30: another raw item\n\n"
        "中文总结与 AI 洞察：\n"
        "1. Sam Altman 更偏个人表达。\n"
        "2. Claude 官方动态更偏产品节奏。"
    )
    result_path = save_result_json(
        tmp_path,
        [{"account": "sama", "url": "https://x.com/sama/status/1"}],
        answer,
    )

    data = json_mod.loads(result_path.read_text(encoding="utf-8"))
    assert data["result"] == answer
    assert "@sama 近 10 条要点" in data["result"]


def test_save_result_json_keeps_short_plain_summary_as_is(tmp_path: Path) -> None:
    answer = "中文总结：\n1. 模型能力继续下放。\n2. 长上下文和 coding 仍是重点。"
    result_path = save_result_json(
        tmp_path,
        [{"account": "claudeai", "url": "https://x.com/claudeai/status/1"}],
        answer,
    )

    data = json_mod.loads(result_path.read_text(encoding="utf-8"))
    assert data["result"] == answer


def test_save_result_json_keeps_long_analysis_without_marker(tmp_path: Path) -> None:
    answer = "\n".join(
        ["已查看 Sam Altman（@sama）最新 10 条 X 动态。", "", "要点概览："]
        + [f"{i}. 这是一条较长的总结性要点，不应被 exporter 裁掉。" for i in range(1, 11)]
        + ["", "AI 新洞察：", "AI agent 正在从聊天工具转向工作流执行层。"]
    )
    result_path = save_result_json(
        tmp_path,
        [{"account": "sama", "url": "https://x.com/sama/status/1"}],
        answer,
    )

    data = json_mod.loads(result_path.read_text(encoding="utf-8"))
    assert data["result"] == answer
    assert "1. 这是一条较长的总结性要点" in data["result"]
    assert "10. 这是一条较长的总结性要点" in data["result"]


def test_save_result_json_reconciles_missing_downloaded_files(tmp_path: Path) -> None:
    result_path = save_result_json(
        tmp_path,
        [
            {"title": "A", "download": {"status": "success", "file": "exists.pdf"}},
            {"title": "B", "download": {"status": "success", "file": "missing.pdf"}},
        ],
        "done",
    )
    (tmp_path / "exists.pdf").write_text("pdf", encoding="utf-8")
    result_path = save_result_json(
        tmp_path,
        [
            {"title": "A", "download": {"status": "success", "file": "exists.pdf"}},
            {"title": "B", "download": {"status": "success", "file": "missing.pdf"}},
        ],
        "done",
    )

    data = json_mod.loads(result_path.read_text(encoding="utf-8"))
    assert data["items"][0]["download"]["file"] == "exists.pdf"
    assert "file" not in data["items"][1]["download"]
    assert data["items"][1]["download"]["status"] == "missing_on_disk"


def test_save_result_json_reconciles_nested_download_files(tmp_path: Path) -> None:
    (tmp_path / "exists.pdf").write_text("pdf", encoding="utf-8")
    result_path = save_result_json(
        tmp_path,
        [
            {"title": "A", "download": {"status": "success", "file": "exists.pdf"}},
            {"title": "B", "download": {"status": "success", "file": "missing.pdf"}},
        ],
        "done",
    )

    data = json_mod.loads(result_path.read_text(encoding="utf-8"))
    assert data["items"][0]["download"]["file"] == "exists.pdf"
    assert "file" not in data["items"][1]["download"]
    assert data["items"][1]["download"]["status"] == "missing_on_disk"


def test_save_result_json_projects_download_entity_into_business_item(tmp_path: Path) -> None:
    (tmp_path / "paper.pdf").write_text("pdf", encoding="utf-8")
    result_path = save_result_json(
        tmp_path,
        [
            {"title": "Paper", "download_link": "https://example.com/paper.pdf"},
            {
                "download_url": "https://example.com/paper.pdf",
                "download": {"status": "success", "file": "paper.pdf"},
            },
        ],
        "done",
    )

    data = json_mod.loads(result_path.read_text(encoding="utf-8"))
    assert data["items_count"] == 2


def test_save_result_json_drops_standalone_download_item_when_richer_item_exists(tmp_path: Path) -> None:
    (tmp_path / "paper.pdf").write_text("pdf", encoding="utf-8")
    result_path = save_result_json(
        tmp_path,
        [
            {
                "download_url": "https://example.com/paper.pdf",
                "download": {"status": "success", "file": "paper.pdf"},
            },
            {
                "title": "Paper",
                "link": "https://example.com/paper",
                "abstract": "demo",
                "download_url": "https://example.com/paper.pdf",
                "download": {"status": "success", "file": "paper.pdf"},
            },
        ],
        "done",
    )

    data = json_mod.loads(result_path.read_text(encoding="utf-8"))
    assert data["items_count"] == 2


def test_check_files_on_disk_accepts_artifacts_file(tmp_path: Path) -> None:
    from hawker_agent.tools.data_tools import check_files_on_disk

    (tmp_path / "artifact.pdf").write_text("pdf", encoding="utf-8")
    report = check_files_on_disk(
        tmp_path,
        [
            {"artifacts": {"file": {"filename": "artifact.pdf"}}},
            {"artifacts": {"file": {"filename": "missing.pdf"}}},
        ],
    )

    assert report["total_downloads"] == 2
    assert report["verified_count"] == 1
    assert report["missing_files"] == ["missing.pdf"]


def test_normalize_items_trims_verbose_pdf_file_payload() -> None:
    items = normalize_items(
        {
            "title": "paper",
            "pdf_file": {
                "ok": True,
                "url": "https://arxiv.org/pdf/2604.21608v1",
                "requested_filename": "paper.pdf",
                "filename": "paper.pdf",
                "path": "/tmp/paper.pdf",
                "size": 123,
                "method": "curl_cffi",
            },
        }
    )

    pdf_file = items[0]["pdf_file"]
    assert pdf_file["url"] == "https://arxiv.org/pdf/2604.21608v1"
    assert pdf_file["filename"] == "paper.pdf"
    assert pdf_file["path"] == "/tmp/paper.pdf"
    assert pdf_file["size"] == 123
    assert "ok" not in pdf_file
    assert "requested_filename" not in pdf_file
    assert "method" not in pdf_file


def test_append_items_observation_reports_changed_and_unchanged() -> None:
    state = CodeAgentState()
    reg = ToolRegistry()
    register_core_actions(reg, state, "/tmp/test")
    append_items = reg.as_namespace_dict()["append_items"]

    asyncio.run(append_items([{"ref": "1", "status": "unknown"}]))
    result = asyncio.run(append_items([{"ref": "1", "status": "success"}]))

    assert result == [{"ref": "1", "status": "success"}]
    assert state.recent_observations[-1].startswith("[append_items] changed=1")


def test_browser_download_auto_records_current_state_evidence() -> None:
    state = CodeAgentState()

    async def fake_browser_download(url: str, filename: str | None = None, run_dir: str | None = None) -> dict[str, object]:
        return {
            "ok": True,
            "url": url,
            "filename": filename or "file.pdf",
            "path": f"{run_dir}/file.pdf",
            "size": 3,
            "method": "curl_cffi",
        }

    sys_dict = build_system_dict(state, {"browser_download": fake_browser_download}, "/tmp/run")
    asyncio.run(sys_dict["browser_download"]("https://example.com/file.pdf", filename="file.pdf"))

    assert state.items.to_list() == [
        {
            "download_url": "https://example.com/file.pdf",
            "download": {
                "status": "success",
                "file": "file.pdf",
                "path": "/tmp/run/file.pdf",
                "size": 3,
            }
        }
    ]


def test_browser_download_merges_back_into_existing_ref_entity() -> None:
    state = CodeAgentState()
    state.items.append(normalize_items([{
        "ref": "paper_1",
        "title": "Paper",
        "link": "https://example.com/paper",
    }]))

    async def fake_browser_download(url: str, filename: str | None = None, run_dir: str | None = None) -> dict[str, object]:
        return {
            "ok": True,
            "url": url,
            "filename": filename or "paper.pdf",
            "path": f"{run_dir}/paper.pdf",
            "size": 3,
            "method": "curl_cffi",
            "ref": "paper_1",
        }

    sys_dict = build_system_dict(state, {"browser_download": fake_browser_download}, "/tmp/run")
    asyncio.run(sys_dict["browser_download"]("https://example.com/paper.pdf", filename="paper.pdf", ref="paper_1"))

    assert state.items.to_list() == [
        {
            "ref": "paper_1",
            "entity_key": "ref:paper_1",
            "title": "Paper",
            "link": "https://example.com/paper",
            "download_url": "https://example.com/paper.pdf",
            "download": {
                "status": "success",
                "file": "paper.pdf",
                "path": "/tmp/run/paper.pdf",
                "size": 3,
            },
        }
    ]


def test_normalize_final_artifact_keeps_business_text_unchanged() -> None:
    text = (
        "完成情况：已采集并查看最近 7 条动态，核心字段包含 time、url、text；"
        "结果数据已保存，正式结果会由系统写入 result.json。"
    )
    artifact = normalize_final_artifact(
        text
    )

    assert artifact["content"] == text
    assert "summary" not in artifact


def test_normalize_final_artifact_drops_summary_field_from_wrapper() -> None:
    content = "第一段总结。\n\n" + ("这是正文。 " * 80)
    artifact = normalize_final_artifact({"type": "text", "content": content, "summary": content})

    assert artifact["content"] == content.strip()
    assert "summary" not in artifact


def test_normalize_final_artifact_parses_expected_json_string() -> None:
    artifact = normalize_final_artifact(
        '{"items": [{"url": "https://example.com"}]}',
        expected_output_format="json",
    )
    assert artifact["type"] == "json"
    assert artifact["items"] == [{"url": "https://example.com"}]


def test_normalize_final_artifact_does_not_guess_business_dict_wrapper() -> None:
    artifact = normalize_final_artifact(
        {
            "type": "article",
            "content": "正文",
            "summary": "摘要",
        }
    )

    assert artifact["type"] == "json"
    assert artifact["content"]["type"] == "article"


def test_recover_items_from_artifact_nested_content_items() -> None:
    artifact = {
        "type": "json",
        "content": {
            "items": [{"url": "https://example.com/a"}, {"url": "https://example.com/b"}],
            "total": 2,
        },
    }

    recovered = recover_items_from_artifact(artifact)
    assert len(recovered) == 2
    assert recovered[0]["url"] == "https://example.com/a"


def test_recover_items_from_artifact_does_not_treat_business_dict_as_one_item() -> None:
    artifact = {
        "type": "json",
        "content": {
            "requested_id": 81,
            "status": "not_found",
            "papers": [{"id": 86}, {"id": 87}],
        },
    }

    assert recover_items_from_artifact(artifact) == []


def test_save_llm_io_json_serializes_complex_payload(tmp_path: Path) -> None:
    class Dummy:
        def __init__(self) -> None:
            self.value = "ok"

    path = save_llm_io_json(
        tmp_path,
        "测试任务",
        records=[
            {
                "step": 1,
                "prompt": {"messages": [{"role": "user", "content": "hello"}]},
                "llm_response": {"raw": Dummy()},
            }
        ],
    )

    assert path.exists()
    data = json_mod.loads(path.read_text(encoding="utf-8"))
    assert data["task"] == "测试任务"
    assert data["steps"] == 1
    assert data["records"][0]["llm_response"]["raw"]["value"] == "ok"


def test_save_llm_io_json_redacts_sensitive_fields_and_non_serializable_values(tmp_path: Path) -> None:
    path = save_llm_io_json(
        tmp_path,
        "测试任务",
        records=[
            {
                "step": 1,
                "prompt": {
                    "headers": {
                        "Authorization": "Bearer secret-token",
                        "x-test": "ok",
                    },
                    "cookies": [{"name": "sid", "value": "abc"}],
                },
                "llm_response": {
                    "raw": object(),
                },
            }
        ],
    )

    data = json_mod.loads(path.read_text(encoding="utf-8"))
    record = data["records"][0]
    assert record["prompt"]["headers"]["Authorization"] == "***redacted***"
    assert record["prompt"]["headers"]["x-test"] == "ok"
    assert record["prompt"]["cookies"] == "***redacted***"
    assert record["llm_response"]["raw"] == "<non-serializable:object>"


class TestHttpTools:
    @pytest.mark.asyncio
    async def test_http_request_uses_httpx_style_json_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        class DummyResponse:
            status_code = 200
            text = '{"ok": true}'
            headers = {"Content-Type": "application/json"}

        class DummyClient:
            async def request(self, method: str, url: str, **kwargs: object) -> DummyResponse:
                captured["method"] = method
                captured["url"] = url
                captured.update(kwargs)
                return DummyResponse()

        async def fake_get_client() -> DummyClient:
            return DummyClient()

        async def fake_validate_url(url: str) -> None:
            return None

        monkeypatch.setattr("hawker_agent.tools.http_tools._get_client", fake_get_client)
        monkeypatch.setattr("hawker_agent.tools.http_tools._validate_url", fake_validate_url)

        raw = await http_request(
            "https://example.com/api",
            method="POST",
            headers={"x-test": "1"},
            json={"page": 1},
        )

        assert raw.startswith("[200]\n")
        assert captured["method"] == "POST"
        assert captured["json"] == {"page": 1}
        assert captured["data"] is None
        assert captured["content"] is None

    @pytest.mark.asyncio
    async def test_http_json_accepts_legacy_body_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        class DummyResponse:
            status_code = 200
            text = '{"items": [1]}'
            headers = {"Content-Type": "application/json"}

        class DummyClient:
            async def request(self, method: str, url: str, **kwargs: object) -> DummyResponse:
                captured.update(kwargs)
                return DummyResponse()

        async def fake_get_client() -> DummyClient:
            return DummyClient()

        async def fake_validate_url(url: str) -> None:
            return None

        monkeypatch.setattr("hawker_agent.tools.http_tools._get_client", fake_get_client)
        monkeypatch.setattr("hawker_agent.tools.http_tools._validate_url", fake_validate_url)

        data = await http_json(
            "https://example.com/api",
            method="POST",
            body='{"page": 1}',
        )

        assert data == {"items": [1]}
        assert captured["content"] == '{"page": 1}'

    @pytest.mark.asyncio
    async def test_http_request_rejects_ambiguous_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_validate_url(url: str) -> None:
            return None

        from hawker_agent.tools import http_tools as http_tools_module
        monkeypatch.setattr(http_tools_module, "_validate_url", fake_validate_url)

        raw = await http_request(
            "https://example.com/api",
            method="POST",
            json={"page": 1},
            data={"page": 1},
        )

        assert raw.startswith("[错误]")
        assert "json/data/content/body" in raw

    @pytest.mark.asyncio
    async def test_validate_url_rejects_private_ip_resolved_from_dns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hawker_agent.tools import http_tools as http_tools_module

        async def fake_resolve_host_ips(hostname: str, port: int | None) -> set[str]:
            assert hostname == "example.com"
            assert port == 443
            return {"169.254.169.254"}

        monkeypatch.setattr(http_tools_module, "_resolve_host_ips", fake_resolve_host_ips)

        with pytest.raises(ValueError, match="private/reserved IP via DNS"):
            await http_tools_module._validate_url("https://example.com/data")

    @pytest.mark.asyncio
    async def test_validate_url_allows_mixed_public_and_reserved_dns_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hawker_agent.tools import http_tools as http_tools_module

        async def fake_resolve_host_ips(hostname: str, port: int | None) -> set[str]:
            assert hostname == "example.com"
            assert port == 443
            return {"169.254.169.254", "93.184.216.34"}

        monkeypatch.setattr(http_tools_module, "_resolve_host_ips", fake_resolve_host_ips)

        await http_tools_module._validate_url("https://example.com/data")


# ─── ACI Tool Refactor ───────────────────────────────────────────


class _StubDomainResponse:
    """Minimal httpx Response stand-in covering the attrs http_request reads."""

    def __init__(self, status_code: int, text: str, content_type: str = "application/json") -> None:
        self.status_code = status_code
        self.text = text
        self.headers = {"Content-Type": content_type}


class _StubClient:
    def __init__(self, response: _StubDomainResponse) -> None:
        self._response = response
        self.last_kwargs: dict[str, object] | None = None

    async def request(self, method: str, url: str, **kwargs: object) -> _StubDomainResponse:  # noqa: ARG002
        self.last_kwargs = kwargs
        return self._response


def _patch_http_client(
    monkeypatch: pytest.MonkeyPatch,
    response: _StubDomainResponse,
) -> _StubClient:
    client = _StubClient(response)

    async def fake_get_client() -> _StubClient:
        return client

    async def fake_validate_url(url: str) -> None:  # noqa: ARG001
        return None

    monkeypatch.setattr("hawker_agent.tools.http_tools._get_client", fake_get_client)
    monkeypatch.setattr("hawker_agent.tools.http_tools._validate_url", fake_validate_url)
    return client


class TestHttpRequestACI:
    @pytest.mark.asyncio
    async def test_long_response_returns_full_body_for_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = "A" * 25_000
        _patch_http_client(monkeypatch, _StubDomainResponse(200, body, content_type="text/plain"))

        raw = await http_request("https://example.com/data", preview_chars=1_000)

        assert raw.startswith("[200]\n")
        status, parsed_body = parse_http_response(raw)
        assert status == 200
        assert parsed_body == body
        assert len(parsed_body) == 25_000

    @pytest.mark.asyncio
    async def test_401_appends_action_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_http_client(monkeypatch, _StubDomainResponse(401, "unauthorized"))

        raw = await http_request("https://example.com/api")

        assert raw.startswith("[401]")
        assert "[hint]" in raw
        assert "Cookie" in raw or "get_cookies" in raw

    @pytest.mark.asyncio
    async def test_429_appends_backoff_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_http_client(monkeypatch, _StubDomainResponse(429, "rate limited"))

        raw = await http_request("https://example.com/api")

        assert "[hint]" in raw
        assert "退避" in raw or "Retry-After" in raw

    @pytest.mark.asyncio
    async def test_5xx_appends_retry_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_http_client(monkeypatch, _StubDomainResponse(503, "down"))

        raw = await http_request("https://example.com/api")

        assert "[hint]" in raw
        assert "503" in raw or "维护" in raw or "限流" in raw

    @pytest.mark.asyncio
    async def test_network_exception_is_translated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx as httpx_module

        class ExplodingClient:
            async def request(self, method: str, url: str, **kwargs: object) -> _StubDomainResponse:  # noqa: ARG002
                raise httpx_module.ReadTimeout("slow upstream")

        async def fake_get_client() -> ExplodingClient:
            return ExplodingClient()

        async def fake_validate_url(url: str) -> None:  # noqa: ARG001
            return None

        monkeypatch.setattr("hawker_agent.tools.http_tools._get_client", fake_get_client)
        monkeypatch.setattr(
            "hawker_agent.tools.http_tools._validate_url", fake_validate_url
        )

        raw = await http_request("https://example.com/api")

        assert raw.startswith("[错误]")
        assert "ReadTimeout" in raw
        assert "[hint]" in raw
        assert "超时" in raw

    @pytest.mark.asyncio
    async def test_url_blocked_returns_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_validate_url(url: str) -> None:  # noqa: ARG001
            raise ValueError("Blocked request to private/reserved IP: 10.0.0.1")

        monkeypatch.setattr(
            "hawker_agent.tools.http_tools._validate_url", fake_validate_url
        )

        raw = await http_request("https://10.0.0.1/api")

        assert raw.startswith("[错误]")
        assert "[hint]" in raw


class TestHttpJsonACI:
    @pytest.mark.asyncio
    async def test_pick_extracts_nested_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_http_client(
            monkeypatch,
            _StubDomainResponse(
                200, json_mod.dumps({"data": {"items": [{"id": 1}, {"id": 2}]}})
            ),
        )

        data = await http_json("https://example.com/api", pick="data.items")

        assert data == [{"id": 1}, {"id": 2}]

    @pytest.mark.asyncio
    async def test_pick_missing_key_raises_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_http_client(
            monkeypatch, _StubDomainResponse(200, json_mod.dumps({"data": {"x": 1}}))
        )

        with pytest.raises(ValueError) as exc:
            await http_json("https://example.com/api", pick="data.items")

        assert "pick='data.items'" in str(exc.value)
        assert "找不到键" in str(exc.value)

    @pytest.mark.asyncio
    async def test_json_pointer_extracts_index(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_http_client(
            monkeypatch,
            _StubDomainResponse(
                200, json_mod.dumps({"data": [{"a": 1}, {"a": 2}]})
            ),
        )

        data = await http_json(
            "https://example.com/api", json_pointer="/data/1"
        )

        assert data == {"a": 2}

    @pytest.mark.asyncio
    async def test_max_items_truncates_list_with_marker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = [{"id": i} for i in range(5)]
        _patch_http_client(
            monkeypatch, _StubDomainResponse(200, json_mod.dumps(payload))
        )

        data = await http_json(
            "https://example.com/api", max_items=2
        )

        assert isinstance(data, list)
        assert len(data) == 3  # 2 条真实 + 1 条 _truncated 提示
        assert data[0] == {"id": 0}
        assert data[1] == {"id": 1}
        assert data[2].get("_truncated") is True
        assert "截断" in data[2].get("hint", "")

    @pytest.mark.asyncio
    async def test_pick_and_json_pointer_are_mutually_exclusive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_http_client(
            monkeypatch, _StubDomainResponse(200, json_mod.dumps({"a": 1}))
        )

        with pytest.raises(ValueError, match="互斥"):
            await http_json("https://example.com/api", pick="a", json_pointer="/a")


def test_status_hint_known_and_fallback() -> None:
    from hawker_agent.tools.http_tools import _status_hint_for

    assert "401" in (_status_hint_for(401) or "")
    assert "429" in (_status_hint_for(429) or "")
    # 未列出的 4xx 走兜底 400 文案
    assert _status_hint_for(418) is not None
    fallback_5xx = _status_hint_for(599)
    assert fallback_5xx is not None
    assert "500" in fallback_5xx or "5xx" in fallback_5xx.lower()
    # 2xx 不应返回行动建议
    assert _status_hint_for(200) is None


# ─── browser.actions cookie helpers ────────────────────────────────


class TestBrowserCookieHelpers:
    def test_cookie_domain_matches_treats_subdomains_as_same(self) -> None:
        from hawker_agent.browser.actions import _cookie_domain_matches

        assert _cookie_domain_matches(".example.com", "example.com") is True
        assert _cookie_domain_matches("api.example.com", "example.com") is True
        assert _cookie_domain_matches("other.com", "example.com") is False
        assert _cookie_domain_matches("", "example.com") is False
        # needle 为空 → 匹配一切
        assert _cookie_domain_matches("other.com", "") is True

    def test_project_cookie_slim_by_default(self) -> None:
        from hawker_agent.browser.actions import _project_cookie

        raw = {
            "name": "sid",
            "value": "abc",
            "domain": "example.com",
            "path": "/",
            "httpOnly": True,
            "sameSite": "Lax",
            "secure": True,
        }
        slim = _project_cookie(raw, verbose=False)
        assert set(slim.keys()) == {"name", "value", "domain", "path"}
        assert _project_cookie(raw, verbose=True) is raw

    def test_truncate_js_raw_preserves_sample_and_marks_truncated(self) -> None:
        from hawker_agent.browser.actions import _truncate_js_raw

        text = "Z" * 50_000
        result = _truncate_js_raw(text)
        assert isinstance(result, dict)
        assert result.get("_truncated") is True
        assert result.get("len") == 50_000
        assert len(result.get("sample", "")) < 50_000

        # 短文本原样返回
        short = "short output"
        assert _truncate_js_raw(short) == short

    def test_urls_differ_ignores_trailing_slash_and_fragment(self) -> None:
        from hawker_agent.browser.actions import _urls_differ

        assert _urls_differ("https://a.com/x", "https://a.com/x/") is False
        assert _urls_differ("https://a.com/x", "https://a.com/x#frag") is False
        assert _urls_differ("https://a.com/x", "https://a.com/y") is True
        assert _urls_differ("", "https://a.com/x") is False
