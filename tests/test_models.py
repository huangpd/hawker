from __future__ import annotations

import dataclasses
import json
import time

import pytest

from hawker_agent.exceptions import (
    BrowserError,
    ConfigurationError,
    CrawlerAgentError,
    ExecutionError,
    LLMError,
    LLMResponseTruncated,
    NoProgressError,
    TokenBudgetExceeded,
)
from hawker_agent.models.cell import CellStatus, CodeCell
from hawker_agent.models.history import CodeAgentHistoryList
from hawker_agent.models.item import ItemStore
from hawker_agent.models.output import CodeAgentModelOutput
from hawker_agent.models.result import CodeAgentResult
from hawker_agent.models.state import CodeAgentState, TokenStats
from hawker_agent.models.step import CodeAgentStepMetadata
from hawker_agent.agent.final_delivery import (
    recover_items_from_final_answer,
    replace_state_items,
)
from hawker_agent.agent.runner import _build_namespace_skip_names
from hawker_agent.agent.evaluator import build_final_evaluation_messages
from hawker_agent.agent.namespace import HawkerNamespace


# ─── exceptions ─────────────────────────────────────────────────


class TestExceptions:
    def test_hierarchy(self) -> None:
        for exc_cls in (
            BrowserError,
            LLMError,
            LLMResponseTruncated,
            ExecutionError,
            TokenBudgetExceeded,
            NoProgressError,
            ConfigurationError,
        ):
            assert issubclass(exc_cls, CrawlerAgentError)

    def test_llm_response_truncated_stores_reason(self) -> None:
        exc = LLMResponseTruncated("上限")
        assert exc.reason == "上限"
        assert "上限" in str(exc)

    def test_catch_base_catches_all(self) -> None:
        for exc_cls in (BrowserError, LLMError, ExecutionError, TokenBudgetExceeded):
            with pytest.raises(CrawlerAgentError):
                raise exc_cls("test")

    def test_llm_response_truncated_is_llm_error(self) -> None:
        assert issubclass(LLMResponseTruncated, LLMError)


# ─── ItemStore ──────────────────────────────────────────────────


class TestItemStore:
    def test_append_basic(self) -> None:
        store = ItemStore()
        added, skipped = store.append([{"url": "a"}, {"url": "b"}])
        assert added == 2
        assert skipped == 0
        assert len(store) == 2

    def test_append_dedup_by_url(self) -> None:
        store = ItemStore()
        store.append([{"url": "a"}])
        added, skipped = store.append([{"url": "a"}])
        assert added == 0
        assert skipped == 1

    def test_append_dedup_by_id(self) -> None:
        store = ItemStore()
        store.append([{"id": "1", "url": "a"}])
        added, skipped = store.append([{"id": "1", "url": "b"}])
        assert added == 0
        assert skipped == 1

    def test_append_dedup_fallback_json(self) -> None:
        store = ItemStore()
        store.append([{"foo": "bar"}])
        added, skipped = store.append([{"foo": "bar"}])
        assert added == 0
        assert skipped == 1

    def test_append_non_dict_skipped(self) -> None:
        store = ItemStore()
        added, skipped = store.append(["not_a_dict", 42, {"url": "a"}])  # type: ignore[list-item]
        assert added == 1
        assert skipped == 2

    def test_to_list_returns_copy(self) -> None:
        store = ItemStore()
        store.append([{"url": "a"}])
        result = store.to_list()
        result.clear()
        assert len(store) == 1

    def test_len_and_bool(self) -> None:
        store = ItemStore()
        assert not store
        assert len(store) == 0
        store.append([{"url": "a"}])
        assert store
        assert len(store) == 1


# ─── TokenStats ─────────────────────────────────────────────────


class TestTokenStats:
    def test_defaults_zero(self) -> None:
        ts = TokenStats()
        assert ts.input_tokens == 0
        assert ts.output_tokens == 0
        assert ts.cached_tokens == 0
        assert ts.total_tokens == 0
        assert ts.cost == 0.0

    def test_add_accumulates(self) -> None:
        ts = TokenStats()
        ts.add(100, 50, 10, 0.01)
        ts.add(200, 100, 20, 0.02)
        assert ts.input_tokens == 300
        assert ts.output_tokens == 150
        assert ts.cached_tokens == 30
        assert ts.total_tokens == 450
        assert ts.cost == pytest.approx(0.03)

    def test_is_over_budget(self) -> None:
        ts = TokenStats()
        ts.add(500, 500, 0, 0.0)
        assert not ts.is_over_budget(1001)
        assert ts.is_over_budget(1000)
        assert ts.is_over_budget(999)

    def test_cost_accumulates(self) -> None:
        ts = TokenStats()
        ts.add(0, 0, 0, 1.5)
        ts.add(0, 0, 0, 2.5)
        assert ts.cost == pytest.approx(4.0)


# ─── CodeAgentState ─────────────────────────────────────────────


class TestCodeAgentState:
    def test_defaults(self) -> None:
        state = CodeAgentState()
        assert state.done is False
        assert state.answer == ""
        assert state.final_answer_requested is None
        assert len(state.items) == 0
        assert state.activity_marker == 0
        assert state.progress_marker == 0

    def test_mark_activity(self) -> None:
        state = CodeAgentState()
        state.mark_activity()
        assert state.activity_marker == 1
        assert state.progress_marker == 1

    def test_snapshot_markers(self) -> None:
        state = CodeAgentState()
        state.mark_activity()
        snap = state.snapshot_markers()
        assert snap == (1, 1)
        state.mark_activity()
        assert snap == (1, 1)  # snapshot unchanged

    def test_is_over_budget_delegates(self) -> None:
        state = CodeAgentState()
        state.token_stats.add(500, 500, 0, 0.0)
        assert state.is_over_budget(1000)
        assert not state.is_over_budget(1001)

    def test_run_id_generated(self) -> None:
        state = CodeAgentState()
        assert len(state.run_id) == 12
        assert all(c in "0123456789abcdef" for c in state.run_id)

    def test_trace_id_generated(self) -> None:
        state = CodeAgentState()
        assert len(state.trace_id) == 32
        assert all(c in "0123456789abcdef" for c in state.trace_id)

    def test_checkpoint_files_default_empty(self) -> None:
        state = CodeAgentState()
        assert state.checkpoint_files == set()

    def test_recover_items_from_final_answer_list(self) -> None:
        payload = json.dumps([{"id": 1, "title": "A"}, {"id": 2, "title": "B"}], ensure_ascii=False)
        items = recover_items_from_final_answer(payload)
        assert len(items) == 2
        assert items[0]["id"] == 1

    def test_recover_items_from_final_answer_items_wrapper(self) -> None:
        payload = json.dumps({"items": [{"url": "https://a"}, {"url": "https://b"}]}, ensure_ascii=False)
        items = recover_items_from_final_answer(payload)
        assert len(items) == 2
        assert items[1]["url"] == "https://b"

    def test_replace_state_items_overwrites_runtime_items(self) -> None:
        state = CodeAgentState()
        state.items.append([{"title": "stale"}, {"title": "old"}])
        replace_state_items(state, [{"title": "new-a"}, {"title": "new-b"}])
        assert state.items.to_list() == [{"title": "new-a"}, {"title": "new-b"}]

    def test_build_namespace_skip_names_uses_system_keys(self) -> None:
        namespace = HawkerNamespace({"nav": object(), "fetch": object(), "json": object()}, "/tmp/run")
        skip_names = _build_namespace_skip_names(namespace)
        assert "nav" in skip_names
        assert "fetch" in skip_names
        assert "json" in skip_names
        assert "run_dir" in skip_names


# ─── CodeCell ───────────────────────────────────────────────────


class TestCodeCell:
    def _make_cell(self, **kwargs) -> CodeCell:  # type: ignore[no-untyped-def]
        defaults = dict(
            step=1,
            thought="分析页面",
            source="nav('https://example.com')",
            output="ok",
            error=None,
            status=CellStatus.SUCCESS,
            duration=1.5,
            usage=TokenStats(),
        )
        defaults.update(kwargs)
        return CodeCell(**defaults)

    def test_frozen(self) -> None:
        cell = self._make_cell()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cell.step = 2  # type: ignore[misc]

    def test_fields(self) -> None:
        usage = TokenStats()
        cell = self._make_cell(usage=usage)
        assert cell.step == 1
        assert cell.thought == "分析页面"
        assert cell.source == "nav('https://example.com')"
        assert cell.output == "ok"
        assert cell.error is None
        assert cell.status == CellStatus.SUCCESS
        assert cell.duration == 1.5
        assert cell.usage is usage

    def test_defaults(self) -> None:
        cell = self._make_cell()
        assert cell.url == ""
        assert cell.items_count == 0


class TestCellStatus:
    def test_values(self) -> None:
        assert CellStatus.PENDING == "pending"
        assert CellStatus.RUNNING == "running"
        assert CellStatus.SUCCESS == "success"
        assert CellStatus.ERROR == "error"


# ─── CodeAgentModelOutput ──────────────────────────────────────


class TestCodeAgentModelOutput:
    def test_has_code_true(self) -> None:
        out = CodeAgentModelOutput(thought="分析", code="nav('url')")
        assert out.has_code is True

    def test_has_code_false(self) -> None:
        out = CodeAgentModelOutput(thought="分析", code="   ")
        assert out.has_code is False

    def test_is_empty(self) -> None:
        out = CodeAgentModelOutput(thought="  ", code="  ")
        assert out.is_empty()

    def test_is_not_empty(self) -> None:
        out = CodeAgentModelOutput(thought="有想法", code="")
        assert not out.is_empty()


# ─── CodeAgentStepMetadata ─────────────────────────────────────


class TestCodeAgentStepMetadata:
    def test_elapsed(self) -> None:
        meta = CodeAgentStepMetadata(step_no=1, started_at=time.time() - 1.0)
        assert meta.elapsed() >= 1.0

    def test_has_progress_new_data(self) -> None:
        state = CodeAgentState()
        meta = CodeAgentStepMetadata(
            step_no=1,
            activity_before=0,
            progress_before=0,
            error="[执行错误]\n有错误",
        )
        state.mark_activity()
        assert meta.has_progress(state)

    def test_has_progress_done(self) -> None:
        state = CodeAgentState()
        state.done = True
        meta = CodeAgentStepMetadata(step_no=1, error="some error")
        assert meta.has_progress(state)

    def test_has_progress_no_error(self) -> None:
        state = CodeAgentState()
        meta = CodeAgentStepMetadata(step_no=1)
        assert meta.has_progress(state)

    def test_no_progress(self) -> None:
        state = CodeAgentState()
        meta = CodeAgentStepMetadata(
            step_no=1,
            activity_before=0,
            progress_before=0,
            error="[执行错误]\n出错了",
        )
        assert not meta.has_progress(state)

    def test_to_cell(self) -> None:
        meta = CodeAgentStepMetadata(step_no=3, started_at=time.time() - 2.0)
        meta.output = "导航成功"
        model_output = CodeAgentModelOutput(thought="分析页面", code="nav('url')")
        usage = TokenStats()
        usage.add(100, 50, 0, 0.01)

        cell = meta.to_cell(model_output, usage, items_count=5)
        assert isinstance(cell, CodeCell)
        assert cell.step == 3
        assert cell.thought == "分析页面"
        assert cell.source == "nav('url')"
        assert cell.output == "导航成功"
        assert cell.error is None
        assert cell.status == CellStatus.SUCCESS
        assert cell.duration >= 2.0
        assert cell.usage is usage
        assert cell.items_count == 5

    def test_to_cell_with_error(self) -> None:
        meta = CodeAgentStepMetadata(step_no=1)
        meta.error = "[执行错误]\nNameError"
        model_output = CodeAgentModelOutput(thought="", code="bad_code()")
        usage = TokenStats()

        cell = meta.to_cell(model_output, usage, items_count=0)
        assert cell.status == CellStatus.ERROR
        assert cell.error == "[执行错误]\nNameError"


# ─── CodeAgentResult ────────────────────────────────────────────


class TestCodeAgentResult:
    def test_str_returns_answer(self) -> None:
        result = CodeAgentResult(answer="完成", success=True)
        assert str(result) == "完成"

    def test_items_count_property(self) -> None:
        result = CodeAgentResult(
            answer="ok",
            success=True,
            items=[{"a": 1}, {"b": 2}],
        )
        assert result.items_count == 2

    def test_defaults(self) -> None:
        result = CodeAgentResult(answer="", success=False)
        assert result.stop_reason == "done"
        assert result.artifact is None
        assert result.run_dir is None
        assert result.log_path is None
        assert result.notebook_path is None
        assert result.result_json_path is None
        assert result.items == []
        assert result.total_steps == 0

    def test_success_flag(self) -> None:
        assert CodeAgentResult(answer="ok", success=True).success is True
        assert CodeAgentResult(answer="fail", success=False).success is False


def test_final_evaluation_prompt_does_not_reference_result_json() -> None:
    messages = build_final_evaluation_messages(
        task="抓取最近 3 条动态并总结",
        final_answer="已完成",
        items=[{"url": "https://example.com"}],
        recent_observations=[],
    )

    assert "items/artifact" in messages[0]["content"]
    assert "result.json" not in messages[0]["content"]


# ─── CodeAgentHistoryList ──────────────────────────────────────


class TestCodeAgentHistoryList:
    def test_from_task(self) -> None:
        h = CodeAgentHistoryList.from_task("抓取数据", "你是爬虫Agent")
        assert len(h) == 1  # one user message (system prompt is separate)

    def test_add_assistant_and_user(self) -> None:
        h = CodeAgentHistoryList.from_task("任务", "系统提示")
        h.add_assistant("response")
        h.add_user("observation")
        assert len(h) == 3

    def test_len(self) -> None:
        h = CodeAgentHistoryList()
        assert len(h) == 0
        h.add_user("hi")
        assert len(h) == 1

    def test_to_prompt_messages_includes_system(self) -> None:
        h = CodeAgentHistoryList.from_task("任务", "系统提示")
        msgs = h.to_prompt_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "系统提示"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "任务"

    def test_inject_dom_ephemeral(self) -> None:
        h = CodeAgentHistoryList.from_task("任务", "系统提示")
        h.inject_dom("<div>DOM</div>")
        msgs = h.to_prompt_messages()
        # DOM should appear as last message
        assert "[Browser State]" in msgs[-1]["content"]
        assert "<div>DOM</div>" in msgs[-1]["content"]
        # Second call should not have DOM
        msgs2 = h.to_prompt_messages()
        assert all("[Browser State]" not in m["content"] for m in msgs2)

    def test_inject_dom_not_in_permanent_history(self) -> None:
        h = CodeAgentHistoryList.from_task("任务", "系统提示")
        h.inject_dom("<div>DOM</div>")
        original_len = len(h)
        h.to_prompt_messages()
        assert len(h) == original_len

    def test_inject_dom_moves_into_workspace_in_notebook_mode(self) -> None:
        h = CodeAgentHistoryList.from_task("任务", "系统提示")
        h.record_step(
            step=1,
            max_steps=5,
            assistant_content="导航\n```python\nnav('https://example.com')\n```",
            observation="[OK] 页面已加载",
            namespace_view={},
            items_count=0,
            total_tokens=100,
            max_total_tokens=1000,
            progress=False,
            had_error=False,
            no_progress_steps=0,
        )
        h.inject_dom("[DOM Diff]\n- 新增区域: dialog")
        msgs = h.to_prompt_messages()
        assert any("[Notebook Workspace]" in m["content"] for m in msgs)
        assert any("[DOM Workspace]" in m["content"] for m in msgs)
        assert any("新增区域: dialog" in m["content"] for m in msgs)
        assert all("[Browser State]" not in m["content"] for m in msgs)

    def test_full_dom_workspace_folds_after_one_prompt(self) -> None:
        h = CodeAgentHistoryList.from_task("任务", "系统提示")
        h.record_step(
            step=1,
            max_steps=5,
            assistant_content="导航\n```python\nnav('https://example.com')\n```",
            observation="[OK] 页面已加载",
            namespace_view={},
            items_count=0,
            total_tokens=100,
            max_total_tokens=1000,
            progress=False,
            had_error=False,
            no_progress_steps=0,
        )
        h.inject_browser_context(
            "<html>very large dom</html>",
            mode="full",
            folded_content="[DOM Summary]\n交互元素: 12",
        )
        msgs_first = h.to_prompt_messages()
        workspace_first = next(m["content"] for m in msgs_first if "[Notebook Workspace]" in m["content"])
        assert "very large dom" in workspace_first
        assert "[mode=full" in workspace_first

        msgs_second = h.to_prompt_messages()
        workspace_second = next(m["content"] for m in msgs_second if "[Notebook Workspace]" in m["content"])
        assert "very large dom" not in workspace_second
        assert "[DOM Summary]" in workspace_second
        assert "[mode=summary" in workspace_second

    def test_build_prompt_package_keeps_split_prompt_parts(self) -> None:
        h = CodeAgentHistoryList.from_task("任务", "系统提示")
        h.record_step(
            step=1,
            max_steps=5,
            assistant_content="导航\n```python\nnav('https://example.com')\n```",
            observation="[OK] 页面已加载",
            namespace_view={"page_index": 1},
            items_count=0,
            total_tokens=100,
            max_total_tokens=1000,
            progress=False,
            had_error=False,
            no_progress_steps=0,
        )
        package = h.build_prompt_package()
        assert package["mode"] == "notebook"
        assert package["system_message"]["role"] == "system"
        assert package["task_message"]["content"] == "任务"
        assert "[Notebook Workspace]" in package["workspace_message"]["content"]
        assert "runtime_snapshot" in package["workspace_sections"]
        assert isinstance(package["source_history_messages"], list)
        assert package["messages"][0]["role"] == "system"

    def test_site_sop_is_rendered_in_notebook_mode(self) -> None:
        h = CodeAgentHistoryList.from_task("打开 https://example.com", "系统提示")
        h.record_step(
            step=1,
            max_steps=5,
            assistant_content="导航\n```python\nawait nav('https://example.com')\n```",
            observation="[OK] 页面已加载",
            namespace_view={},
            items_count=0,
            total_tokens=100,
            max_total_tokens=1000,
            progress=False,
            had_error=False,
            no_progress_steps=0,
        )
        h.set_site_sop(
            "Domain: example.com\nGolden Rule: 优先 SSR 提取\n\n## Gotchas\n- 不要直接猜排序参数"
        )
        workspace = h.build_prompt_package()["workspace_message"]["content"]
        assert "[Site SOP]" in workspace
        assert "Golden Rule: 优先 SSR 提取" in workspace
        assert "不要直接猜排序参数" in workspace

    def test_compress_stub_passthrough(self) -> None:
        h = CodeAgentHistoryList.from_task("任务", "系统提示")
        h.add_assistant("response 1")
        h.add_user("observation 1")
        msgs = h.to_prompt_messages()
        # system + task + assistant + user = 4 messages
        assert len(msgs) == 4


# ─── Package imports ────────────────────────────────────────────


class TestPackageImports:
    def test_models_package_exports(self) -> None:
        from hawker_agent.models import (
            CellStatus,
            CodeAgentHistoryList,
            CodeAgentModelOutput,
            CodeAgentResult,
            CodeAgentState,
            CodeAgentStepMetadata,
            CodeCell,
            ItemStore,
            TokenStats,
        )

        for cls in (
            CellStatus,
            CodeAgentHistoryList,
            CodeAgentModelOutput,
            CodeAgentResult,
            CodeAgentState,
            CodeAgentStepMetadata,
            CodeCell,
            ItemStore,
            TokenStats,
        ):
            assert cls is not None

    def test_top_level_package_exports(self) -> None:
        from hawker_agent import CodeAgentResult, CodeAgentState, CrawlerAgentError

        assert CodeAgentResult is not None
        assert CodeAgentState is not None
        assert CrawlerAgentError is not None
