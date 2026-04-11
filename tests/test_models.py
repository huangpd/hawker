from __future__ import annotations

import dataclasses
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
        assert result.run_dir is None
        assert result.log_path is None
        assert result.notebook_path is None
        assert result.result_json_path is None
        assert result.items == []
        assert result.total_steps == 0

    def test_success_flag(self) -> None:
        assert CodeAgentResult(answer="ok", success=True).success is True
        assert CodeAgentResult(answer="fail", success=False).success is False


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
