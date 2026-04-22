from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hawker_agent.agent.evaluator import FinalEvaluation
from hawker_agent.agent.final_delivery import process_final_answer_request
from hawker_agent.models.state import CodeAgentState
from hawker_agent.models.step import CodeAgentStepMetadata


@pytest.mark.asyncio
async def test_process_final_answer_request_rejects_when_step_has_error() -> None:
    state = CodeAgentState()
    state.final_answer_requested = "完成"
    state.final_artifact_requested = {"type": "text", "content": "完成"}
    history = MagicMock()
    step_meta = CodeAgentStepMetadata(step_no=3, activity_before=1, progress_before=1)
    step_meta.error = "[执行错误] boom"

    result = await process_final_answer_request(
        task="提取标题",
        step=3,
        state=state,
        step_meta=step_meta,
        history=history,
        observation="[执行错误] boom",
    )

    assert "[final_answer已拒绝] 本步有执行错误" in result
    assert state.final_answer_requested is None
    assert state.final_artifact_requested is None
    history.add_user.assert_not_called()


@pytest.mark.asyncio
async def test_process_final_answer_request_rejects_when_evaluator_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hawker_agent.agent import final_delivery as final_delivery_mod

    state = CodeAgentState()
    state.final_answer_requested = "完成"
    state.final_artifact_requested = {"type": "text", "content": "完成"}
    state.llm_records = [{"execution": {"observation": "已采集 3 条"}}]
    history = MagicMock()
    step_meta = CodeAgentStepMetadata(step_no=3, activity_before=1, progress_before=1)

    async def fake_evaluate_final_delivery(**kwargs):
        return FinalEvaluation(accept=False, reason="字段缺失")

    monkeypatch.setattr(final_delivery_mod, "evaluate_final_delivery", fake_evaluate_final_delivery)

    result = await process_final_answer_request(
        task="提取标题",
        step=3,
        state=state,
        step_meta=step_meta,
        history=history,
        observation="[无输出]",
    )

    assert "[final_answer已拒绝] 字段缺失" in result
    assert state.final_answer_requested is None
    assert state.final_artifact_requested is None
    history.add_user.assert_called_once()


@pytest.mark.asyncio
async def test_process_final_answer_request_no_longer_rejects_first_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hawker_agent.agent import final_delivery as final_delivery_mod

    state = CodeAgentState()
    state.final_answer_requested = "完成"
    state.final_artifact_requested = {"type": "text", "content": "完成"}
    history = MagicMock()
    step_meta = CodeAgentStepMetadata(step_no=1, activity_before=0, progress_before=0)

    async def fake_evaluate_final_delivery(**kwargs):
        return FinalEvaluation(accept=True, reason="ok")

    monkeypatch.setattr(final_delivery_mod, "evaluate_final_delivery", fake_evaluate_final_delivery)

    result = await process_final_answer_request(
        task="提取标题",
        step=1,
        state=state,
        step_meta=step_meta,
        history=history,
        observation="[无输出]",
    )

    assert result == "[无输出]"
    assert state.done is True
    assert state.answer == "完成"
    history.add_user.assert_not_called()


@pytest.mark.asyncio
async def test_process_final_answer_request_accepts_and_replaces_inline_json_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hawker_agent.agent import final_delivery as final_delivery_mod

    state = CodeAgentState()
    state.items.append([{"title": "stale"}])
    state.final_answer_requested = '{"items": [{"title": "A"}, {"title": "B"}]}'
    state.final_artifact_requested = {
        "type": "json",
        "content": {"items": [{"title": "A"}, {"title": "B"}]},
        "items": [{"title": "A"}, {"title": "B"}],
    }
    state.llm_records = [{"execution": {"observation": "已采集 2 条"}}]
    history = MagicMock()
    step_meta = CodeAgentStepMetadata(step_no=3, activity_before=1, progress_before=1)

    async def fake_evaluate_final_delivery(**kwargs):
        return FinalEvaluation(accept=True, reason="ok")

    monkeypatch.setattr(final_delivery_mod, "evaluate_final_delivery", fake_evaluate_final_delivery)

    result = await process_final_answer_request(
        task="请直接返回 JSON，提取 title",
        step=3,
        state=state,
        step_meta=step_meta,
        history=history,
        observation="[无输出]",
    )

    assert result == "[无输出]"
    assert state.done is True
    assert state.answer == '{"items": [{"title": "A"}, {"title": "B"}]}'
    assert state.final_artifact == state.final_artifact_requested
    assert state.items.to_list() == [{"title": "A"}, {"title": "B"}]
    history.add_user.assert_not_called()


@pytest.mark.asyncio
async def test_process_final_answer_request_accepts_and_replaces_summary_mode_items_from_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hawker_agent.agent import final_delivery as final_delivery_mod

    state = CodeAgentState()
    state.items.append([{"title": "stale"}, {"title": "old"}])
    state.final_answer_requested = "已完成，共 10 篇。"
    state.final_artifact_requested = {
        "type": "json",
        "content": {"papers": [{"title": "A"}, {"title": "B"}]},
        "items": [{"title": "A"}, {"title": "B"}],
    }
    history = MagicMock()
    step_meta = CodeAgentStepMetadata(step_no=4, activity_before=1, progress_before=1)

    async def fake_evaluate_final_delivery(**kwargs):
        return FinalEvaluation(accept=True, reason="ok")

    monkeypatch.setattr(final_delivery_mod, "evaluate_final_delivery", fake_evaluate_final_delivery)

    result = await process_final_answer_request(
        task="抓取论文标题和链接，并下载",
        step=4,
        state=state,
        step_meta=step_meta,
        history=history,
        observation="[无输出]",
    )

    assert result == "[无输出]"
    assert state.done is True
    assert state.items.to_list() == [{"title": "A"}, {"title": "B"}]
    history.add_user.assert_not_called()
