from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hawker_agent.agent.step_runtime import run_agent_step
from hawker_agent.models.output import CodeAgentModelOutput
from hawker_agent.models.state import CodeAgentState


class _FakeHistory:
    def __init__(self) -> None:
        self.added_users: list[str] = []
        self.recorded_steps: list[dict] = []
        self.injected_dom: list[str] = []

    def build_prompt_package(self) -> dict:
        return {"messages": [{"role": "user", "content": "task"}]}

    def add_user(self, content: str) -> None:
        self.added_users.append(content)

    def inject_dom(self, dom: str) -> None:
        self.injected_dom.append(dom)

    def record_step(self, **kwargs) -> None:
        self.recorded_steps.append(kwargs)


class _FakeNamespace:
    def get_llm_view(self) -> dict:
        return {"products": [1, 2]}


class _FakeCfg:
    max_total_tokens = 100
    max_no_progress_steps = 2


def _fake_response(
    *,
    text: str = "分析",
    input_tokens: int = 10,
    output_tokens: int = 5,
    total_tokens: int = 15,
    is_truncated: bool = False,
    truncate_reason: str = "max_output_tokens",
):
    return SimpleNamespace(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=0,
        total_tokens=total_tokens,
        cost=0.0,
        is_truncated=is_truncated,
        truncate_reason=truncate_reason,
        raw={"id": "resp"},
    )


@pytest.mark.asyncio
async def test_run_agent_step_skips_on_truncated_response_without_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hawker_agent.agent import step_runtime as step_runtime_mod

    state = CodeAgentState()
    history = _FakeHistory()
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=_fake_response(is_truncated=True))

    monkeypatch.setattr(
        step_runtime_mod,
        "parse_response",
        lambda text: CodeAgentModelOutput(thought="分析", code=""),
    )

    result = await run_agent_step(
        step=2,
        task="提取标题",
        max_steps=5,
        cfg=_FakeCfg(),
        llm=llm,
        history=history,
        namespace=_FakeNamespace(),
        state=state,
        log_path="/tmp/run.log",
        cells=[],
        no_progress_steps=1,
        inject_reflection_prompts=lambda *args, **kwargs: None,
    )

    assert result.skipped is True
    assert result.no_progress_steps == 1
    assert len(state.llm_records) == 1
    assert "响应异常" in history.added_users[0]


@pytest.mark.asyncio
async def test_run_agent_step_returns_done_stop_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hawker_agent.agent import step_runtime as step_runtime_mod

    state = CodeAgentState()
    state.final_answer_requested = "完成"
    history = _FakeHistory()
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=_fake_response(text="思考\n```python\npass\n```"))

    monkeypatch.setattr(
        step_runtime_mod,
        "parse_response",
        lambda text: CodeAgentModelOutput(thought="思考", code="pass"),
    )

    async def fake_execute(*args, **kwargs):
        return "[无输出]"

    async def fake_process_final_answer_request(**kwargs):
        kwargs["state"].done = True
        kwargs["state"].answer = "完成"
        return kwargs["observation"]

    monkeypatch.setattr(step_runtime_mod, "execute", fake_execute)
    monkeypatch.setattr(step_runtime_mod, "process_final_answer_request", fake_process_final_answer_request)

    cells: list = []
    result = await run_agent_step(
        step=3,
        task="提取标题",
        max_steps=5,
        cfg=_FakeCfg(),
        llm=llm,
        history=history,
        namespace=_FakeNamespace(),
        state=state,
        log_path="/tmp/run.log",
        cells=cells,
        no_progress_steps=0,
        inject_reflection_prompts=lambda *args, **kwargs: None,
    )

    assert result.stop_reason == "done"
    assert result.no_progress_steps == 0
    assert len(cells) == 1
    assert len(history.recorded_steps) == 1


@pytest.mark.asyncio
async def test_run_agent_step_returns_no_progress_stop_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hawker_agent.agent import step_runtime as step_runtime_mod

    state = CodeAgentState()
    history = _FakeHistory()
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=_fake_response(text="思考\n```python\nboom\n```"))

    monkeypatch.setattr(
        step_runtime_mod,
        "parse_response",
        lambda text: CodeAgentModelOutput(thought="思考", code="boom"),
    )

    async def fake_execute(*args, **kwargs):
        return "[执行错误] boom"

    monkeypatch.setattr(step_runtime_mod, "execute", fake_execute)

    result = await run_agent_step(
        step=4,
        task="提取标题",
        max_steps=6,
        cfg=_FakeCfg(),
        llm=llm,
        history=history,
        namespace=_FakeNamespace(),
        state=state,
        log_path="/tmp/run.log",
        cells=[],
        no_progress_steps=1,
        inject_reflection_prompts=lambda *args, **kwargs: None,
    )

    assert result.stop_reason == "no_progress"
    assert result.no_progress_steps == 2
