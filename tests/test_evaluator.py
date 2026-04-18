from __future__ import annotations

import pytest

from hawker_agent.agent.evaluator import (
    _parse_final_evaluation,
    build_final_evaluation_messages,
    extract_task_requirements,
)
from hawker_agent.agent.runner import _collect_recent_observations
from hawker_agent.models.state import CodeAgentState


class TestEvaluator:
    def test_extract_task_requirements_from_field_block(self) -> None:
        requirements = extract_task_requirements(
            """
            1. 打开页面
            2. 提取字段:
            - title: 标题
            - URL: 链接
            - abstract: 摘要
            返回前 5 条
            """
        )
        assert requirements.required_fields == ["title", "URL", "abstract"]
        assert requirements.expected_count_hint == 5
        assert requirements.expects_inline_json is False
        assert requirements.delivery_mode == "summary_with_structured_items"

    def test_extract_task_requirements_detects_inline_json(self) -> None:
        requirements = extract_task_requirements(
            "请提取 `title` 和 `url`，并直接返回 JSON"
        )
        assert requirements.required_fields == ["title", "url"]
        assert requirements.expects_inline_json is True
        assert requirements.delivery_mode == "inline_json"

    def test_parse_final_evaluation_json(self) -> None:
        result = _parse_final_evaluation('{"accept": false, "reason": "样本字段缺失"}')
        assert result is not None
        assert result.accept is False
        assert result.reason == "样本字段缺失"

    def test_parse_final_evaluation_json_with_missing_requirements(self) -> None:
        result = _parse_final_evaluation(
            '{"accept": false, "reason": "缺少字段", "missing_requirements": ["title", "URL"]}'
        )
        assert result is not None
        assert result.accept is False
        assert result.missing_requirements == ["title", "URL"]

    def test_build_final_evaluation_messages_contains_key_context(self) -> None:
        messages = build_final_evaluation_messages(
            task="提取标题和URL",
            final_answer="已完成",
            items=[{"title": "A", "url": "https://a"}],
            recent_observations=["提取 1 条"],
        )
        assert len(messages) == 2
        assert "final_answer" in messages[1]["content"]
        assert "样本" in messages[1]["content"]
        assert "summary_with_structured_items" in messages[1]["content"]
        assert "不能因为样本条数少于 items_count 就拒绝" in messages[0]["content"]
        assert "优先依据任务要求验收产出物" in messages[0]["content"]

    def test_collect_recent_observations(self) -> None:
        state = CodeAgentState()
        state.llm_records = [
            {"execution": {"observation": "第一页 20 条"}},
            {"execution": {"observation": ""}},
            {"execution": {"observation": "第二页 18 条"}},
        ]
        observations = _collect_recent_observations(state, limit=2)
        assert observations == ["第一页 20 条", "第二页 18 条"]

    def test_collect_recent_observations_skips_error_noise(self) -> None:
        state = CodeAgentState()
        state.llm_records = [
            {"execution": {"observation": "[执行错误]\nNameError: x"}},
            {"execution": {"observation": "未找到 ```python``` 代码块"}},
            {"execution": {"observation": "已保存 7 条数据"}},
        ]
        observations = _collect_recent_observations(state, limit=2)
        assert observations == ["已保存 7 条数据"]

    @pytest.mark.asyncio
    async def test_final_evaluator_does_not_force_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hawker_agent.agent import evaluator as evaluator_mod

        captured_kwargs: dict = {}

        class FakeResponse:
            text = '{"accept": true, "reason": "ok"}'
            input_tokens = 1
            output_tokens = 1
            cached_tokens = 0
            total_tokens = 2
            cost = 0.0

        class FakeClient:
            def __init__(self, cfg): ...

            async def complete_with_model(self, *args, **kwargs):
                captured_kwargs.update(kwargs)
                return FakeResponse()

        class FakeSettings:
            final_evaluator_enabled = True
            small_model_name = "mini"
            final_evaluator_reasoning_effort = ""

        monkeypatch.setattr(evaluator_mod, "get_settings", lambda: FakeSettings())
        monkeypatch.setattr(evaluator_mod, "LLMClient", FakeClient)

        state = CodeAgentState()
        result = await evaluator_mod.evaluate_final_delivery(
            task="提取标题",
            final_answer="完成",
            items=[{"title": "A"}],
            recent_observations=["已提取 1 条"],
            state=state,
        )

        assert result is not None
        assert "temperature" not in captured_kwargs
