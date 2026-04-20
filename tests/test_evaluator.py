from __future__ import annotations

import pytest

from hawker_agent.agent.evaluator import (
    _parse_final_evaluation,
    build_final_evaluation_messages,
    extract_task_requirements,
)
from hawker_agent.models.state import CodeAgentState


class TestEvaluator:
    def test_extract_task_requirements_defaults_to_summary_delivery(self) -> None:
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
        assert requirements.delivery_mode == "summary_with_structured_items"
        assert requirements.expected_output_format is None

    def test_extract_task_requirements_detects_inline_json(self) -> None:
        requirements = extract_task_requirements(
            "请提取 `title` 和 `url`，并直接返回 JSON"
        )
        assert requirements.delivery_mode == "inline_json"
        assert requirements.expected_output_format == "json"

    def test_extract_task_requirements_no_longer_guesses_markdown(self) -> None:
        requirements = extract_task_requirements("请整理成 Markdown 返回，并保留二级标题")
        assert requirements.expected_output_format is None

    def test_extract_task_requirements_no_longer_guesses_text(self) -> None:
        requirements = extract_task_requirements("请给我一个纯文本总结，不要 Markdown")
        assert requirements.expected_output_format is None

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
        assert "不要仅凭 URL 编号" in messages[0]["content"]
        assert "拒绝必须基于任务文本、items 样本或最近观察中的显式证据" in messages[0]["content"]

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
