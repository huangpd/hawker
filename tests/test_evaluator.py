from __future__ import annotations

import pytest

from hawker_agent.agent.evaluator import (
    _build_evidence_report,
    _select_sample_items,
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
        assert "不得因样本量少于 items_count 拒绝" in messages[0]["content"]
        assert "只有任务明确要求下载、保存、上传文件" in messages[0]["content"]
        assert "拒绝标准" in messages[0]["content"]
        assert "单独的业务字段链接不构成文件交付证据" in messages[0]["content"]
        assert "证据统计" in messages[1]["content"]

    def test_build_evidence_report_counts_generic_evidence(self) -> None:
        report = _build_evidence_report(
            [
                {"download": {"status": "success", "file": "a.pdf"}},
                {"artifacts": {"file": {"filename": "b.pdf"}}, "facts": {"downloaded": True}},
            ],
            {"verified_count": 1, "obs_verified_count": 1, "missing_files": [], "empty_files": []},
        )

        assert report == {
            "items_count": 2,
            "file_evidence_items": 2,
            "verified_files": 1,
            "verified_obs_files": 1,
            "missing_files": 0,
            "empty_files": 0,
        }

    def test_build_evidence_report_ignores_business_link_field_names(self) -> None:
        report = _build_evidence_report(
            [
                {"title": "A", "download_link": "https://example.com/a.pdf"},
                {"title": "B", "pdf_url": "https://example.com/b.pdf"},
            ],
            {"verified_count": 1, "obs_verified_count": 0, "missing_files": [], "empty_files": []},
        )

        assert report["file_evidence_items"] == 0
        assert report["verified_files"] == 1

    def test_select_sample_items_prefers_richer_current_state_records(self) -> None:
        items = [
            {"ref": "1", "download": {"status": "unknown"}},
            {"ref": "2", "title": "B", "download": {"status": "success", "file": "b.pdf"}},
            {"ref": "3", "download": {"status": "unknown"}},
            {"ref": "4", "title": "D", "download": {"status": "success", "file": "d.pdf", "size": 10}},
        ]

        sample = _select_sample_items(items, limit=2)

        assert sample == [
            {"ref": "4", "title": "D", "download": {"status": "success", "file": "d.pdf", "size": 10}},
            {"ref": "2", "title": "B", "download": {"status": "success", "file": "b.pdf"}},
        ]

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
