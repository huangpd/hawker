from __future__ import annotations

import json

from hawker_agent.agent.compressor import (
    build_namespace_snapshot,
    build_summary_message,
    extract_observation_text,
    compress_messages,
    format_preview,
    semantic_observation_preview,
    summarize_namespace_value,
    truncate_output,
)
from hawker_agent.models.history import CodeAgentHistoryList


# ─── format_preview ─────────────────────────────────────────────


class TestFormatPreview:
    def test_short_text(self) -> None:
        assert format_preview("hello world") == "hello world"

    def test_collapses_whitespace(self) -> None:
        assert format_preview("hello   \n  world") == "hello world"

    def test_truncates(self) -> None:
        text = "a" * 200
        result = format_preview(text, limit=50)
        assert len(result) == 53  # 50 + "..."
        assert result.endswith("...")


# ─── truncate_output ────────────────────────────────────────────


class TestTruncateOutput:
    def test_short_text_unchanged(self) -> None:
        assert truncate_output("hello", limit=100) == "hello"

    def test_long_text_truncated(self) -> None:
        text = "x" * 5000
        result = truncate_output(text, limit=100)
        assert len(result) <= 200  # truncated + suffix
        assert "截断" in result

    def test_json_list_structured_truncation(self) -> None:
        data = [{"id": i, "name": f"item_{i}"} for i in range(50)]
        text = json.dumps(data)
        # limit large enough for the pruned result but smaller than the full text
        result = truncate_output(text, limit=800)
        parsed = json.loads(result)
        assert any("_truncated" in item for item in parsed if isinstance(item, dict))

    def test_json_dict_structured_truncation(self) -> None:
        data = {f"key_{i}": f"value_{i}" for i in range(50)}
        text = json.dumps(data)
        result = truncate_output(text, limit=500)
        parsed = json.loads(result)
        assert "_truncated" in parsed

    def test_non_json_fallback(self) -> None:
        text = "not json " * 500
        result = truncate_output(text, limit=100)
        assert "截断" in result


# ─── build_summary_message ──────────────────────────────────────


class TestBuildSummaryMessage:
    def test_basic_summary(self) -> None:
        history = [
            {"role": "assistant", "content": "分析\n```python\nnav('url')\n```"},
            {"role": "user", "content": "Observation:\n导航成功"},
        ]
        result = build_summary_message(history)
        assert result["role"] == "user"
        assert "Step 1" in result["content"]
        assert "分析" in result["content"]

    def test_multiple_steps(self) -> None:
        history = [
            {"role": "assistant", "content": "step1 thought\n```python\ncode1()\n```"},
            {"role": "user", "content": "Observation:\nresult1"},
            {"role": "assistant", "content": "step2 thought\n```python\ncode2()\n```"},
            {"role": "user", "content": "Observation:\nresult2"},
        ]
        result = build_summary_message(history)
        assert "Step 1" in result["content"]
        assert "Step 2" in result["content"]

    def test_assistant_without_user(self) -> None:
        history = [
            {"role": "assistant", "content": "只有 assistant 消息"},
        ]
        result = build_summary_message(history)
        assert "Step 1" in result["content"]


class TestExtractObservationText:
    def test_extracts_observation_section(self) -> None:
        content = "[RuntimeStatus] 已采集: 1条\n\nObservation:\n提取成功"
        assert extract_observation_text(content) == "提取成功"


class TestSemanticObservationPreview:
    def test_summarizes_json_list_of_dicts(self) -> None:
        payload = json.dumps(
            [{"title": "a", "price": 1, "url": "/1"}, {"title": "b", "price": 2, "url": "/2"}] * 20,
            ensure_ascii=False,
        )
        result = semantic_observation_preview(payload)
        assert "已返回 40 条数据" in result
        assert "Schema" in result
        assert "title" in result

    def test_summarizes_long_plain_text(self) -> None:
        text = "\n".join(f"line {i}" for i in range(10))
        result = semantic_observation_preview(text)
        assert "共 10 行 Observation" in result


class TestNamespaceSnapshot:
    def test_summarize_namespace_value(self) -> None:
        assert summarize_namespace_value([{"id": 1}]).startswith("list(1)")
        assert summarize_namespace_value({"a": 1, "b": 2}).startswith("dict(2 keys")
        assert summarize_namespace_value("hello").startswith("str(5)")

    def test_build_namespace_snapshot(self) -> None:
        snapshot = build_namespace_snapshot(
            {
                "run_dir": "/tmp/run",
                "items": [{"id": 1}],
                "page_index": 3,
                "_temp": "skip",
            }
        )
        assert "run_dir" not in snapshot
        assert "_temp" not in snapshot
        assert "- items:" in snapshot
        assert "- page_index:" in snapshot


# ─── compress_messages ──────────────────────────────────────────


class TestCompressMessages:
    @staticmethod
    def _fake_count_tokens(messages: list[dict[str, str]]) -> int:
        """简单 token 计数：每条消息 100 tokens。"""
        return len(messages) * 100

    def test_below_threshold_unchanged(self) -> None:
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        result = compress_messages(messages, threshold=10000, count_tokens_fn=self._fake_count_tokens)
        assert result == messages

    def test_few_messages_unchanged(self) -> None:
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        result = compress_messages(
            messages, threshold=100, count_tokens_fn=self._fake_count_tokens
        )
        assert result == messages

    def test_compression_occurs(self) -> None:
        # 1 task + 10 assistant/user pairs = 21 messages, each 100 tokens = 2100
        messages = [{"role": "user", "content": "任务"}]
        for i in range(10):
            messages.append(
                {"role": "assistant", "content": f"思考{i}\n```python\ncode{i}()\n```"}
            )
            messages.append({"role": "user", "content": f"Observation:\n结果{i}"})

        result = compress_messages(
            messages, threshold=500, count_tokens_fn=self._fake_count_tokens
        )
        # Should be shorter than original
        assert len(result) < len(messages)
        # First message (task) preserved
        assert result[0] == messages[0]
        # Last 4 messages preserved
        assert result[-4:] == messages[-4:]
        # Middle is summary
        assert "摘要" in result[1]["content"]


# ─── CodeAgentHistoryList compression integration ───────────────


class TestHistoryCompression:
    @staticmethod
    def _fake_count_tokens(messages: list[dict]) -> int:
        return len(messages) * 100

    def test_no_tokenizer_passthrough(self) -> None:
        h = CodeAgentHistoryList.from_task("任务", "系统提示")
        for i in range(15):
            h.add_assistant(f"response {i}")
            h.add_user(f"observation {i}")
        # Without tokenizer, no compression
        msgs = h.to_prompt_messages()
        # system + task + 30 messages = 32
        assert len(msgs) == 32

    def test_with_tokenizer_compresses(self) -> None:
        h = CodeAgentHistoryList.from_task(
            "任务",
            "系统提示",
            compression_threshold=500,
        )
        h._count_tokens_fn = self._fake_count_tokens
        for i in range(15):
            h.add_assistant(f"思考{i}\n```python\ncode{i}()\n```")
            h.add_user(f"Observation:\n结果{i}")

        msgs = h.to_prompt_messages()
        # Should be compressed (fewer than 32 messages)
        assert len(msgs) < 32
        # System message still first
        assert msgs[0]["role"] == "system"

    def test_notebook_workspace_mode_uses_progress_and_failure_notes(self) -> None:
        h = CodeAgentHistoryList.from_task("抓取任务", "系统提示")
        h.record_step(
            step=1,
            max_steps=10,
            assistant_content="先导航\n```python\nnav('https://example.com')\n```",
            observation='[append_items] +2 -> total=2\n[{"title":"A","url":"/a"},{"title":"B","url":"/b"}]',
            namespace_view={"products": [{"title": "A"}], "page_index": 1},
            items_count=2,
            total_tokens=300,
            max_total_tokens=2000,
            progress=True,
            had_error=False,
            no_progress_steps=0,
        )
        h.record_step(
            step=2,
            max_steps=10,
            assistant_content="继续点击\n```python\nclick('.next')\n```",
            observation="[执行错误]\nTimeoutError: selector not found",
            namespace_view={"products": [{"title": "A"}], "page_index": 1},
            items_count=2,
            total_tokens=500,
            max_total_tokens=2000,
            progress=False,
            had_error=True,
            no_progress_steps=1,
        )

        msgs = h.to_prompt_messages()
        assert msgs[0]["role"] == "system"
        workspace = msgs[2]["content"]
        assert "[Notebook Workspace]" in workspace
        assert "[Milestones]" in workspace
        assert "[Long-Term Memory]" in workspace
        assert "TimeoutError" in workspace
        assert "title" in workspace or "total=2" in workspace
        assert "products" in workspace
