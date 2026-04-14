from __future__ import annotations

import json

from hawker_agent.agent.compressor import (
    build_summary_message,
    extract_observation_text,
    compress_messages,
    format_preview,
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
