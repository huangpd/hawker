from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hawker_agent.exceptions import LLMError
from hawker_agent.llm.client import (
    LLMResponse,
    _detect_truncation,
    _extract_usage,
    _normalize_api_base,
    _normalize_model_name,
    _usage_to_dict,
)
from hawker_agent.llm.cost import calculate_cost
from hawker_agent.llm.tokenizer import count_tokens, count_tokens_text
from hawker_agent.observability import clear_log_context, configure_logging, set_log_context


# ─── tokenizer ──────────────────────────────────────────────────


class TestCountTokens:
    def test_fallback_on_error(self) -> None:
        with patch("hawker_agent.llm.tokenizer._litellm_token_counter", side_effect=Exception):
            result = count_tokens(
                [{"role": "user", "content": "hello world"}], "test-model"
            )
            assert result > 0

    def test_calls_litellm(self) -> None:
        with patch("hawker_agent.llm.tokenizer._litellm_token_counter", return_value=42) as mock:
            result = count_tokens(
                [{"role": "user", "content": "hello"}], "openai/gpt-4"
            )
            assert result == 42
            mock.assert_called_once()


class TestCountTokensText:
    def test_fallback_on_error(self) -> None:
        with patch("hawker_agent.llm.tokenizer._litellm_token_counter", side_effect=Exception):
            result = count_tokens_text("hello world", "test-model")
            assert result > 0

    def test_calls_litellm(self) -> None:
        with patch("hawker_agent.llm.tokenizer._litellm_token_counter", return_value=10):
            result = count_tokens_text("hello", "openai/gpt-4")
            assert result == 10


# ─── cost ───────────────────────────────────────────────────────


class TestCalculateCost:
    def test_returns_float(self) -> None:
        with patch("hawker_agent.llm.cost._litellm_completion_cost", return_value=0.05):
            assert calculate_cost(object()) == pytest.approx(0.05)

    def test_returns_zero_on_error(self) -> None:
        with patch("hawker_agent.llm.cost._litellm_completion_cost", side_effect=Exception):
            assert calculate_cost(object()) == 0.0


# ─── client helpers ─────────────────────────────────────────────


class TestNormalizeApiBase:
    def test_none(self) -> None:
        assert _normalize_api_base(None) is None

    def test_empty(self) -> None:
        assert _normalize_api_base("") is None

    def test_strips_trailing_slash(self) -> None:
        assert _normalize_api_base("https://api.example.com/") == "https://api.example.com"

    def test_strips_chat_completions(self) -> None:
        assert (
            _normalize_api_base("https://api.example.com/v1/chat/completions")
            == "https://api.example.com"
        )

    def test_strips_responses(self) -> None:
        assert (
            _normalize_api_base("https://api.example.com/v1/responses")
            == "https://api.example.com"
        )

    def test_no_suffix(self) -> None:
        assert _normalize_api_base("https://api.example.com") == "https://api.example.com"


class TestNormalizeModelName:
    def test_adds_openai_prefix(self) -> None:
        assert _normalize_model_name("gpt-4") == "openai/gpt-4"

    def test_preserves_existing_prefix(self) -> None:
        assert _normalize_model_name("anthropic/claude-3") == "anthropic/claude-3"


class TestUsageToDict:
    def test_none(self) -> None:
        assert _usage_to_dict(None) == {}

    def test_dict(self) -> None:
        d = {"prompt_tokens": 100}
        assert _usage_to_dict(d) is d

    def test_model_dump(self) -> None:
        obj = MagicMock()
        obj.model_dump.return_value = {"prompt_tokens": 50}
        del obj.dict  # remove dict method so model_dump is used
        assert _usage_to_dict(obj) == {"prompt_tokens": 50}

    def test_dict_method(self) -> None:
        obj = MagicMock(spec=[])
        obj.dict = MagicMock(return_value={"prompt_tokens": 50})
        assert _usage_to_dict(obj) == {"prompt_tokens": 50}


class TestExtractUsage:
    def test_standard_keys(self) -> None:
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        inp, out, cached, total = _extract_usage(usage)
        assert inp == 100
        assert out == 50
        assert cached == 0
        assert total == 150

    def test_alternate_keys(self) -> None:
        usage = {"input_tokens": 200, "output_tokens": 80}
        inp, out, cached, total = _extract_usage(usage)
        assert inp == 200
        assert out == 80
        assert total == 280

    def test_cached_from_prompt_details(self) -> None:
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 30},
        }
        _, _, cached, _ = _extract_usage(usage)
        assert cached == 30

    def test_cached_from_cache_read(self) -> None:
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cache_read_input_tokens": 25,
        }
        _, _, cached, _ = _extract_usage(usage)
        assert cached == 25

    def test_empty_usage(self) -> None:
        inp, out, cached, total = _extract_usage({})
        assert inp == 0
        assert out == 0
        assert cached == 0
        assert total == 0


class TestDetectTruncation:
    def test_incomplete_status(self) -> None:
        obj = SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_tokens"),
        )
        is_trunc, reason = _detect_truncation("some text", obj)
        assert is_trunc
        assert "截断" in reason  # type: ignore[operator]

    def test_output_near_limit(self) -> None:
        obj = SimpleNamespace(
            status="complete",
            usage=SimpleNamespace(output_tokens=950),
            max_output_tokens=1000,
        )
        is_trunc, reason = _detect_truncation("some text", obj)
        assert is_trunc
        assert "上限" in reason  # type: ignore[operator]

    def test_repeated_output(self) -> None:
        text = "abcdef" * 50
        obj = SimpleNamespace(status="complete")
        is_trunc, reason = _detect_truncation(text, obj)
        assert is_trunc
        assert "重复" in reason  # type: ignore[operator]

    def test_normal_response(self) -> None:
        obj = SimpleNamespace(status="complete")
        is_trunc, reason = _detect_truncation("正常响应内容", obj)
        assert not is_trunc
        assert reason is None


# ─── LLMResponse ────────────────────────────────────────────────


class TestLLMResponse:
    def test_defaults(self) -> None:
        r = LLMResponse(
            text="hello",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
            cost=0.01,
        )
        assert r.is_truncated is False
        assert r.truncate_reason is None
        assert r.raw is None


# ─── LLMClient ──────────────────────────────────────────────────


class TestLLMClient:
    def teardown_method(self) -> None:
        clear_log_context()

    def _make_settings(self) -> MagicMock:
        cfg = MagicMock()
        cfg.openai_api_key = "test-key"
        cfg.openai_base_url = None
        cfg.model_name = "gpt-4"
        cfg.reasoning_effort = ""
        return cfg

    def test_complete_success(self) -> None:
        from hawker_agent.llm.client import LLMClient

        cfg = self._make_settings()
        client = LLMClient(cfg=cfg)

        mock_response = SimpleNamespace(
            output_text="思考内容\n```python\nprint('hi')\n```",
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
            status="complete",
            max_output_tokens=4096,
        )
        mock_response.usage.model_dump = lambda: {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        with (
            patch("hawker_agent.llm.client._litellm_responses", return_value=mock_response),
            patch("hawker_agent.llm.client.calculate_cost", return_value=0.01),
        ):
            result = client.complete([{"role": "user", "content": "test"}])

        assert isinstance(result, LLMResponse)
        assert "思考内容" in result.text
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cost == pytest.approx(0.01)
        assert not result.is_truncated

    def test_complete_raises_llm_error(self) -> None:
        from hawker_agent.llm.client import LLMClient

        cfg = self._make_settings()
        client = LLMClient(cfg=cfg)

        with patch(
            "hawker_agent.llm.client._litellm_responses",
            side_effect=ConnectionError("timeout"),
        ):
            with pytest.raises(LLMError, match="LiteLLM 请求失败"):
                client.complete([{"role": "user", "content": "test"}])

    def test_complete_empty_response_raises(self) -> None:
        from hawker_agent.llm.client import LLMClient

        cfg = self._make_settings()
        client = LLMClient(cfg=cfg)

        mock_response = SimpleNamespace(output_text="", usage=None, status="complete")

        with patch("hawker_agent.llm.client._litellm_responses", return_value=mock_response):
            with pytest.raises(LLMError, match="返回空"):
                client.complete([{"role": "user", "content": "test"}])

    def test_complete_logs_with_context(self, caplog: pytest.LogCaptureFixture) -> None:
        from hawker_agent.llm.client import LLMClient

        configure_logging(force=True)
        set_log_context(trace_id="trace-llm", run_id="run-llm", step=4)

        cfg = self._make_settings()
        client = LLMClient(cfg=cfg)
        mock_response = SimpleNamespace(
            output_text="ok",
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            status="complete",
            max_output_tokens=1024,
        )
        mock_response.usage.model_dump = lambda: {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

        with (
            patch("hawker_agent.llm.client._litellm_responses", return_value=mock_response),
            patch("hawker_agent.llm.client.calculate_cost", return_value=0.01),
            caplog.at_level("INFO", logger="hawker_agent.llm.client"),
        ):
            client.complete([{"role": "user", "content": "test"}])

        assert any(record.trace_id == "trace-llm" for record in caplog.records)
        assert any(record.run_id == "run-llm" for record in caplog.records)
        assert any(record.step == "4" for record in caplog.records)
