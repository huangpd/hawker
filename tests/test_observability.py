from __future__ import annotations

import logging

from pathlib import Path

from hawker_agent.config import Settings
from hawker_agent.observability import (
    bind_log_context,
    clear_log_context,
    collect_observations,
    configure_logging,
    emit_observation,
    generate_trace_id,
    get_log_context,
    set_log_context,
)
from hawker_agent.storage.logger import init_run_dir


class TestTraceId:
    def test_generate_trace_id_is_hex(self) -> None:
        trace_id = generate_trace_id()
        assert len(trace_id) == 32
        assert all(char in "0123456789abcdef" for char in trace_id)


class TestLogContext:
    def teardown_method(self) -> None:
        clear_log_context()

    def test_set_log_context(self) -> None:
        ctx = set_log_context(trace_id="trace-1", run_id="run-1", step=3)
        assert ctx.trace_id == "trace-1"
        assert ctx.run_id == "run-1"
        assert ctx.step == "3"
        assert get_log_context() == ctx

    def test_bind_log_context_restores_previous_value(self) -> None:
        set_log_context(trace_id="outer", run_id="run-a", step=1)

        with bind_log_context(trace_id="inner", step=2):
            assert get_log_context().trace_id == "inner"
            assert get_log_context().run_id == "run-a"
            assert get_log_context().step == "2"

        restored = get_log_context()
        assert restored.trace_id == "outer"
        assert restored.run_id == "run-a"
        assert restored.step == "1"


class TestLoggingIntegration:
    def teardown_method(self) -> None:
        clear_log_context()

    def test_log_record_includes_context(self, caplog) -> None:  # type: ignore[no-untyped-def]
        configure_logging(force=True)
        set_log_context(trace_id="trace-abc", run_id="run-xyz", step=9)

        logger = logging.getLogger("hawker_agent.test")
        with caplog.at_level(logging.INFO, logger="hawker_agent.test"):
            logger.info("hello")

        record = caplog.records[-1]
        assert record.trace_id == "trace-abc"
        assert record.run_id == "run-xyz"
        assert record.step == "9"

    def test_configure_logging_writes_file(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        log_path = tmp_path / "app.log"
        configure_logging(log_path=log_path, force=True)
        set_log_context(trace_id="trace-file", run_id="run-file", step=5)

        logging.getLogger("hawker_agent.file").info("written")

        content = log_path.read_text(encoding="utf-8")
        assert "trace_id=trace-file" in content
        assert "run_id=run-file" in content
        assert "step=5" in content
        assert "written" in content

    def test_emit_observation_prints_without_sink(
        self,
        capsys,  # type: ignore[no-untyped-def]
    ) -> None:
        configure_logging(force=True)
        emit_observation("[obs] something happened")

        stdout = capsys.readouterr().out
        assert "[obs] something happened" in stdout

    def test_emit_observation_uses_active_sink(
        self,
        capsys,  # type: ignore[no-untyped-def]
    ) -> None:
        with collect_observations() as sink:
            emit_observation("[obs] buffered")

        assert sink == ["[obs] buffered"]
        assert capsys.readouterr().out == ""

    def test_configure_logging_replaces_previous_file_handler(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        first_path = tmp_path / "first.log"
        second_path = tmp_path / "second.log"

        configure_logging(log_path=first_path, force=True)
        logging.getLogger("hawker_agent.file").info("first")

        configure_logging(log_path=second_path)
        logging.getLogger("hawker_agent.file").info("second")

        first_content = first_path.read_text(encoding="utf-8")
        second_content = second_path.read_text(encoding="utf-8")
        assert "first" in first_content
        assert "second" not in first_content
        assert "second" in second_content

    def test_init_run_dir_sets_trace_context(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        cfg = Settings.model_construct(
            openai_api_key="test-key",
            model_name="test-model",
            scrape_dir=Path(tmp_path),
            log_level="INFO",
        )

        run_dir, log_path = init_run_dir(
            "demo task",
            cfg,
            run_id="run-fixed-id",
            trace_id="trace-fixed-id-1234567890abcdef",
        )

        assert run_dir.name == "run-fixed-id"
        assert "Trace ID: trace-fixed-id-1234567890abcdef" in log_path.read_text(encoding="utf-8")
        assert (run_dir / "app.log").exists()
        assert get_log_context().trace_id == "trace-fixed-id-1234567890abcdef"
        assert get_log_context().run_id == "run-fixed-id"
