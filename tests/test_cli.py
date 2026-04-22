from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from hawker_agent.cli import app, entrypoint
from hawker_agent.config import default_global_config_path, default_scrape_dir, get_settings

runner = CliRunner()


def test_cli_help_renders() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "HawkerAgent" in result.stdout


def test_cli_config_init_writes_env_template() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["config", "init", "--output", ".env", "--no-interactive"])
        assert result.exit_code == 0
        env_file = Path(".env")
        assert env_file.exists()
        text = env_file.read_text(encoding="utf-8")
        assert "OPENAI_API_KEY" in text
        assert "MODEL_NAME" in text


def test_cli_config_init_refuses_overwrite_without_force() -> None:
    with runner.isolated_filesystem():
        Path(".env").write_text("OPENAI_API_KEY=old\n", encoding="utf-8")
        result = runner.invoke(app, ["config", "init", "--output", ".env", "--no-interactive"])
        assert result.exit_code == 1
        assert "已存在" in result.stdout
        assert Path(".env").read_text(encoding="utf-8") == "OPENAI_API_KEY=old\n"


def test_cli_config_show_masks_secrets(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-1234567890")
    monkeypatch.setenv("MODEL_NAME", "openai/gpt-5.4")

    get_settings.cache_clear()
    try:
        result = runner.invoke(app, ["config", "show"])
    finally:
        get_settings.cache_clear()

    assert result.exit_code == 0
    assert "OPENAI_API_KEY" in result.stdout
    assert "sk-1...7890" in result.stdout
    assert "openai/gpt-5.4" in result.stdout
    assert "Config Sources" in result.stdout


def test_cli_doctor_reports_missing_required_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    from hawker_agent.config import get_settings

    with runner.isolated_filesystem():
        get_settings.cache_clear()
        try:
            result = runner.invoke(app, ["doctor"])
        finally:
            get_settings.cache_clear()

    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.stdout
    assert "MODEL_NAME" in result.stdout


def test_entrypoint_does_not_rewrite_config_subcommand() -> None:
    argv = ["hawker", "config", "init"]
    with patch("hawker_agent.cli.app") as mock_app, patch("sys.argv", argv):
        entrypoint()

    assert mock_app.called
    assert argv == ["hawker", "config", "init"]


def test_cli_config_init_default_writes_global_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    get_settings.cache_clear()
    try:
        result = runner.invoke(app, ["config", "init", "--force", "--no-interactive"])
    finally:
        get_settings.cache_clear()

    assert result.exit_code == 0
    target = tmp_path / "home" / "hawker" / "config.env"
    assert target.exists()


def test_cli_config_set_updates_local_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["config", "set", "model", "openai/gpt-5.4"])
        assert result.exit_code == 0
        text = default_global_config_path().read_text(encoding="utf-8")
        assert "MODEL_NAME=openai/gpt-5.4" in text


def test_cli_config_init_interactive_writes_answers() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(
            app,
            ["config", "init", "--output", ".env"],
            input="sk-test\nopenai/gpt-5.4\n\n\n30\n200000\nn\n",
        )
        assert result.exit_code == 0
        text = Path(".env").read_text(encoding="utf-8")
        assert "OPENAI_API_KEY=sk-test" in text
        assert "MODEL_NAME=openai/gpt-5.4" in text


def test_cli_doctor_reports_global_and_local_config_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    with runner.isolated_filesystem():
        get_settings.cache_clear()
        try:
            result = runner.invoke(app, ["doctor"])
        finally:
            get_settings.cache_clear()

    assert result.exit_code == 1
    assert "Config path" in result.stdout


def test_blank_storage_paths_fall_back_to_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    global_env = default_global_config_path()
    global_env.parent.mkdir(parents=True, exist_ok=True)
    global_env.write_text(
        "OPENAI_API_KEY=sk-test\n"
        "MODEL_NAME=openai/gpt-5.4\n"
        "SCRAPE_DIR=\n"
        "KNOWLEDGE_DB_PATH=\n",
        encoding="utf-8",
    )
    with runner.isolated_filesystem():
        get_settings.cache_clear()
        try:
            cfg = get_settings()
        finally:
            get_settings.cache_clear()

    assert cfg.scrape_dir == default_scrape_dir()
    assert cfg.knowledge_db_path == default_scrape_dir() / "knowledge.db"


def test_blank_browser_channel_falls_back_to_none(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    global_env = default_global_config_path()
    global_env.parent.mkdir(parents=True, exist_ok=True)
    global_env.write_text(
        "OPENAI_API_KEY=sk-test\n"
        "MODEL_NAME=openai/gpt-5.4\n"
        "BROWSER_CHANNEL=\n",
        encoding="utf-8",
    )
    with runner.isolated_filesystem():
        get_settings.cache_clear()
        try:
            cfg = get_settings()
        finally:
            get_settings.cache_clear()

    assert cfg.browser_channel is None
