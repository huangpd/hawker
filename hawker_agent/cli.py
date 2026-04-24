from __future__ import annotations

import asyncio
import subprocess
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hawker_agent.config import Settings
from hawker_agent.config import default_global_config_path
from hawker_agent.config import get_settings

app = typer.Typer(help="HawkerAgent — LLM 驱动的自主网络爬虫")
config_app = typer.Typer(help="Initialize and inspect Hawker configuration.")
app.add_typer(config_app, name="config")
console = Console()


@dataclass(frozen=True)
class DoctorRow:
    label: str
    ok: bool
    detail: str

_CONFIG_GROUPS: dict[str, list[str]] = {
    "LLM": [
        "openai_api_key",
        "openai_base_url",
        "model_name",
        "reasoning_effort",
        "small_model_name",
    ],
    "Runtime": [
        "max_steps",
        "max_total_tokens",
        "max_no_progress_steps",
        "message_compression_tokens",
    ],
    "Sidecars": [
        "healer_enabled",
        "healer_reasoning_effort",
        "healer_max_attempts",
        "final_evaluator_enabled",
        "final_evaluator_reasoning_effort",
        "observer_enabled",
        "observer_reasoning_effort",
    ],
    "Browser": [
        "headless",
        "browser_provider",
        "browser_executable_path",
        "browser_user_data_dir",
        "browser_profile_directory",
        "browser_storage_state",
        "browser_channel",
        "browser_cdp_url",
        "browser_use_api_key",
        "browser_use_base_url",
        "browser_use_profile_id",
        "browser_use_proxy_country_code",
        "browser_use_keep_alive",
        "browser_use_enable_recording",
    ],
    "Storage & Logs": [
        "scrape_dir",
        "knowledge_db_path",
        "log_level",
    ],
    "Observability": [
        "langfuse_public_key",
        "langfuse_secret_key",
        "langfuse_base_url",
        "langfuse_environment",
        "langfuse_release",
        "searlo_api_key",
    ],
}

_SECRET_FIELDS = {"openai_api_key", "langfuse_public_key", "langfuse_secret_key"}

_CONFIG_KEY_MAP: dict[str, str] = {
    "api-key": "OPENAI_API_KEY",
    "openai-api-key": "OPENAI_API_KEY",
    "model": "MODEL_NAME",
    "model-name": "MODEL_NAME",
    "base-url": "OPENAI_BASE_URL",
    "openai-base-url": "OPENAI_BASE_URL",
    "small-model": "SMALL_MODEL_NAME",
    "small-model-name": "SMALL_MODEL_NAME",
    "max-steps": "MAX_STEPS",
    "max-total-tokens": "MAX_TOTAL_TOKENS",
    "headless": "HEADLESS",
    "scrape-dir": "SCRAPE_DIR",
    "knowledge-db-path": "KNOWLEDGE_DB_PATH",
    "log-level": "LOG_LEVEL",
}

_SUBCOMMANDS = {"config", "doctor", "run"}


def _run_task(task: str, max_steps: int | None) -> None:
    cfg = get_settings()
    if not cfg.openai_api_key:
        console.print(f"[red]错误: OPENAI_API_KEY 未设置。请检查主配置文件: {default_global_config_path()}[/red]")
        sys.exit(1)

    console.print(Panel(f"[bold blue]任务开始:[/bold blue]\n{task}", title="HawkerAgent", expand=False))

    try:
        from hawker_agent.agent.runner import run

        result = asyncio.run(run(task, max_steps=max_steps or cfg.max_steps))

        # 结果输出
        status_color = "green" if result.success else "yellow"
        console.print(f"\n[bold {status_color}]运行结束 ({result.stop_reason}):[/bold {status_color}]")
        console.print(Panel(result.answer, title="回答", border_style=status_color))

        # 耗时格式化
        m, s = divmod(int(result.total_duration), 60)
        duration_str = f"{m}分{s}秒" if m > 0 else f"{s}秒"

        summary = (
            f"📊 [bold]统计汇总:[/bold]\n"
            f"  - 任务状态: [bold {status_color}]{result.stop_reason}[/bold {status_color}]\n"
            f"  - 采集数据: [cyan]{result.items_count}[/cyan] 条\n"
            f"  - 迭代步数: [cyan]{result.total_steps}[/cyan] 步\n"
            f"  - 总计耗时: [bold cyan]{duration_str}[/bold cyan] ({result.total_duration:.1f}s)\n"
            f"  - 消耗费用: [cyan]${result.token_stats.cost:.4f}[/cyan]\n"
            f"  - Token 统计: [dim]{result.token_stats.total_tokens:,} (in:{result.token_stats.input_tokens:,} out:{result.token_stats.output_tokens:,} cache:{result.token_stats.cached_tokens:,})[/dim]"
        )
        console.print(summary)

        if result.run_dir:
            console.print(f"\n📁 运行产物保存至: [underline]{result.run_dir.resolve()}[/underline]")
            if result.notebook_path:
                console.print(f"📓 Notebook: {result.notebook_path.resolve()}")
            if result.result_json_path:
                console.print(f"📊 JSON 结果: {result.result_json_path.resolve()}")
            if result.llm_io_path:
                console.print(f"🧠 LLM I/O: {result.llm_io_path.resolve()}")

    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断运行。[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]运行出错: {e}[/red]")
        logger = sys.modules.get("logging")
        if logger:
            logger.getLogger("hawker_agent.cli").exception("CLI 异常")
        sys.exit(1)


def _config_target(output: Path | None = None) -> Path:
    if output is not None:
        return output
    return default_global_config_path()


def _load_config_target(path: Path) -> dict[str, str]:
    if path.exists():
        return {**_default_config_values(), **_parse_env_text(path.read_text(encoding="utf-8"))}
    return _default_config_values()


def _write_config_target(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_env_values(values), encoding="utf-8")


def _render_config_sources() -> None:
    primary = default_global_config_path()
    lines = [f"config: {primary}"]
    source_panel = "\n".join(lines)
    console.print(Panel(source_panel, title="Config Sources", expand=False))


def _render_config_group(cfg: Settings, group: str, fields: list[str], *, reveal_secrets: bool) -> None:
    table = Table(title=group, show_header=True, header_style="bold")
    table.add_column("Name")
    table.add_column("Value")
    for name in fields:
        value = getattr(cfg, name)
        shown = str(value) if reveal_secrets else _mask_value(name, value)
        table.add_row(name.upper(), shown)
    console.print(table)


def _mask_value(name: str, value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if name not in _SECRET_FIELDS:
        return text
    if not text:
        return ""
    if len(text) <= 8:
        return "***"
    return f"{text[:4]}...{text[-4:]}"


def _settings_or_exit() -> Settings:
    try:
        return get_settings()
    except Exception as exc:
        console.print(f"[red]配置加载失败: {exc}[/red]")
        console.print(f"运行 [bold]hawker config init[/bold] 生成主配置文件: {default_global_config_path()}")
        raise typer.Exit(2) from exc


def _env_template_text() -> str:
    example = Path(".env.example")
    if example.exists():
        return example.read_text(encoding="utf-8")
    return """# Hawker environment
OPENAI_API_KEY=
MODEL_NAME=openai/gpt-5.4
OPENAI_BASE_URL=
SMALL_MODEL_NAME=
MAX_STEPS=30
MAX_TOTAL_TOKENS=200000
HEADLESS=false
# Browser mode: local | browser_use_cloud
BROWSER_PROVIDER=local
# For browser_use_cloud, set BROWSER_USE_API_KEY and usually BROWSER_USE_PROFILE_ID.
BROWSER_USE_API_KEY=
BROWSER_USE_PROFILE_ID=
BROWSER_USE_PROXY_COUNTRY_CODE=us
SCRAPE_DIR=
LOG_LEVEL=INFO
SEARLO_API_KEY=
"""


def _default_config_values() -> dict[str, str]:
    return {
        "OPENAI_API_KEY": "",
        "MODEL_NAME": "openai/gpt-5.4",
        "OPENAI_BASE_URL": "",
        "SMALL_MODEL_NAME": "gpt-5.4-mini",
        "MAX_STEPS": "30",
        "MAX_TOTAL_TOKENS": "200000",
        "MAX_NO_PROGRESS_STEPS": "10",
        "MESSAGE_COMPRESSION_TOKENS": "12000",
        "HEALER_ENABLED": "true",
        "HEALER_REASONING_EFFORT": "",
        "HEALER_MAX_ATTEMPTS": "3",
        "FINAL_EVALUATOR_ENABLED": "true",
        "FINAL_EVALUATOR_REASONING_EFFORT": "",
        "OBSERVER_ENABLED": "true",
        "OBSERVER_REASONING_EFFORT": "",
        "HEADLESS": "false",
        "BROWSER_PROVIDER": "local",
        "BROWSER_EXECUTABLE_PATH": "",
        "BROWSER_USER_DATA_DIR": "",
        "BROWSER_PROFILE_DIRECTORY": "Default",
        "BROWSER_STORAGE_STATE": "",
        "BROWSER_CHANNEL": "",
        "BROWSER_CDP_URL": "",
        "BROWSER_USE_API_KEY": "",
        "BROWSER_USE_BASE_URL": "",
        "BROWSER_USE_PROFILE_ID": "",
        "BROWSER_USE_PROXY_COUNTRY_CODE": "",
        "BROWSER_USE_KEEP_ALIVE": "false",
        "BROWSER_USE_ENABLE_RECORDING": "false",
        "SCRAPE_DIR": "",
        "KNOWLEDGE_DB_PATH": "",
        "LOG_LEVEL": "INFO",
        "LANGFUSE_PUBLIC_KEY": "",
        "LANGFUSE_SECRET_KEY": "",
        "LANGFUSE_BASE_URL": "",
        "LANGFUSE_ENVIRONMENT": "development",
        "LANGFUSE_RELEASE": "",
        "SEARLO_API_KEY": "",
    }


def _parse_env_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _render_env_values(values: dict[str, str]) -> str:
    lines = [
        "# Hawker CLI configuration",
        "# Generated by `hawker config init`.",
        "",
        "# LLM",
        f"OPENAI_API_KEY={values.get('OPENAI_API_KEY', '')}",
        f"MODEL_NAME={values.get('MODEL_NAME', 'openai/gpt-5.4')}",
        f"OPENAI_BASE_URL={values.get('OPENAI_BASE_URL', '')}",
        f"REASONING_EFFORT={values.get('REASONING_EFFORT', '')}",
        f"SMALL_MODEL_NAME={values.get('SMALL_MODEL_NAME', '')}",
        "",
        "# Runtime",
        f"MAX_STEPS={values.get('MAX_STEPS', '30')}",
        f"MAX_TOTAL_TOKENS={values.get('MAX_TOTAL_TOKENS', '200000')}",
        f"MAX_NO_PROGRESS_STEPS={values.get('MAX_NO_PROGRESS_STEPS', '10')}",
        f"MESSAGE_COMPRESSION_TOKENS={values.get('MESSAGE_COMPRESSION_TOKENS', '12000')}",
        "",
        "# Sidecars",
        f"HEALER_ENABLED={values.get('HEALER_ENABLED', 'true')}",
        f"HEALER_REASONING_EFFORT={values.get('HEALER_REASONING_EFFORT', '')}",
        f"HEALER_MAX_ATTEMPTS={values.get('HEALER_MAX_ATTEMPTS', '3')}",
        f"FINAL_EVALUATOR_ENABLED={values.get('FINAL_EVALUATOR_ENABLED', 'true')}",
        f"FINAL_EVALUATOR_REASONING_EFFORT={values.get('FINAL_EVALUATOR_REASONING_EFFORT', '')}",
        f"OBSERVER_ENABLED={values.get('OBSERVER_ENABLED', 'true')}",
        f"OBSERVER_REASONING_EFFORT={values.get('OBSERVER_REASONING_EFFORT', '')}",
        "",
        "# Browser",
        f"HEADLESS={values.get('HEADLESS', 'false')}",
        "# Choose `local` for local Chromium/Chrome, or `browser_use_cloud` to create",
        "# a Browser Use Cloud session automatically and connect through its CDP URL.",
        f"BROWSER_PROVIDER={values.get('BROWSER_PROVIDER', 'local')}",
        f"BROWSER_EXECUTABLE_PATH={values.get('BROWSER_EXECUTABLE_PATH', '')}",
        f"BROWSER_USER_DATA_DIR={values.get('BROWSER_USER_DATA_DIR', '')}",
        f"BROWSER_PROFILE_DIRECTORY={values.get('BROWSER_PROFILE_DIRECTORY', 'Default')}",
        f"BROWSER_STORAGE_STATE={values.get('BROWSER_STORAGE_STATE', '')}",
        f"BROWSER_CHANNEL={values.get('BROWSER_CHANNEL', '')}",
        "# Use BROWSER_CDP_URL directly if you already have an existing browser endpoint.",
        f"BROWSER_CDP_URL={values.get('BROWSER_CDP_URL', '')}",
        "# Browser Use Cloud settings. PROFILE_ID is strongly recommended so login",
        "# state and cookies can persist across sessions.",
        f"BROWSER_USE_API_KEY={values.get('BROWSER_USE_API_KEY', '')}",
        f"BROWSER_USE_BASE_URL={values.get('BROWSER_USE_BASE_URL', '')}",
        f"BROWSER_USE_PROFILE_ID={values.get('BROWSER_USE_PROFILE_ID', '')}",
        f"BROWSER_USE_PROXY_COUNTRY_CODE={values.get('BROWSER_USE_PROXY_COUNTRY_CODE', '')}",
        f"BROWSER_USE_KEEP_ALIVE={values.get('BROWSER_USE_KEEP_ALIVE', 'false')}",
        f"BROWSER_USE_ENABLE_RECORDING={values.get('BROWSER_USE_ENABLE_RECORDING', 'false')}",
        "",
        "# Storage & logs",
        f"SCRAPE_DIR={values.get('SCRAPE_DIR', '')}",
        f"KNOWLEDGE_DB_PATH={values.get('KNOWLEDGE_DB_PATH', '')}",
        f"LOG_LEVEL={values.get('LOG_LEVEL', 'INFO')}",
        "",
        "# Observability",
        f"LANGFUSE_PUBLIC_KEY={values.get('LANGFUSE_PUBLIC_KEY', '')}",
        f"LANGFUSE_SECRET_KEY={values.get('LANGFUSE_SECRET_KEY', '')}",
        f"LANGFUSE_BASE_URL={values.get('LANGFUSE_BASE_URL', '')}",
        f"LANGFUSE_ENVIRONMENT={values.get('LANGFUSE_ENVIRONMENT', 'development')}",
        f"LANGFUSE_RELEASE={values.get('LANGFUSE_RELEASE', '')}",
        f"SEARLO_API_KEY={values.get('SEARLO_API_KEY', '')}",
        "",
    ]
    return "\n".join(lines)


def _interactive_config_values() -> dict[str, str]:
    console.print(
        Panel(
            "Let's set up Hawker. Press Enter to accept defaults.",
            title="Welcome to Hawker",
            expand=False,
        )
    )
    values = _default_config_values()
    values["OPENAI_API_KEY"] = typer.prompt("OpenAI-compatible API key", hide_input=True)
    values["MODEL_NAME"] = typer.prompt("Main model", default=values["MODEL_NAME"])
    values["OPENAI_BASE_URL"] = typer.prompt(
        "OpenAI-compatible base URL (optional)",
        default=values["OPENAI_BASE_URL"],
        show_default=False,
    )
    values["SMALL_MODEL_NAME"] = typer.prompt(
        "Small model for evaluator/healer (optional)",
        default=values["SMALL_MODEL_NAME"],
        show_default=True,
    )
    values["MAX_STEPS"] = typer.prompt("Max steps", default=values["MAX_STEPS"])
    values["MAX_TOTAL_TOKENS"] = typer.prompt("Max total tokens", default=values["MAX_TOTAL_TOKENS"])
    values["HEADLESS"] = "true" if typer.confirm("Run browser in headless mode?", default=False) else "false"
    return values


@app.callback()
def main() -> None:
    """Hawker CLI root."""
    return None


@app.command("run")
def run_command(
    task: str = typer.Argument(..., help="描述你要爬取的任务"),
    max_steps: int | None = typer.Option(None, "--max-steps", "-s", help="最大允许迭代步数；未传时使用 config.env 中的 MAX_STEPS"),
) -> None:
    """Run a Hawker task explicitly as a subcommand."""
    _run_task(task, max_steps)


@config_app.command("init")
def config_init(
    output: Path | None = typer.Option(None, "--output", "-o", help="配置文件输出路径。"),
    force: bool = typer.Option(False, "--force", "-f", help="覆盖已存在的配置文件。"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="是否使用交互式配置向导。"),
) -> None:
    """Create the Hawker configuration file."""
    target = _config_target(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        console.print(f"[yellow]{target} 已存在。使用 --force 覆盖。[/yellow]")
        raise typer.Exit(1)
    if interactive:
        values = _interactive_config_values()
        _write_config_target(target, values)
    else:
        target.write_text(_env_template_text(), encoding="utf-8")
    console.print(f"[green]已写入配置文件: {target}[/green]")


@config_app.command("show")
def config_show(
    reveal_secrets: bool = typer.Option(False, "--reveal-secrets", help="显示完整密钥。默认会脱敏。"),
) -> None:
    """Show effective Hawker configuration grouped by purpose."""
    cfg = _settings_or_exit()
    _render_config_sources()
    for group, fields in _CONFIG_GROUPS.items():
        _render_config_group(cfg, group, fields, reveal_secrets=reveal_secrets)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="配置项名称，例如 api-key / model / max-steps。"),
    value: str | None = typer.Argument(None, help="配置值。未提供时将交互式输入。"),
) -> None:
    """Set one configuration value in the Hawker config file."""
    env_key = _CONFIG_KEY_MAP.get(key.lower(), key.upper().replace("-", "_"))
    target = _config_target()
    values = _load_config_target(target)
    if value is None:
        value = typer.prompt(env_key, hide_input=env_key in {"OPENAI_API_KEY", "LANGFUSE_SECRET_KEY"})
    values[env_key] = value
    _write_config_target(target, values)
    console.print(f"[green]已更新 {env_key} -> {target}[/green]")


@config_app.command("edit")
def config_edit(
) -> None:
    """Open the Hawker config file in $EDITOR."""
    target = _config_target()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        _write_config_target(target, _default_config_values())
    editor = os.getenv("EDITOR") or os.getenv("VISUAL") or "vi"
    try:
        subprocess.run([editor, str(target)], check=True)
    except FileNotFoundError as exc:
        console.print(f"[red]找不到编辑器: {editor}[/red]")
        raise typer.Exit(1) from exc
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]编辑器退出失败: {exc}[/red]")
        raise typer.Exit(exc.returncode) from exc

def _check_path_writable(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".hawker_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True, str(path)
    except OSError as exc:
        return False, str(exc)


def _optional_path_row(label: str, path: Path) -> DoctorRow:
    detail = str(path) if path.exists() else f"{path} (optional, not found)"
    return DoctorRow(label, True, detail)


def _doctor_rows() -> list[DoctorRow]:
    rows: list[DoctorRow] = [
        DoctorRow("Playwright package", importlib.util.find_spec("playwright") is not None, "importable"),
        DoctorRow("browser-use package", importlib.util.find_spec("browser_use") is not None, "importable"),
    ]
    rows.append(_optional_path_row("Config path", default_global_config_path()))

    try:
        cfg = get_settings()
    except Exception as exc:
        rows.extend(
            [
                DoctorRow("OPENAI_API_KEY", False, "missing"),
                DoctorRow("MODEL_NAME", False, "missing"),
                DoctorRow("Settings", False, str(exc)),
            ]
        )
        return rows

    rows.extend(
        [
            DoctorRow("OPENAI_API_KEY", bool(cfg.openai_api_key.strip()), "configured"),
            DoctorRow("MODEL_NAME", bool(cfg.model_name.strip()), cfg.model_name),
        ]
    )
    scrape_ok, scrape_detail = _check_path_writable(cfg.scrape_dir)
    db_ok, db_detail = _check_path_writable(cfg.knowledge_db_path.parent)
    rows.append(DoctorRow("SCRAPE_DIR writable", scrape_ok, scrape_detail))
    rows.append(DoctorRow("KNOWLEDGE_DB_PATH parent", db_ok, db_detail))
    if cfg.browser_user_data_dir:
        rows.append(
            DoctorRow(
                "BROWSER_USER_DATA_DIR",
                cfg.browser_user_data_dir.exists(),
                str(cfg.browser_user_data_dir),
            )
        )
    if cfg.browser_storage_state:
        rows.append(
            DoctorRow(
                "BROWSER_STORAGE_STATE",
                cfg.browser_storage_state.exists(),
                str(cfg.browser_storage_state),
            )
        )
    return rows


@app.command("doctor")
def doctor() -> None:
    """Check whether the local environment is ready to run Hawker."""
    rows = _doctor_rows()
    table = Table(title="Hawker Doctor", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    failed = False
    for row in rows:
        failed = failed or not row.ok
        table.add_row(row.label, "[green]OK[/green]" if row.ok else "[red]FAIL[/red]", row.detail)
    console.print(table)
    if failed:
        raise typer.Exit(1)


def entrypoint() -> None:
    """Console-script entrypoint for packaged CLI installs."""
    if len(sys.argv) > 1:
        first = sys.argv[1]
        if not first.startswith("-") and first not in _SUBCOMMANDS:
            sys.argv.insert(1, "run")
    app()


if __name__ == "__main__":
    entrypoint()
