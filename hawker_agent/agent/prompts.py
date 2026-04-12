from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent.parent / "templates"),
    autoescape=select_autoescape([]),
    keep_trailing_newline=True,
)


def build_system_prompt(
    async_capabilities: str,
    sync_capabilities: str,
    tool_desc: str,
    instructions: str = ""
) -> str:
    """渲染系统提示词模板。"""
    template = _env.get_template("system_prompt.jinja2")
    return template.render(
        async_capabilities=async_capabilities,
        sync_capabilities=sync_capabilities,
        tool_desc=tool_desc,
        instructions=instructions
    )
