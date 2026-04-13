from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent.parent / "templates"),
    autoescape=select_autoescape([]),
    keep_trailing_newline=False,
)

def _safe_minify_prompt(prompt: str) -> str:
    """
    安全地压缩提示词：
    1. 移除每行行尾的多余空格（不影响代码块的左缩进）。
    2. 将连续超过 2 个以上的空行压缩为单空行。
    这能在不破坏 Markdown 结构和注意力机制的前提下，节约无用的 Token 消耗。
    """
    # 移除行尾空格
    lines = [line.rstrip() for line in prompt.splitlines()]
    cleaned_prompt = "\n".join(lines)
    # 将连续的 3个及以上的换行符 (\n\n\n+) 替换为 2个 (\n\n)
    minified = re.sub(r'\n{3,}', '\n\n', cleaned_prompt)
    return minified.strip()

def build_system_prompt(
    async_capabilities: str,
    sync_capabilities: str,
    instructions: str = ""
) -> str:
    """渲染并安全压缩系统提示词。"""
    template = _env.get_template("system_prompt.jinja2")
    raw_prompt = template.render(
        async_capabilities=async_capabilities,
        sync_capabilities=sync_capabilities,
        instructions=instructions
    )
    return _safe_minify_prompt(raw_prompt)
