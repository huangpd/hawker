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
    """安全地精简 Prompt 以节省 token。

    精简步骤：
    1. 移除每行的行尾空格（保留行首缩进）。
    2. 将连续 3 个及以上的换行符压缩为 2 个换行符。

    Args:
        prompt (str): 原始的 Prompt 字符串。

    Returns:
        str: 精简后的 Prompt 字符串。
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
    """渲染并安全地精简系统 Prompt。

    Args:
        async_capabilities (str): 格式化后的异步工具功能列表。
        sync_capabilities (str): 格式化后的同步工具功能列表。
        instructions (str): 额外的自定义指令。默认为 ""。

    Returns:
        str: 最终渲染并精简后的系统 Prompt。
    """
    template = _env.get_template("system_prompt.jinja2")
    raw_prompt = template.render(
        async_capabilities=async_capabilities,
        sync_capabilities=sync_capabilities,
        instructions=instructions
    )
    return _safe_minify_prompt(raw_prompt)
