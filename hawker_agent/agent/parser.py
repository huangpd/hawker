from __future__ import annotations

import re

from hawker_agent.models.output import CodeAgentModelOutput


def parse_response(text: str) -> CodeAgentModelOutput:
    """
    从 LLM 响应中提取思考文本和所有代码块。

    支持多个 ```python 块（合并）、```js name 命名块（注入为变量）、
    以及无语言标记的 ``` 块作为 fallback。

    迁移自 main.py _parse_response。
    """
    # 匹配所有 fenced code blocks: ```lang [name]\n...\n```
    block_pattern = re.compile(r"(`{3,})(\w+)(?:\s+(\w+))?\n(.*?)\1", re.DOTALL)
    matches = list(block_pattern.finditer(text))

    if not matches:
        # fallback: 无语言标记的 ``` 块
        generic = re.compile(r"```\n(.*?)```", re.DOTALL)
        g_matches = list(generic.finditer(text))
        if g_matches:
            thought = text[: g_matches[0].start()].strip()
            code = "\n\n".join(m.group(1).strip() for m in g_matches if m.group(1).strip())
            return CodeAgentModelOutput(thought=thought, code=code)
        return CodeAgentModelOutput(thought=text.strip(), code="")

    thought = text[: matches[0].start()].strip()
    python_blocks: list[str] = []
    js_named_blocks: list[str] = []

    for m in matches:
        lang = m.group(2).lower()
        var_name = m.group(3)
        content = m.group(4).rstrip()
        if not content:
            continue
        if lang == "python":
            python_blocks.append(content)
        elif lang in ("js", "javascript") and var_name:
            # 命名 JS 块 → 注入为 Python 字符串变量
            escaped = content.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            js_named_blocks.append(f'{var_name} = "{escaped}"')

    code_parts = js_named_blocks + python_blocks
    code = "\n\n".join(code_parts)
    return CodeAgentModelOutput(thought=thought, code=code)
