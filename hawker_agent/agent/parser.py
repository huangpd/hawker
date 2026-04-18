from __future__ import annotations

import json
import re

from hawker_agent.models.output import CodeAgentModelOutput


def parse_response(text: str) -> CodeAgentModelOutput:
    """从 LLM 响应中提取思考文本和代码块。

    支持多个 ```python 块（合并）、```js 命名块（注入为 Python 变量）以及无标签的 ``` 块作为备选方案。

    Args:
        text (str): 来自 LLM 的原始响应文本。

    Returns:
        CodeAgentModelOutput: 包含提取出的思考内容和合并后的可执行 Python 代码的对象。
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

        # fallback: 截断导致代码块未闭合，尝试提取最后一个 ```python 到文本结尾
        truncated_python = re.search(r"```python(?:\s+\w+)?\n(.*)$", text, re.DOTALL)
        if truncated_python:
            start = truncated_python.start()
            thought = text[:start].strip()
            code = truncated_python.group(1).strip()
            if code:
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
        elif lang in ("js", "javascript") and var_name and var_name.isidentifier():
            # 命名 JS 块 → 注入为 Python 字符串变量
            js_named_blocks.append(f"{var_name} = {json.dumps(content, ensure_ascii=False)}")

    code_parts = js_named_blocks + python_blocks
    code = "\n\n".join(code_parts)
    return CodeAgentModelOutput(thought=thought, code=code)
