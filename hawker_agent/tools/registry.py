from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ToolSpec:
    """单个工具的元信息。"""

    name: str
    fn: Callable
    description: str
    signature: str
    return_type: str


class ToolRegistry:
    """工具注册表，替换原 TOOLS dict 和 _build_tool_desc。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, fn: Callable, name: str | None = None) -> Callable:
        """注册工具，可作为装饰器使用。"""
        tool_name = name or fn.__name__
        doc = inspect.getdoc(fn) or ""
        summary = doc.splitlines()[0].strip() if doc else ""
        sig = inspect.signature(fn, eval_str=True)
        resolved = inspect.get_annotations(fn, eval_str=True)
        ret = resolved.get("return", str)
        ret_name = getattr(ret, "__name__", str(ret))
        self._tools[tool_name] = ToolSpec(
            name=tool_name,
            fn=fn,
            description=summary,
            signature=str(sig),
            return_type=ret_name,
        )
        return fn

    def build_description(self) -> str:
        """生成注入 system prompt 的工具描述文本。"""
        lines = []
        for spec in self._tools.values():
            is_async = inspect.iscoroutinefunction(spec.fn)
            prefix = "async " if is_async else ""
            lines.append(f"- {prefix}{spec.name}{spec.signature} -> {spec.return_type}: {spec.description}")
        return "\n".join(lines)

    def as_namespace_dict(self) -> dict[str, Callable]:
        """返回供 executor namespace 使用的 {name: fn} 字典。"""
        return {name: spec.fn for name, spec in self._tools.items()}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
