from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


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
        """生成详细的工具文档（包含参数、文档字符串等）。"""
        lines = []
        for spec in self._tools.values():
            is_async = inspect.iscoroutinefunction(inspect.unwrap(spec.fn))
            prefix = "async " if is_async else ""
            lines.append(f"- {prefix}{spec.name}{spec.signature} -> {spec.return_type}: {spec.description}")
        return "\n".join(lines)

    def build_capabilities_list(self, kind: Literal["async", "sync"]) -> str:
        """根据类型（异步或同步）生成高层能力概览。"""
        lines = []
        for spec in self._tools.values():
            is_async = inspect.iscoroutinefunction(inspect.unwrap(spec.fn))

            if kind == "async" and is_async:
                lines.append(f"- `await {spec.name}{spec.signature}`: {spec.description}")
            elif kind == "sync" and not is_async:
                lines.append(f"- `{spec.name}{spec.signature}`: {spec.description}")

        # 特殊处理非注册工具
        if kind == "async":
            lines.append("- `await asyncio.sleep(n)`: 显式等待 n 秒。")

        return "\n".join(lines) if lines else "- (暂无)"

    def as_namespace_dict(self) -> dict[str, Callable]:
        """返回供 executor namespace 使用的 {name: fn} 字典。"""
        return {name: spec.fn for name, spec in self._tools.items()}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
