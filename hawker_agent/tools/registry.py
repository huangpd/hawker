from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from hawker_agent.observability import trace

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """单个工具的元信息。"""

    name: str
    fn: Callable
    description: str
    signature: str
    return_type: str
    category: str | None = None


class ToolRegistry:
    """工具注册表，替换原 TOOLS dict 和 _build_tool_desc。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def _wrap_tool(self, fn: Callable, name: str) -> Callable:
        """为工具函数增加 Trace 包装。"""
        # 检查是否是异步函数 (注意要 unwrap 掉已有的装饰器)
        if inspect.iscoroutinefunction(inspect.unwrap(fn)):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with trace(f"tool_{name}", is_tool=True) as span:
                    # 记录输入参数（排除掉内部大对象，只记录字符串形式）
                    span.data["kwargs"] = {k: str(v)[:200] for k, v in kwargs.items() if k not in ("session", "run_dir", "history")}
                    try:
                        result = await fn(*args, **kwargs)
                        # 如果是长字符串或复杂对象，记录摘要
                        span.data["result_len"] = len(str(result)) if result is not None else 0
                        return result
                    except Exception as e:
                        span.status = "error"
                        span.data["error"] = str(e)
                        raise
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with trace(f"tool_{name}", is_tool=True) as span:
                    span.data["kwargs"] = {k: str(v)[:200] for k, v in kwargs.items() if k not in ("session", "run_dir", "history")}
                    try:
                        result = fn(*args, **kwargs)
                        span.data["result_len"] = len(str(result)) if result is not None else 0
                        return result
                    except Exception as e:
                        span.status = "error"
                        span.data["error"] = str(e)
                        raise
            return sync_wrapper

    def register(self, fn: Callable, name: str | None = None, category: str | None = None) -> Callable:
        """注册工具，可作为装饰器使用。"""
        tool_name = name or fn.__name__
        
        # 包装原始函数以支持打点统计
        wrapped_fn = self._wrap_tool(fn, tool_name)
        
        doc = inspect.getdoc(fn) or ""
        summary = doc.splitlines()[0].strip() if doc else ""
        sig = inspect.signature(fn, eval_str=True)
        resolved = inspect.get_annotations(fn, eval_str=True)
        ret = resolved.get("return", str)
        ret_name = getattr(ret, "__name__", str(ret))
        self._tools[tool_name] = ToolSpec(
            name=tool_name,
            fn=wrapped_fn,
            description=summary,
            signature=str(sig),
            return_type=ret_name,
            category=category,
        )
        return wrapped_fn

    def build_description(self) -> str:
        """生成详细的工具文档（包含参数、文档字符串等）。"""
        lines = []
        for spec in self._tools.values():
            is_async = inspect.iscoroutinefunction(inspect.unwrap(spec.fn))
            prefix = "async " if is_async else ""
            lines.append(f"- {prefix}{spec.name}{spec.signature} -> {spec.return_type}: {spec.description}")
        return "\n".join(lines)

    def _get_clean_signature(self, fn: Callable) -> str:
        """提取不含类型注解和内部参数的干净签名。"""
        sig = inspect.signature(inspect.unwrap(fn))
        params = []
        for name, p in sig.parameters.items():
            # 隐藏系统注入的内部参数
            if name in ("session", "run_dir"):
                continue

            p_str = name
            # 处理默认值
            if p.default is not inspect.Parameter.empty:
                p_str += f"={repr(p.default)}"
            # 处理 **kwargs
            elif p.kind == inspect.Parameter.VAR_KEYWORD:
                p_str = "**kwargs"

            params.append(p_str)

        return f"({', '.join(params)})"

    def build_capabilities_list(self, kind: Literal["async", "sync"]) -> str:
        """根据类型生成高质量的能力清单，按类别分组。"""
        from collections import defaultdict
        
        grouped_tools: dict[str | None, list[str]] = defaultdict(list)
        
        for spec in self._tools.values():
            is_async = inspect.iscoroutinefunction(inspect.unwrap(spec.fn))

            # 过滤同步/异步
            if (kind == "async" and not is_async) or (kind == "sync" and is_async):
                continue
                
            # 使用精简后的签名
            clean_sig = self._get_clean_signature(spec.fn)
            
            if kind == "async":
                line = f"- `await {spec.name}{clean_sig}`: {spec.description}"
            else:
                line = f"- `{spec.name}{clean_sig}`: {spec.description}"
                
            grouped_tools[spec.category].append(line)

        # 补全内置异步工具
        if kind == "async":
            grouped_tools["网络 & 数据"].append("- `await asyncio.sleep(n)`: 显式等待 n 秒，常用于确保页面渲染完成。")

        # 排序并生成文本
        # 指定想要的类别顺序（可配置，若没匹配到则放在最后）
        category_order = ["导航与页面", "交互", "网络 & 数据", "数据保存"]
        
        lines = []
        for cat in category_order:
            if cat in grouped_tools and grouped_tools[cat]:
                if kind == "async":  # 同步工具不需要额外打这么多分类标题
                    lines.append(f"\n**{cat}**")
                lines.extend(grouped_tools[cat])
                
        # 处理其他未指定的分类或无分类的工具
        for cat, tools in grouped_tools.items():
            if cat not in category_order and tools:
                if kind == "async":
                    cat_name = cat if cat else "其他工具"
                    lines.append(f"\n**{cat_name}**")
                lines.extend(tools)
                
        return "\n".join(lines).strip() if lines else "- (无)"

    def as_namespace_dict(self) -> dict[str, Callable]:
        """返回供 executor namespace 使用的 {name: fn} 字典。"""
        return {name: spec.fn for name, spec in self._tools.items()}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
