from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hawker_agent.tools.registry import ToolRegistry


def get_type_signature(d: dict, max_keys: int = 10) -> str:
    """生成 dict 的类型签名字符串，用于 http_json 摘要。"""
    parts = []
    for k, v in list(d.items())[:max_keys]:
        if isinstance(v, list):
            inner = type(v[0]).__name__ if v else "Any"
            parts.append(f"{k}: list[{inner}]")
        elif isinstance(v, dict):
            sub_keys = ",".join(list(v.keys())[:5])
            parts.append(f"{k}: dict{{{sub_keys}}}")
        else:
            parts.append(f"{k}: {type(v).__name__}")
    return "{" + ", ".join(parts) + "}"


def parse_http_response(raw: str) -> tuple[int, str]:
    """解析 http_request() 返回值为 (status_code, body)。"""
    raw = raw.strip()
    if raw.startswith("[错误]"):
        raise RuntimeError(raw)
    match = re.match(r"^\[(\d{3})\]\s*\n?", raw)
    if not match:
        raise ValueError(f"无法解析 http_request 返回值: {raw[:120]}")
    return int(match.group(1)), raw[match.end() :]


def clean_items(items: list) -> list[dict]:
    """过滤非 dict 和 _truncated 标记元素。"""
    if not isinstance(items, list):
        raise TypeError(f"clean_items() 需要 list，收到 {type(items).__name__}")
    return [item for item in items if isinstance(item, dict) and not item.get("_truncated")]


def ensure(condition: object, message: str) -> None:
    """断言，不满足时抛 RuntimeError。"""
    if not condition:
        raise RuntimeError(message)


def normalize_items(items: object) -> list[dict]:
    """
    将 str/dict/list 统一转为 list[dict]，并调用 clean_items 过滤。
    迁移自 main.py _normalize_items。
    """
    if isinstance(items, str):
        items = json.loads(items)
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        raise TypeError(f"items 必须是 list/dict/JSON 字符串，收到 {type(items).__name__}")
    return clean_items(items)


def summarize_json(data: object) -> str:
    """生成 http_json 返回值的摘要字符串。"""
    if isinstance(data, list):
        if not data:
            return "[http_json] 返回空列表 []"
        sample = data[0]
        sample_str = json.dumps(sample, ensure_ascii=False)
        if len(sample_str) > 150:
            sample_str = sample_str[:150] + "..."
        sig = f" | 签名: {get_type_signature(sample)}" if isinstance(sample, dict) else ""
        return f"[http_json] {len(data)} 条{sig} | 样本: {sample_str}"
    if isinstance(data, dict):
        return f"[http_json] dict | {get_type_signature(data)}"
    return f"[http_json] {type(data).__name__}"


def save_file(data: str, filename: str, run_dir: str) -> str:
    """保存数据到 run_dir 下的文件。"""
    filepath = os.path.join(run_dir, filename)
    try:
        parsed = json.loads(data)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        count = len(parsed) if isinstance(parsed, list) else 1
        return f"[OK] 已保存 {count} 条记录到 {filepath}"
    except json.JSONDecodeError:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(data)
        return f"[OK] 已保存文本到 {filepath}"


def register_data_tools(registry: ToolRegistry) -> None:
    """将数据处理辅助工具注册到工具注册表。"""
    registry.register(clean_items)
    registry.register(ensure)
    registry.register(normalize_items)
    registry.register(save_file)
    registry.register(summarize_json)
    registry.register(parse_http_response)
