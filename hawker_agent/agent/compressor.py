from __future__ import annotations

import json
from typing import Any
from collections.abc import Callable

from hawker_agent.agent.parser import parse_response


def format_preview(text: str, limit: int = 180) -> str:
    """将文本压缩为单行预览，必要时进行截断。

    Args:
        text (str): 要格式化的文本。
        limit (int): 预览的最大长度。默认为 180。

    Returns:
        str: 压缩后的单行预览。
    """
    preview = " ".join(text.split())
    if len(preview) <= limit:
        return preview
    return preview[:limit] + "..."


def truncate_output(text: str, limit: int = 3000) -> str:
    """截断输出文本，同时尝试保持 JSON 的结构完整性。

    Args:
        text (str): 要截断的输出文本。
        limit (int): 允许的最大长度。默认为 3000。

    Returns:
        str: 截断后的文本。
    """
    if len(text) <= limit:
        return text
    # 尝试结构化截断：保持 JSON 语法完整
    stripped = text.strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, list) and len(data) > 6:
            keep = 3
            pruned = (
                data[:keep]
                + [{"_truncated": f"...省略 {len(data) - keep * 2} 条，共 {len(data)} 条"}]
                + data[-keep:]
            )
            result = json.dumps(pruned, ensure_ascii=False)
            if len(result) <= limit:
                return result
        elif isinstance(data, dict) and len(data) > 10:
            keys = list(data.keys())
            pruned = {k: data[k] for k in keys[:10]}
            pruned["_truncated"] = f"...省略 {len(keys) - 10} 个键，共 {len(keys)} 个"
            result = json.dumps(pruned, ensure_ascii=False)
            if len(result) <= limit:
                return result
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return text[:limit] + f"\n... [截断，共{len(text)}字符]"


def extract_observation_text(content: str) -> str:
    """从消息中提取观察 (observation) 部分，忽略控制信息。

    Args:
        content (str): 消息内容。

    Returns:
        str: 提取出的观察文本。
    """
    marker = "Observation:\n"
    if marker not in content:
        return content
    return content.split(marker, 1)[1]


def _short_json(value: Any, limit: int = 120) -> str:
    """将任意值压缩为短 JSON 片段。

    Args:
        value (Any): 要压缩的值。
        limit (int): 最大长度。默认为 120。

    Returns:
        str: 该值的简短字符串表示。
    """
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    return format_preview(text, limit)


def semantic_observation_preview(text: str, limit: int = 320) -> str:
    """将观察结果压缩为高密度的语义摘要。

    对于大型列表或字典，它优先返回计数、模式 (schema) 和样本，而不是全部内容。

    Args:
        text (str): 要压缩的观察文本。
        limit (int): 摘要的最大长度。默认为 320。

    Returns:
        str: 高密度的语义摘要。
    """
    stripped = text.strip()
    if not stripped:
        return "[无输出]"

    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(lines) > 5:
            preview = " | ".join(lines[:3])
            return truncate_output(
                f"{preview} | ... 共 {len(lines)} 行 Observation",
                limit,
            )
        return truncate_output(format_preview(stripped, limit), limit)

    if isinstance(data, list):
        total = len(data)
        if not data:
            return "已返回 0 条数据。"
        if all(isinstance(item, dict) for item in data[:10]):
            schema = sorted({key for item in data[:10] for key in item.keys()})
            samples = [
                {key: item[key] for key in list(item.keys())[:4]}
                for item in data[:2]
            ]
            summary = (
                f"已返回 {total} 条数据。Schema: {schema[:8]}。"
                f" 样本: {_short_json(samples, 180)}"
            )
            return truncate_output(summary, limit)
        samples = [_short_json(item, 60) for item in data[:3]]
        return truncate_output(
            f"已返回列表 {total} 项。前 3 项: {samples}",
            limit,
        )

    if isinstance(data, dict):
        keys = list(data.keys())
        if len(keys) > 8:
            samples = {key: data[key] for key in keys[:4]}
            summary = (
                f"已返回对象，包含 {len(keys)} 个字段。"
                f" 关键字段: {keys[:8]}。样本: {_short_json(samples, 160)}"
            )
            return truncate_output(summary, limit)
        return truncate_output(f"对象结果: {_short_json(data, limit)}", limit)

    return truncate_output(format_preview(str(data), limit), limit)


def summarize_namespace_value(value: Any, limit: int = 90) -> str:
    """为笔记本状态视图生成变量摘要。

    Args:
        value (Any): 要摘要的变量值。
        limit (int): 摘要的最大长度。默认为 90。

    Returns:
        str: 变量类型和内容的简要摘要。
    """
    if isinstance(value, list):
        if not value:
            return "list(0)"
        return f"list({len(value)}) sample={_short_json(value[0], min(limit, 60))}"
    if isinstance(value, dict):
        keys = list(value.keys())
        return f"dict({len(keys)} keys: {', '.join(map(str, keys[:4]))})"
    if isinstance(value, tuple):
        return f"tuple({len(value)})"
    if isinstance(value, set):
        return f"set({len(value)})"
    if isinstance(value, str):
        return f"str({len(value)}): {format_preview(value, limit)}"
    if isinstance(value, (int, float, bool)) or value is None:
        return repr(value)
    return format_preview(f"{type(value).__name__}: {value}", limit)


def build_namespace_snapshot(namespace_view: dict[str, Any], max_items: int = 8) -> str:
    """将会话变量压缩为用于 Prompt 的状态快照。

    Args:
        namespace_view (dict[str, Any]): 当前会话的变量。
        max_items (int): 快照中包含的最大变量数。默认为 8。

    Returns:
        str: 表示变量快照的格式化字符串。
    """
    visible_items = [
        (name, value)
        for name, value in sorted(namespace_view.items())
        if name != "run_dir" and not name.startswith("_")
    ]
    if not visible_items:
        return "无持久化变量。"

    lines = []
    for name, value in visible_items[:max_items]:
        lines.append(f"- {name}: {summarize_namespace_value(value)}")

    remaining = len(visible_items) - max_items
    if remaining > 0:
        lines.append(f"- ... 另有 {remaining} 个变量未展开")
    return "\n".join(lines)


def build_summary_message(history: list[dict[str, str]]) -> dict[str, str]:
    """将历史消息对压缩为一条摘要消息。

    Args:
        history (list[dict[str, str]]): 助手 (assistant) 和用户 (user) 消息列表。

    Returns:
        dict[str, str]: 包含摘要的单条用户消息。
    """
    lines = ["以下是较早步骤的简要摘要："]
    step_no = 1
    i = 0
    while i < len(history):
        assistant_msg = history[i]["content"] if history[i]["role"] == "assistant" else ""
        observation = ""
        if i + 1 < len(history) and history[i + 1]["role"] == "user":
            observation = extract_observation_text(history[i + 1]["content"])
            i += 2
        else:
            i += 1
        parsed = parse_response(assistant_msg)
        lines.append(f"Step {step_no} 分析: {format_preview(parsed.thought, 220)}")
        lines.append(f"Step {step_no} 代码: {format_preview(parsed.code, 220)}")
        lines.append(f"Step {step_no} 输出: {format_preview(observation, 220)}")
        step_no += 1
    return {"role": "user", "content": truncate_output("\n".join(lines), 6000)}


def compress_messages(
    messages: list[dict[str, str]],
    threshold: int,
    count_tokens_fn: Callable[[list[dict[str, str]]], int],
) -> list[dict[str, str]]:
    """当消息历史超过 token 阈值时进行压缩。

    保留第一条消息（任务）和最后 4 条消息（最近 2 步），中间的所有内容都将被压缩。

    Args:
        messages (list[dict[str, str]]): 当前的消息列表。
        threshold (int): 触发压缩的 token 阈值。
        count_tokens_fn (Callable): 计算消息 token 数的函数。

    Returns:
        list[dict[str, str]]: 压缩后的消息列表。
    """
    if count_tokens_fn(messages) <= threshold or len(messages) <= 6:
        return messages
    middle = messages[1:-4]
    if not middle:
        return messages
    return messages[:1] + [build_summary_message(middle)] + messages[-4:]
