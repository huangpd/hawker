from __future__ import annotations

import json
from collections.abc import Callable

from hawker_agent.agent.parser import parse_response


def format_preview(text: str, limit: int = 180) -> str:
    """将文本压缩为单行预览，超长部分截断。"""
    preview = " ".join(text.split())
    if len(preview) <= limit:
        return preview
    return preview[:limit] + "..."


def truncate_output(text: str, limit: int = 3000) -> str:
    """
    截断输出文本，尽量保持 JSON 结构完整。
    迁移自 main.py _truncate_output。
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
    """从历史消息中提取 Observation 段，忽略 RuntimeStatus 等控制信息。"""
    marker = "Observation:\n"
    if marker not in content:
        return content
    return content.split(marker, 1)[1]


def build_summary_message(history: list[dict[str, str]]) -> dict[str, str]:
    """
    将历史消息对（assistant + user observation）压缩为摘要。
    迁移自 main.py _build_summary_message。
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
    """
    当历史超过 threshold token 时，压缩中间部分为摘要。
    保留最早 1 条（任务消息）和最新 4 条（即最近 2 个步骤）。
    迁移自 main.py _compress_messages。

    注意：原版保留 messages[:2]（system + task），但在新架构中 system prompt
    不在 _messages 列表中（由 CodeAgentHistoryList 单独管理），因此这里保留
    _messages[:1]（task 消息）。
    """
    if count_tokens_fn(messages) <= threshold or len(messages) <= 6:
        return messages
    middle = messages[1:-4]
    if not middle:
        return messages
    return messages[:1] + [build_summary_message(middle)] + messages[-4:]
