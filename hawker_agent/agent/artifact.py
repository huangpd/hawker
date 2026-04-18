from __future__ import annotations

import json
from typing import Any, Literal

from hawker_agent.tools.data_tools import normalize_items

_EXPLICIT_ARTIFACT_TYPES = {"text", "json", "markdown"}


def normalize_final_artifact(
    answer: object,
    expected_output_format: Literal["text", "json", "markdown"] | None = None,
) -> dict[str, Any]:
    """将 final_answer 输入规范化为统一 artifact 结构。"""
    if isinstance(answer, str):
        text = answer.strip()
        if text and text[0] in "[{":
            try:
                parsed = json.loads(text)
            except Exception:
                return _apply_expected_output_format(
                    _finalize_artifact({"type": "text", "content": answer}),
                    expected_output_format,
                )
            artifact = _artifact_from_structured(parsed)
            artifact.setdefault("content", answer)
            return _apply_expected_output_format(_finalize_artifact(artifact), expected_output_format)
        return _apply_expected_output_format(
            _finalize_artifact({"type": "text", "content": answer}),
            expected_output_format,
        )

    if isinstance(answer, (list, dict)):
        return _apply_expected_output_format(
            _finalize_artifact(_artifact_from_structured(answer)),
            expected_output_format,
        )

    text = str(answer)
    return _apply_expected_output_format(
        _finalize_artifact({"type": "text", "content": text}),
        expected_output_format,
    )


def artifact_to_answer_text(artifact: dict[str, Any]) -> str:
    """提取给用户展示/兼容旧流程的最终 answer 文本。"""
    artifact_type = str(artifact.get("type") or "").strip().lower()
    if artifact_type == "json":
        content = artifact.get("content")
        if content is not None and not isinstance(content, str):
            return json.dumps(content, ensure_ascii=False, indent=2)

        items = artifact.get("items")
        if items:
            return json.dumps({"items": items}, ensure_ascii=False, indent=2)

    content = artifact.get("content")
    if isinstance(content, str):
        return content

    items = artifact.get("items")
    if items:
        return json.dumps({"items": items}, ensure_ascii=False, indent=2)

    if content is not None:
        return json.dumps(content, ensure_ascii=False, indent=2)

    return ""


def recover_items_from_artifact(artifact: dict[str, Any] | None) -> list[dict]:
    """从统一 artifact 结构中恢复结构化 items。"""
    if not artifact:
        return []

    items = artifact.get("items")
    if items:
        return normalize_items(items)

    content = artifact.get("content")
    if isinstance(content, dict) and isinstance(content.get("items"), list):
        try:
            return normalize_items(content["items"])
        except Exception:
            return []
    if isinstance(content, (list, dict)):
        try:
            return normalize_items(content)
        except Exception:
            return []

    return []


def _artifact_from_structured(value: list | dict) -> dict[str, Any]:
    """将 list/dict 结构包装成 artifact。"""
    if isinstance(value, dict):
        artifact_type = str(value.get("type") or "").strip().lower()
        has_explicit_payload = "content" in value or "items" in value
        # 只有显式声明 type 且带 content/items 的对象才视为 artifact wrapper。
        # 普通业务 dict 可能合法包含 type/content/summary 字段，不能靠字段名猜测。
        is_explicit_wrapper = artifact_type in _EXPLICIT_ARTIFACT_TYPES and has_explicit_payload
        is_items_delivery = "items" in value and not artifact_type
        if is_explicit_wrapper or is_items_delivery:
            content = value.get("content")
            items = value.get("items")
            normalized: dict[str, Any] = {
                "type": artifact_type or "json",
            }
            if content is not None:
                normalized["content"] = content
            if items is not None:
                try:
                    normalized["items"] = normalize_items(items)
                except Exception:
                    normalized["items"] = items
            if "content" not in normalized:
                normalized["content"] = value
            return normalized

    artifact: dict[str, Any] = {"type": "json", "content": value}
    try:
        artifact["items"] = normalize_items(value)
    except Exception:
        pass
    return artifact


def _finalize_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """规范 artifact 的基础结构，但不改写正文 content 语义。"""
    normalized = dict(artifact)
    content = normalized.get("content")

    if isinstance(content, str):
        normalized["content"] = content.strip()

    normalized.pop("summary", None)
    return normalized


def _apply_expected_output_format(
    artifact: dict[str, Any],
    expected_output_format: Literal["text", "json", "markdown"] | None,
) -> dict[str, Any]:
    """按任务要求做最小化格式对齐，不改写正文内容。"""
    if not expected_output_format:
        return artifact

    normalized = dict(artifact)
    artifact_type = str(normalized.get("type") or "").strip().lower()
    content = normalized.get("content")

    if expected_output_format == "markdown":
        if artifact_type == "text" and isinstance(content, str):
            normalized["type"] = "markdown"
        return normalized

    if expected_output_format == "text":
        if artifact_type == "markdown":
            normalized["type"] = "text"
        return normalized

    if expected_output_format == "json":
        if artifact_type == "json":
            return normalized
        if isinstance(content, str):
            stripped = content.strip()
            if stripped[:1] in "[{":
                try:
                    parsed = json.loads(stripped)
                except Exception:
                    return normalized
                return _finalize_artifact(_artifact_from_structured(parsed))
        return normalized

    return normalized
