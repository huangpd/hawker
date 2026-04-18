from __future__ import annotations

import json
from typing import Any

from hawker_agent.tools.data_tools import normalize_items

_EXPLICIT_ARTIFACT_TYPES = {"text", "json", "markdown"}


def normalize_final_artifact(answer: object) -> dict[str, Any]:
    """将 final_answer 输入规范化为统一 artifact 结构。"""
    if isinstance(answer, str):
        text = answer.strip()
        if text and text[0] in "[{":
            try:
                parsed = json.loads(text)
            except Exception:
                return _finalize_artifact({"type": "text", "content": answer, "summary": answer})
            artifact = _artifact_from_structured(parsed)
            artifact.setdefault("content", answer)
            artifact.setdefault("summary", answer)
            return _finalize_artifact(artifact)
        return _finalize_artifact({"type": "text", "content": answer, "summary": answer})

    if isinstance(answer, (list, dict)):
        return _finalize_artifact(_artifact_from_structured(answer))

    text = str(answer)
    return _finalize_artifact({"type": "text", "content": text, "summary": text})


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

    summary = artifact.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary

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
            summary = value.get("summary")
            items = value.get("items")
            normalized: dict[str, Any] = {
                "type": artifact_type or "json",
            }
            if content is not None:
                normalized["content"] = content
            if isinstance(summary, str):
                normalized["summary"] = summary
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
    """规范 artifact 的 summary，但不改写正文 content。"""
    normalized = dict(artifact)
    content = normalized.get("content")
    summary = normalized.get("summary")

    if isinstance(content, str):
        normalized["content"] = content.strip()
        if not isinstance(summary, str) or not summary.strip():
            normalized["summary"] = _build_summary_from_text(normalized["content"])

    if isinstance(normalized.get("summary"), str):
        normalized["summary"] = _build_summary_from_text(normalized["summary"])

    if normalized.get("summary") == normalized.get("content") and isinstance(normalized.get("content"), str):
        normalized["summary"] = _build_summary_from_text(normalized["content"])

    return normalized


def _build_summary_from_text(text: str, max_len: int = 280) -> str:
    """从正文生成简短摘要。"""
    compact = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rstrip() + "..."
