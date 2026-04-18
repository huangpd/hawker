from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from hawker_agent.models.cell import CodeCell

logger = logging.getLogger(__name__)

_SENSITIVE_KEYWORDS = (
    "authorization",
    "token",
    "secret",
    "password",
    "passwd",
    "cookie",
    "session",
    "api_key",
    "apikey",
    "access_key",
    "refresh_key",
)


def _is_sensitive_key(key: str) -> bool:
    """判断一个字段名是否应在导出时脱敏。"""
    normalized = key.strip().lower()
    return any(keyword in normalized for keyword in _SENSITIVE_KEYWORDS)


def _to_jsonable(value: Any) -> Any:
    """将复杂对象转换为可 JSON 序列化的结构。

    尝试处理 Path 对象、字典、列表、集合以及具有 model_dump 或 dict 方法的对象。

    Args:
        value (Any): 待转换的对象。

    Returns:
        Any: 转换后的可 JSON 序列化对象。
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        jsonable: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if _is_sensitive_key(key):
                jsonable[key] = "***redacted***"
            else:
                jsonable[key] = _to_jsonable(v)
        return jsonable
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _to_jsonable(value.model_dump())  # type: ignore[union-attr]
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _to_jsonable(value.dict())  # type: ignore[union-attr]
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _to_jsonable(vars(value))
        except Exception:
            pass
    return f"<non-serializable:{type(value).__name__}>"


def export_notebook(cells: list[CodeCell], task: str, run_dir: Path) -> Path:
    """将执行历史导出为 Jupyter Notebook (.ipynb) 文件。

    将代理的每一步思考、代码执行及输出结果格式化为 Notebook 单元格。

    Args:
        cells (list[CodeCell]): 包含执行历史的单元格记录列表。
        task (str): 原始任务描述。
        run_dir (Path): 导出文件的保存目录。

    Returns:
        Path: 生成的 .ipynb 文件路径。
    """
    nb = new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {"name": "python", "version": "3.11.0"}

    # 任务描述
    nb.cells.append(new_markdown_cell(f"# HawkerAgent Task\n\n{task}"))

    for cell in cells:
        # 思考过程作为 markdown cell
        if cell.thought:
            nb.cells.append(
                new_markdown_cell(
                    f"**Step {cell.step}** ({cell.duration:.1f}s | items: {cell.items_count})"
                    f"\n\n{cell.thought}"
                )
            )

        if not cell.source:
            # 无代码块 — 跳过空 code cell
            if cell.output or cell.error:
                note = cell.error or cell.output or ""
                nb.cells.append(new_markdown_cell(f"*[无代码块]* 输出:\n```\n{note}\n```"))
            continue

        # 代码 + 输出
        code_cell = new_code_cell(cell.source)
        code_cell.execution_count = cell.step
        if cell.output:
            code_cell.outputs.append(
                nbformat.v4.new_output(
                    output_type="stream",
                    name="stdout",
                    text=cell.output,
                )
            )
        if cell.error:
            code_cell.outputs.append(
                nbformat.v4.new_output(
                    output_type="error",
                    ename="ExecutionError",
                    evalue=cell.error.split("\n")[0] if cell.error else "",
                    traceback=cell.error.splitlines() if cell.error else [],
                )
            )
        nb.cells.append(code_cell)

    nb_path = run_dir / "run.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    logger.info("Notebook 已保存: %s", nb_path)
    return nb_path


def save_result_json(
    run_dir: Path,
    items: list[dict],
    answer: str,
    final_artifact: dict[str, Any] | None = None,
    checkpoint_files: set[str] | None = None,
) -> Path:
    """保存最终结果为 JSON 文件，并清理中间检查点。

    Args:
        run_dir (Path): 结果文件的保存目录。
        items (list[dict]): 采集到的所有数据项。
        answer (str): 任务的最终摘要回答。
        checkpoint_files (set[str] | None, optional): 待清理的中间检查点文件集合。

    Returns:
        Path: 生成的 result.json 文件路径。
    """
    result_dir = run_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / "result.json"

    result_text = _build_result_text(answer, final_artifact, items)
    artifact_payload = _compact_artifact_for_result(final_artifact, items)

    data = {
        "result": result_text,
        "artifact": artifact_payload,
        "items": items,
        "items_count": len(items),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("结果已保存: %s (%d 条数据)", path, len(items))

    # 正式 result.json 已包含完整数据，清理 run_dir 根目录下的 checkpoint 文件
    for fname in checkpoint_files or set():
        ckpt = run_dir / fname
        if ckpt.resolve() == path.resolve():
            logger.debug("跳过清理与正式结果同名的 checkpoint: %s", ckpt)
            continue
        if ckpt.exists():
            os.remove(ckpt)
            logger.debug("已清理 checkpoint: %s", ckpt)

    return path


def _build_result_text(answer: str, artifact: dict[str, Any] | None, items: list[dict]) -> str:
    """构建面向人类阅读的顶层 result 文本。

    对 JSON 交付，机器可读数据已经由 ``items`` 或 ``artifact.content`` 承载，
    顶层 ``result`` 不再保存一份截断 JSON，避免三处重复表达同一份数据。
    """
    if isinstance(artifact, dict) and str(artifact.get("type") or "").lower() == "json":
        content = artifact.get("content")
        if items:
            return f"[结构化 JSON 结果] 共 {len(items)} 条记录，详见 items 字段。"
        if isinstance(content, dict):
            message = content.get("message")
            status = content.get("status")
            if status or message:
                parts = ["[JSON 结果]"]
                if status:
                    parts.append(f"status={status}")
                if message:
                    parts.append(str(message))
                parts.append("详见 artifact.content 字段。")
                return " ".join(parts)
        return "[JSON 结果] 详见 artifact.content 字段。"

    if len(answer) > 2000:
        return answer[:1990] + "... [结果已截断，详见 artifact.content 或 items 字段]"
    return answer


def _compact_artifact_for_result(
    artifact: dict[str, Any] | None,
    items: list[dict],
) -> dict[str, Any] | None:
    """压缩落盘 artifact，避免与顶层 items 重复存储。

    规则：
    - ``items`` 是结构化列表的权威来源。
    - 如果 JSON artifact 的正文就是同一份 items，则只保留引用标记。
    - 如果 artifact 还包含非 items 的业务元数据，则保留这些正文。
    """
    if artifact is None:
        return None

    payload = _to_jsonable(artifact)
    if not isinstance(payload, dict):
        return payload

    artifact_type = str(payload.get("type") or "").lower()
    if artifact_type != "json":
        return payload

    content = payload.get("content")
    artifact_items = payload.get("items")
    if isinstance(content, list) and content == items:
        return {"type": "json", "content_ref": "items"}
    if isinstance(content, dict) and content.get("items") == items and set(content.keys()) == {"items"}:
        return {"type": "json", "content_ref": "items"}
    if isinstance(artifact_items, list) and artifact_items == items:
        payload.pop("items", None)
    return payload


def save_llm_io_json(
    run_dir: Path,
    task: str,
    records: list[dict[str, Any]],
    healing_records: list[dict[str, Any]] | None = None,
    evaluator_records: list[dict[str, Any]] | None = None,
) -> Path:
    """保存单次任务的完整 LLM 输入输出交互记录。

    Args:
        run_dir (Path): 运行目录。
        task (str): 任务描述。
        records (list[dict[str, Any]]): 每一步的交互详情记录。

    Returns:
        Path: 导出的 llm_io.json 文件路径。
    """
    path = run_dir / "llm_io.json"
    payload = {
        "task": task,
        "steps": len(records),
        "records": _to_jsonable(records),
        "healing_records": _to_jsonable(healing_records or []),
        "evaluator_records": _to_jsonable(evaluator_records or []),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("LLM 输入输出记录已保存: %s", path)
    return path
