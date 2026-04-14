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


def _to_jsonable(value: Any) -> Any:
    """将复杂对象尽量转换为可 JSON 序列化的结构。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
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
    return repr(value)


def export_notebook(cells: list[CodeCell], task: str, run_dir: Path) -> Path:
    """将执行历史导出为 Jupyter Notebook (.ipynb) 文件。"""
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
    checkpoint_files: set[str] | None = None,
) -> Path:
    """保存结果 JSON 文件，并清理 checkpoint。"""
    path = run_dir / "result.json"
    
    # 提示：result.json 的 'result' 字段应仅作为语义总结。
    # 如果 answer 过长（可能包含冗余样本或大块文本），进行摘要式截断。
    summary = answer
    if len(answer) > 2000:
        summary = answer[:1990] + "... [摘要已截断，详见 items 字段]"

    data = {
        "result": summary,
        "items": items,
        "items_count": len(items),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("结果已保存: %s (%d 条数据)", path, len(items))

    # result.json 已包含完整数据，清理 checkpoint 文件
    for fname in checkpoint_files or set():
        ckpt = run_dir / fname
        if ckpt.resolve() == path.resolve():
            logger.debug("跳过清理与正式结果同名的 checkpoint: %s", ckpt)
            continue
        if ckpt.exists():
            os.remove(ckpt)
            logger.debug("已清理 checkpoint: %s", ckpt)

    return path


def save_llm_io_json(
    run_dir: Path,
    task: str,
    records: list[dict[str, Any]],
) -> Path:
    """
    保存单次任务的完整大模型输入输出记录。

    参数:
        run_dir (Path): 本次运行目录。
        task (str): 任务描述。
        records (list[dict[str, Any]]): 每步的 LLM 输入输出记录。

    返回:
        Path: 导出的 JSON 文件路径。
    """
    path = run_dir / "llm_io.json"
    payload = {
        "task": task,
        "steps": len(records),
        "records": _to_jsonable(records),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("LLM 输入输出记录已保存: %s", path)
    return path
