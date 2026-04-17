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
