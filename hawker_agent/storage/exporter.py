from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from hawker_agent.models.cell import CodeCell

logger = logging.getLogger(__name__)


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
    data = {
        "result": answer,
        "items": items,
        "items_count": len(items),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("结果已保存: %s (%d 条数据)", path, len(items))

    # result.json 已包含完整数据，清理 checkpoint 文件
    for fname in checkpoint_files or set():
        ckpt = run_dir / fname
        if ckpt.exists():
            os.remove(ckpt)
            logger.debug("已清理 checkpoint: %s", ckpt)

    return path
