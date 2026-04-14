from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from hawker_agent.models.state import TokenStats


@dataclass
class CodeAgentResult:
    """代理运行结果容器。

    该类作为 run() 方法的最终返回值，包含了任务是否成功、模型生成的最终答案、采集到的数据以及运行期间的各项统计元数据。

    Attributes:
        answer (str): 模型生成的最终任务回复。
        success (bool): 任务是否判定为成功完成。
        items (list[dict]): 运行过程中采集到的所有数据项列表。
        run_id (str): 本次运行的唯一 ID。
        model_name (str): 使用的模型名称。
        total_steps (int): 执行的总步数。
        total_duration (float): 运行总耗时（秒）。
        token_stats (TokenStats): 运行过程中的 token 消耗及费用统计。
        stop_reason (Literal["done", "token_budget", "no_progress", "max_steps"]): 任务停止的具体原因。
        run_dir (Path | None): 运行产物的输出目录路径。
        log_path (Path | None): 运行日志文件的路径。
        notebook_path (Path | None): 导出的 Jupyter Notebook 文件路径。
        result_json_path (Path | None): 最终结果 JSON 文件的路径。
        llm_io_path (Path | None): 模型交互记录文件的路径。
    """

    # 核心输出
    answer: str
    success: bool

    # 采集数据
    items: list[dict] = field(default_factory=list)

    # 运行元数据
    run_id: str = ""
    model_name: str = ""
    total_steps: int = 0
    total_duration: float = 0.0
    token_stats: TokenStats = field(default_factory=TokenStats)
    stop_reason: Literal["done", "token_budget", "no_progress", "max_steps"] = "done"

    # 产物路径（任务失败时可能为 None）
    run_dir: Path | None = None
    log_path: Path | None = None
    notebook_path: Path | None = None
    result_json_path: Path | None = None
    llm_io_path: Path | None = None

    @property
    def items_count(self) -> int:
        """获取采集到的数据项总数。

        Returns:
            int: 项目数量。
        """
        return len(self.items)

    def __str__(self) -> str:
        """向后兼容支持：直接打印结果对象将输出 answer 内容。

        Returns:
            str: 任务答案文本。
        """
        return self.answer
