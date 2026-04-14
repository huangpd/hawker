from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CodeAgentModelOutput:
    """LLM 响应的语义解析结果。

    该类用于存储从模型原始输出中解析出的思考过程和执行代码，替换了原有的简单元组返回方式。

    Attributes:
        thought (str): 模型生成的思考过程或推理逻辑。
        code (str): 模型生成的待执行 Python 代码。
        has_code (bool): 标识代码部分是否非空。由 __post_init__ 自动设置。
    """

    thought: str
    code: str
    has_code: bool = field(init=False)

    def __post_init__(self) -> None:
        """初始化后置处理，设置 has_code 状态。"""
        self.has_code = bool(self.code.strip())

    def is_empty(self) -> bool:
        """检查解析结果是否为空（即思考和代码均为空）。

        Returns:
            bool: 若全为空则返回 True，否则返回 False。
        """
        return not self.thought.strip() and not self.code.strip()
