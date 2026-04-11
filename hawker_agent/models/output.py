from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CodeAgentModelOutput:
    """
    LLM 响应的语义解析结果，替换 _parse_response 的 tuple[str, str] 返回值。

    层次区分：
    - LLMResponse (llm/client.py) = 原始 LLM 输出（text + usage）
    - CodeAgentModelOutput (本类) = Agent 层语义解析（thought + code）
    """

    thought: str
    code: str
    has_code: bool = field(init=False)

    def __post_init__(self) -> None:
        self.has_code = bool(self.code.strip())

    def is_empty(self) -> bool:
        return not self.thought.strip() and not self.code.strip()
