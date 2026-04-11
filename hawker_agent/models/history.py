from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class CodeAgentHistoryList:
    """
    包装 LLM 对话历史，替换 run() 中的裸 list[dict] 操作。

    三项职责（原分散在 run() 主循环中）：
    1. 管理永久历史（add_assistant / add_user）
    2. 消息压缩（_compress，原 _compress_messages 逻辑）
    3. DOM 状态临时注入（inject_dom，原 _browser_state_holder 逻辑）
       —— DOM 内容仅出现在下次 to_prompt_messages() 返回中，不进入永久历史
    """

    _system_prompt: str = ""
    _messages: list[dict] = field(default_factory=list)
    _pending_dom: str | None = None
    _compression_threshold: int = 12_000
    _count_tokens_fn: Callable[[list[dict]], int] | None = None

    @classmethod
    def from_task(
        cls,
        task: str,
        system_prompt: str,
        compression_threshold: int = 12_000,
    ) -> CodeAgentHistoryList:
        inst = cls(
            _system_prompt=system_prompt,
            _compression_threshold=compression_threshold,
        )
        inst.add_user(task.strip())
        return inst

    # -- 写入 --

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def inject_dom(self, dom: str) -> None:
        """
        注入浏览器 DOM 状态。
        DOM 内容不进入永久历史，仅在下次 to_prompt_messages() 中临时追加。
        """
        self._pending_dom = dom

    # -- 读取（供 LLM 调用） --

    def to_prompt_messages(self) -> list[dict]:
        """
        返回完整消息列表供 LLM 调用：
        [system] + 压缩后历史 + （如有）临时 DOM 注入。
        调用后 pending_dom 自动清除。
        """
        system_msg = {"role": "system", "content": self._system_prompt}
        compressed = self._compress(self._messages)
        result = [system_msg] + compressed

        if self._pending_dom:
            result = result + [
                {"role": "user", "content": f"[Browser State]\n{self._pending_dom}"}
            ]
            self._pending_dom = None

        return result

    # -- 压缩（原 _compress_messages + _build_summary_message） --

    def _compress(self, messages: list[dict]) -> list[dict]:
        """
        当 _count_tokens_fn 可用时，调用 compressor 模块执行真正的压缩。
        未提供 tokenizer 时（测试、向后兼容），直接返回原消息。
        """
        if self._count_tokens_fn is None:
            return list(messages)
        from hawker_agent.agent.compressor import compress_messages

        return compress_messages(messages, self._compression_threshold, self._count_tokens_fn)

    def __len__(self) -> int:
        return len(self._messages)
