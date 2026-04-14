from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal
from typing import Any


@dataclass
class MemoryNote:
    """里程碑/经验教训的高密度长期记忆条目。"""

    step_start: int
    step_end: int
    category: str
    summary: str

    def render(self) -> str:
        span = (
            f"Step {self.step_start}"
            if self.step_start == self.step_end
            else f"Step {self.step_start}-{self.step_end}"
        )
        return f"- {span} [{self.category}] {self.summary}"


@dataclass
class DOMWorkspaceEntry:
    """DOM Workspace 的可衰减上下文条目。"""

    mode: Literal["summary", "diff", "full"]
    content: str
    folded_content: str
    ttl: int | None = None

    def render(self) -> str:
        ttl_text = "persistent" if self.ttl is None else str(self.ttl)
        return (
            f"[mode={self.mode} | ttl={ttl_text}]\n"
            f"{self.content}"
        )

    def advance(self) -> None:
        """在一次 prompt 使用后推进生命周期，必要时自动折叠。"""
        if self.ttl is None:
            return
        if self.ttl > 0:
            self.ttl -= 1
        if self.ttl == 0:
            self.mode = "summary"
            self.content = self.folded_content
            self.ttl = None


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
    _task: str = ""
    _notebook_mode_enabled: bool = False
    _recent_message_window: int = 8
    _max_milestones: int = 8
    _max_lessons: int = 8
    _runtime_snapshot: str = ""
    _namespace_snapshot: str = ""
    _dom_workspace: DOMWorkspaceEntry | None = None
    _milestones: list[MemoryNote] = field(default_factory=list)
    _lessons: list[MemoryNote] = field(default_factory=list)

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
            _task=task.strip(),
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
        Notebook 模式下会并入 DOM Workspace；普通模式下仍以临时消息注入。
        """
        self._pending_dom = dom

    def inject_browser_context(
        self,
        content: str,
        *,
        mode: Literal["summary", "diff", "full"] = "full",
        folded_content: str | None = None,
    ) -> None:
        """
        注入带生命周期控制的浏览器上下文。

        参数:
            content (str): 当前要展示的浏览器上下文。
            mode (Literal["summary", "diff", "full"]): 上下文模式。
            folded_content (str | None): 上下文过期后的折叠摘要。

        返回:
            None: 更新内部 DOM Workspace 或待注入上下文。
        """
        folded = folded_content or content
        if not self._notebook_mode_enabled:
            self._pending_dom = content
            return

        ttl_map: dict[str, int | None] = {
            "summary": None,
            "diff": 2,
            "full": 1,
        }
        self._dom_workspace = DOMWorkspaceEntry(
            mode=mode,
            content=content,
            folded_content=folded,
            ttl=ttl_map[mode],
        )

    # -- 读取（供 LLM 调用） --

    def to_prompt_messages(self) -> list[dict]:
        """
        返回完整消息列表供 LLM 调用：
        [system] + 压缩后历史 + （如有）临时 DOM 注入。
        调用后 pending_dom 自动清除。
        """
        return self.build_prompt_package()["messages"]

    def build_prompt_package(self) -> dict[str, Any]:
        """
        构建本次发送给模型的 prompt 包，并保留压缩前后的拆分信息。

        参数:
            无

        返回:
            dict[str, Any]: 包含最终 messages、拆分区块和被折叠部分的调试信息。
        """
        system_msg = {"role": "system", "content": self._system_prompt}
        if self._notebook_mode_enabled and self._pending_dom:
            self.inject_browser_context(self._pending_dom, mode="full")
            self._pending_dom = None

        if self._notebook_mode_enabled:
            notebook_package = self._build_notebook_messages()
            result = [system_msg] + notebook_package["messages"]
            package = {
                "mode": "notebook",
                "system_message": system_msg,
                "task_message": notebook_package["task_message"],
                "workspace_message": notebook_package["workspace_message"],
                "recent_messages": notebook_package["recent_messages"],
                "omitted_messages": notebook_package["omitted_messages"],
                "messages": result,
                "source_history_messages": list(self._messages),
                "workspace_sections": {
                    "runtime_snapshot": self._runtime_snapshot or "尚未执行任何步骤。",
                    "milestones": [note.render() for note in self._milestones],
                    "long_term_memory": [note.render() for note in self._lessons],
                    "namespace_snapshot": self._namespace_snapshot or "无持久化变量。",
                    "dom_workspace": self._dom_workspace.render() if self._dom_workspace else "暂无页面增量上下文。",
                },
            }
        else:
            compressed = self._compress(self._messages)
            result = [system_msg] + compressed
            package = {
                "mode": "compressed_history",
                "system_message": system_msg,
                "messages": result,
                "source_history_messages": list(self._messages),
                "compressed_history_messages": compressed,
            }

        if self._pending_dom:
            browser_state_msg = {"role": "user", "content": f"[Browser State]\n{self._pending_dom}"}
            result = result + [browser_state_msg]
            package["ephemeral_browser_state"] = browser_state_msg
            package["messages"] = result
            self._pending_dom = None

        if self._notebook_mode_enabled and self._dom_workspace is not None:
            self._dom_workspace.advance()

        return package

    def record_step(
        self,
        *,
        step: int,
        max_steps: int,
        assistant_content: str,
        observation: str,
        namespace_view: dict[str, Any],
        items_count: int,
        total_tokens: int,
        max_total_tokens: int,
        progress: bool,
        had_error: bool,
        no_progress_steps: int,
    ) -> None:
        """
        进入 Notebook 状态流模式。
        保留最近少量原始对话作为 STM，同时把历史提纯为里程碑和经验教训。
        """
        self._notebook_mode_enabled = True
        self.add_assistant(assistant_content)
        from hawker_agent.agent.compressor import truncate_output, build_namespace_snapshot

        status_line = (
            f"[RuntimeStatus] 已采集: {items_count}条"
            f" | 步骤: {step}/{max_steps}"
            f" | token: {total_tokens:,}/{max_total_tokens:,}"
        )
        obs_for_history = truncate_output(observation, 1500)
        self.add_user(f"{status_line}\n\nObservation:\n{obs_for_history}")

        self._runtime_snapshot = (
            f"已采集 {items_count} 条 | 步骤 {step}/{max_steps}"
            f" | token {total_tokens:,}/{max_total_tokens:,}"
            f" | 连续无进展 {no_progress_steps} 步"
        )
        self._namespace_snapshot = build_namespace_snapshot(namespace_view)
        self._update_long_term_memory(
            step=step,
            assistant_content=assistant_content,
            observation=observation,
            progress=progress,
            had_error=had_error,
            no_progress_steps=no_progress_steps,
        )

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

    def _build_notebook_messages(self) -> dict[str, Any]:
        task_msg = self._messages[:1] or [{"role": "user", "content": self._task}]
        recent = self._messages[1:]
        if len(recent) > self._recent_message_window:
            recent = recent[-self._recent_message_window:]

        workspace_msg = {"role": "user", "content": self._build_workspace_context()}
        result = list(task_msg) + [workspace_msg] + recent
        fitted, omitted = self._fit_notebook_messages(result)
        return {
            "messages": fitted,
            "task_message": task_msg[0],
            "workspace_message": workspace_msg,
            "recent_messages": recent,
            "omitted_messages": omitted,
        }

    def _build_workspace_context(self) -> str:
        milestone_lines = [note.render() for note in self._milestones] or ["- 暂无已确认里程碑"]
        lesson_lines = [note.render() for note in self._lessons] or ["- 暂无失败经验"]
        milestones_text = "\n".join(milestone_lines)
        lessons_text = "\n".join(lesson_lines)
        dom_workspace_text = self._dom_workspace.render() if self._dom_workspace else "暂无页面增量上下文。"
        return (
            "[Notebook Workspace]\n"
            "下面是已经压缩过的长期状态，不要回头重放旧步骤。\n\n"
            "[Runtime Snapshot]\n"
            f"{self._runtime_snapshot or '尚未执行任何步骤。'}\n\n"
            "[Milestones]\n"
            f"{milestones_text}\n\n"
            "[Long-Term Memory]\n"
            f"{lessons_text}\n\n"
            "[Namespace Snapshot]\n"
            f"{self._namespace_snapshot or '无持久化变量。'}\n\n"
            "[DOM Workspace]\n"
            f"{dom_workspace_text}\n\n"
            "[STM Policy]\n"
            "后续原始消息只保留最近少量步骤用于调试；更早历史已经提纯进上面的里程碑和经验教训。"
        )

    def _fit_notebook_messages(self, messages: list[dict]) -> tuple[list[dict], list[dict]]:
        if self._count_tokens_fn is None:
            return messages, []
        fitted = list(messages)
        omitted: list[dict] = []
        while len(fitted) > 3 and self._count_tokens_fn(fitted) > self._compression_threshold:
            if len(fitted) <= 4:
                break
            omitted.append(fitted.pop(2))
        return fitted, omitted

    def _update_long_term_memory(
        self,
        *,
        step: int,
        assistant_content: str,
        observation: str,
        progress: bool,
        had_error: bool,
        no_progress_steps: int,
    ) -> None:
        from hawker_agent.agent.compressor import format_preview, semantic_observation_preview
        from hawker_agent.agent.parser import parse_response

        parsed = parse_response(assistant_content)
        thought = format_preview(parsed.thought or "未提供分析", 120)
        observation_summary = semantic_observation_preview(observation, 260)
        category = self._categorize_note(f"{parsed.thought}\n{parsed.code}\n{observation}")

        if progress:
            milestone = observation_summary if observation_summary != "[无输出]" else thought
            self._append_note(
                self._milestones,
                MemoryNote(step_start=step, step_end=step, category=category, summary=milestone),
                self._max_milestones,
            )

        if had_error:
            lesson = f"{thought} -> {observation_summary}"
            self._append_note(
                self._lessons,
                MemoryNote(step_start=step, step_end=step, category=category, summary=lesson),
                self._max_lessons,
            )
            return

        if no_progress_steps >= 2:
            lesson = f"{thought} -> 连续无进展，最近 Observation: {observation_summary}"
            self._append_note(
                self._lessons,
                MemoryNote(step_start=step, step_end=step, category=category, summary=lesson),
                self._max_lessons,
            )

    @staticmethod
    def _categorize_note(text: str) -> str:
        lowered = text.lower()
        keyword_groups = {
            "login": ("login", "cookie", "captcha", "验证码", "登录"),
            "api": ("api", "http", "network", "xhr", "fetch", "接口"),
            "extract": ("append_items", "提取", "采集", "schema", "样本", "json"),
            "navigation": ("nav", "click", "tab", "page", "列表页", "跳转", "翻页"),
            "download": ("download", "checkpoint", "保存", "文件"),
            "error": ("error", "exception", "traceback", "失败", "超时", "403", "选择器"),
        }
        for label, keywords in keyword_groups.items():
            if any(keyword in lowered for keyword in keywords):
                return label
        return "general"

    @staticmethod
    def _append_note(notes: list[MemoryNote], note: MemoryNote, max_size: int) -> None:
        if notes and notes[-1].category == note.category and note.step_start <= notes[-1].step_end + 2:
            notes[-1].step_end = note.step_end
            notes[-1].summary = note.summary
        else:
            notes.append(note)
        if len(notes) > max_size:
            del notes[:-max_size]

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value
