from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal
from typing import Any


@dataclass
class MemoryNote:
    """里程碑/经验教训的高密度长期记忆条目。

    Attributes:
        step_start (int): 该记录覆盖的起始步骤序号。
        step_end (int): 该记录覆盖的结束步骤序号。
        category (str): 记录所属分类（如 login, api, extract 等）。
        summary (str): 核心内容简要说明。
    """

    step_start: int
    step_end: int
    category: str
    summary: str

    def render(self) -> str:
        """渲染单条记忆条目为文本。

        Returns:
            str: 格式化后的字符串。
        """
        span = (
            f"Step {self.step_start}"
            if self.step_start == self.step_end
            else f"Step {self.step_start}-{self.step_end}"
        )
        return f"- {span} [{self.category}] {self.summary}"


@dataclass
class DOMWorkspaceEntry:
    """DOM Workspace 的可衰减上下文条目。

    该类用于管理模型接收到的 DOM 视图及其生命周期。

    Attributes:
        mode (Literal["summary", "diff", "full"]): 当前展示模式。
        content (str): 当前要呈现的文本内容。
        folded_content (str): 内容过期/折叠后要显示的摘要。
        ttl (int | None): 生命周期（步数计数），None 表示永久保留。
    """

    mode: Literal["summary", "diff", "full"]
    content: str
    folded_content: str
    ttl: int | None = None

    def render(self) -> str:
        """渲染当前上下文条目为文本。

        Returns:
            str: 格式化后的字符串。
        """
        ttl_text = "persistent" if self.ttl is None else str(self.ttl)
        return (
            f"[mode={self.mode} | ttl={ttl_text}]\n"
            f"{self.content}"
        )

    def advance(self) -> None:
        """在一次 prompt 使用后推进生命周期。

        当 ttl 减至 0 时，自动将内容折叠为 summary 模式。
        """
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
    """代理对话历史管理器。

    包装 LLM 对话历史，负责管理永久历史、消息压缩以及 DOM 状态的临时/工作区注入。

    Attributes:
        _system_prompt (str): 系统提示词内容。
        _messages (list[dict]): 基础对话历史。
        _pending_dom (str | None): 待在下一次对话中注入的 DOM 状态文本。
        _compression_threshold (int): 触发历史压缩的 token 阈值。
        _count_tokens_fn (Callable[[list[dict]], int] | None): 用于计算 token 数量的函数。
        _task (str): 原始任务描述。
        _notebook_mode_enabled (bool): 是否开启 Notebook 模式。
        _recent_message_window (int): 在 Notebook 模式中保留的最近原始消息数量。
        _max_milestones (int): 最大里程碑存储数量。
        _max_lessons (int): 最大经验教训存储数量。
        _runtime_snapshot (str): 当前运行时快照文本。
        _namespace_snapshot (str): 命名空间快照文本。
        _memory_workspace (list[str]): 召回到的站点经验和策略记忆。
        _dom_workspace (DOMWorkspaceEntry | None): 增量 DOM 工作区。
        _milestones (list[MemoryNote]): 存储的里程碑。
        _lessons (list[MemoryNote]): 存储的经验教训。
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
    _memory_workspace: list[str] = field(default_factory=list)
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
        """通过初始任务构建历史记录对象。

        Args:
            task (str): 代理要执行的任务描述。
            system_prompt (str): 系统初始指令。
            compression_threshold (int, optional): 压缩阈值。 Defaults to 12_000.

        Returns:
            CodeAgentHistoryList: 初始化完成的历史记录管理器。
        """
        inst = cls(
            _system_prompt=system_prompt,
            _compression_threshold=compression_threshold,
            _task=task.strip(),
        )
        inst.add_user(task.strip())
        return inst

    # -- 写入 --

    def add_assistant(self, content: str) -> None:
        """添加助手的回复消息。

        Args:
            content (str): 助手回复的正文。
        """
        self._messages.append({"role": "assistant", "content": content})

    def add_user(self, content: str) -> None:
        """添加用户消息。

        Args:
            content (str): 用户输入或反馈的正文。
        """
        self._messages.append({"role": "user", "content": content})

    def inject_dom(self, dom: str) -> None:
        """注入待下发的浏览器 DOM 状态信息。

        在普通模式下，这将作为一次性的临时消息注入；在 Notebook 模式下，它可能被并入 DOM 工作区。

        Args:
            dom (str): 捕获到的 DOM 视图文本。
        """
        self._pending_dom = dom

    def inject_browser_context(
        self,
        content: str,
        *,
        mode: Literal["summary", "diff", "full"] = "full",
        folded_content: str | None = None,
    ) -> None:
        """在 Notebook 模式中注入带生命周期控制的浏览器上下文。

        Args:
            content (str): 要注入的当前浏览器上下文。
            mode (Literal["summary", "diff", "full"], optional): 上下文模式。默认为 "full"。
            folded_content (str | None, optional): 超过 TTL 后使用的折叠文本。
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

    def set_memory_workspace(self, entries: list[str]) -> None:
        """设置当前任务的 Memory Workspace 条目。"""
        cleaned: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            text = entry.strip()
            if not text or text in seen:
                continue
            cleaned.append(text)
            seen.add(text)
        self._memory_workspace = cleaned[:6]

    # -- 读取（供 LLM 调用） --

    def to_prompt_messages(self) -> list[dict]:
        """构建供模型调用的完整消息列表。

        合并系统消息、已压缩的历史消息以及当前的临时 DOM 状态。调用该方法会清空 pending_dom。

        Returns:
            list[dict]: 符合 OpenAI/LiteLLM 规范的消息列表。
        """
        return self.build_prompt_package()["messages"]

    def build_prompt_package(self) -> dict[str, Any]:
        """构建本次发送给模型的完整数据包。

        该数据包包含最终的 messages 以及用于调试的内部拆分区块信息。

        Returns:
            dict[str, Any]: 包含 "messages" 列表及各种辅助调试信息的字典。
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
                    "memory_workspace": list(self._memory_workspace),
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
        """将当前步骤结果记录到历史中，并视情况推进 Notebook 状态。

        此方法将对话历史转化为提纯后的里程碑、经验教训，并更新运行时快照。

        Args:
            step (int): 当前步骤序号。
            max_steps (int): 总最大步数限制。
            assistant_content (str): 助手该步骤生成的思考与代码。
            observation (str): 代码执行后的观察结果。
            namespace_view (dict[str, Any]): 当前代码命名空间视图。
            items_count (int): 当前累计采集到的项目数。
            total_tokens (int): 当前累计消耗的 token 数。
            max_total_tokens (int): 最大 token 额度限制。
            progress (bool): 本步骤是否取得实质性进展。
            had_error (bool): 本步骤是否执行报错。
            no_progress_steps (int): 当前连续无进展的步数计数。
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
        """执行消息压缩。

        当 token 计数函数可用时，根据压缩阈值对历史消息执行压缩逻辑。

        Args:
            messages (list[dict]): 待压缩的消息列表。

        Returns:
            list[dict]: 压缩后的消息列表。
        """
        if self._count_tokens_fn is None:
            return list(messages)
        from hawker_agent.agent.compressor import compress_messages

        return compress_messages(messages, self._compression_threshold, self._count_tokens_fn)

    def _build_notebook_messages(self) -> dict[str, Any]:
        """构建 Notebook 模式下的消息分块信息。

        Returns:
            dict[str, Any]: 包含任务、工作区、最近消息及忽略消息的数据字典。
        """
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
        """构建工作区上下文的正文文本。

        Returns:
            str: 格式化后的工作区内容。
        """
        milestone_lines = [note.render() for note in self._milestones] or ["- 暂无已确认里程碑"]
        lesson_lines = [note.render() for note in self._lessons] or ["- 暂无失败经验"]
        milestones_text = "\n".join(milestone_lines)
        lessons_text = "\n".join(lesson_lines)
        memory_lines = self._memory_workspace or ["- 暂无站点经验记忆"]
        memory_text = "\n".join(memory_lines)
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
            "[Memory Workspace]\n"
            f"{memory_text}\n\n"
            "[DOM Workspace]\n"
            f"{dom_workspace_text}\n\n"
            "[STM Policy]\n"
            "后续原始消息只保留最近少量步骤用于调试；更早历史已经提纯进上面的里程碑 and 经验教训。"
        )

    def _fit_notebook_messages(self, messages: list[dict]) -> tuple[list[dict], list[dict]]:
        """确保 Notebook 消息不超长，必要时剔除较旧的消息。

        Args:
            messages (list[dict]): 待检查的消息列表。

        Returns:
            tuple[list[dict], list[dict]]: (保留的消息列表, 被剔除的消息列表)。
        """
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
        """基于当前步骤结果更新长期记忆（里程碑和经验教训）。

        Args:
            step (int): 当前步骤。
            assistant_content (str): 助手指令内容。
            observation (str): 观察输出内容。
            progress (bool): 是否有进展。
            had_error (bool): 是否报错。
            no_progress_steps (int): 连续无进展步数。
        """
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
        """根据文本内容对记忆条目进行分类。

        Args:
            text (str): 原始文本内容。

        Returns:
            str: 识别出的类别标签。
        """
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
        """向记忆列表中追加新条目，处理相邻相似类别的合并，并维护最大列表容量。

        Args:
            notes (list[MemoryNote]): 目标记忆列表。
            note (MemoryNote): 待追加的新记忆条目。
            max_size (int): 列表允许的最大容量。
        """
        if notes and notes[-1].category == note.category and note.step_start <= notes[-1].step_end + 2:
            notes[-1].step_end = note.step_end
            notes[-1].summary = note.summary
        else:
            notes.append(note)
        if len(notes) > max_size:
            del notes[:-max_size]

    def export_memory_notes(self) -> dict[str, list[MemoryNote]]:
        """导出当前任务沉淀出的里程碑和经验教训。"""
        return {
            "milestones": list(self._milestones),
            "lessons": list(self._lessons),
        }

    def __len__(self) -> int:
        """获取永久消息列表的长度。

        Returns:
            int: 消息总数。
        """
        return len(self._messages)

    @property
    def system_prompt(self) -> str:
        """获取当前的系统提示词。

        Returns:
            str: 系统提示词文本。
        """
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """设置新的系统提示词。

        Args:
            value (str): 新的提示词文本。
        """
        self._system_prompt = value
