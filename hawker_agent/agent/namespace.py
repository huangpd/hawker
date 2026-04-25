from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import inspect
import json
import logging
import re
import time
import urllib
import urllib.parse
import urllib.request
from collections import ChainMap
from collections.abc import MutableSequence
from collections.abc import Callable
from pathlib import Path
from typing import Any, TYPE_CHECKING

from hawker_agent.observability import emit_observation
from hawker_agent.agent.artifact import artifact_to_answer_text, normalize_final_artifact
from hawker_agent.tools.data_tools import (
    analyze_json_structure,
    clean_items,
    ensure,
    normalize_items,
    parse_http_response,
    save_file,
    summarize_json,
)

if TYPE_CHECKING:
    from hawker_agent.models.state import CodeAgentState
    from hawker_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class HawkerNamespace:
    """代码代理的分层命名空间管理。

    管理三个层级的变量：
    1. system: 系统注入的工具和库（对 LLM 只读）。
    2. session: 跨步骤的持久变量（对 LLM 可见，可序列化）。
    3. cell_local: 当前步骤的临时变量（对 LLM 隐藏）。

    Attributes:
        system (dict): 只读的系统工具和库。
        session (dict[str, Any]): 持久化的会话级变量。
        cell_local (dict[str, Any]): 临时的步骤级变量。
    """

    def __init__(self, system_dict: dict, run_dir: str):
        """初始化 HawkerNamespace。

        Args:
            system_dict (dict): 系统工具和库的字典。
            run_dir (str): 当前运行的目录路径。
        """
        self.system = system_dict
        self.session: dict[str, Any] = {"run_dir": run_dir}
        self.cell_local: dict[str, Any] = {}
        
    @property
    def exec_view(self) -> ChainMap:
        """返回用于 exec() 的合并视图。

        优先级顺序：cell_local > session > system。

        Returns:
            ChainMap: 合并后的变量视图。
        """
        return ChainMap(self.cell_local, self.session, self.system)

    def commit(self):
        """将符合条件的变量从 cell_local 提升到 session。

        在步骤执行成功后调用。满足持久化条件的临时变量会被移动到会话层。
        """
        added = []
        for k, v in self.cell_local.items():
            if self._should_persist(k, v):
                self.session[k] = v
                added.append(k)
        
        if added:
            logger.debug("Namespace commit: 提升变量 %s 到 session", added)
        self.cell_local.clear()

    def rollback(self):
        """清除临时变量。

        在步骤执行失败后调用，以重置本地状态。
        """
        self.cell_local.clear()

    def _should_persist(self, name: str, value: Any) -> bool:
        """确定一个变量是否应该持久化到会话中。

        持久化规则：
        1. 不是系统保留字。
        2. 不以下划线开头（例如 _tmp）。
        3. 变量名长度大于 1（排除 i, j, x 等）。
        4. 值不是模块。

        Args:
            name (str): 变量名称。
            value (Any): 变量值。

        Returns:
            bool: 如果变量应该持久化则返回 True，否则返回 False。
        """
        if name in self.system:
            return False
        if name.startswith("_"):
            return False
        if len(name) <= 1:
            return False
        if inspect.ismodule(value):
            return False
        return True

    def get_llm_view(self) -> dict:
        """返回对 LLM 可见的变量视图。

        仅包含非内部的会话变量。

        Returns:
            dict: 对 LLM 可见的变量字典。
        """
        return {k: v for k, v in self.session.items() if not k.startswith("_")}


def _bind_callable_to_state(state: CodeAgentState, fn: Callable) -> Callable:
    """为工具函数注入日志上下文并处理浏览器结果。

    Args:
        state (CodeAgentState): 当前代理状态。
        fn (Callable): 要绑定的函数。

    Returns:
        Callable: 包装后的函数。
    """
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapped(*args: object, **kwargs: object) -> object:
            with state.bind_log_context():
                result = await fn(*args, **kwargs)
                if hasattr(result, "dom") and hasattr(result, "summary"):
                    state.pending_dom = getattr(result, "dom")
                    return getattr(result, "summary")
                return result
        return async_wrapped

    @functools.wraps(fn)
    def wrapped(*args: object, **kwargs: object) -> object:
        with state.bind_log_context():
            result = fn(*args, **kwargs)
            if hasattr(result, "dom") and hasattr(result, "summary"):
                state.pending_dom = getattr(result, "dom")
                return getattr(result, "summary")
            return result
    return wrapped


def _supports_kwarg(fn: Callable, arg_name: str) -> bool:
    """判断函数是否显式声明某个关键字参数，或接受任意 kwargs。"""
    try:
        sig = inspect.signature(inspect.unwrap(fn))
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == arg_name:
            return True
    return False


class ClearableList(MutableSequence):
    """同步列表写操作到 ItemStore 的可变序列包装器。

    这里不能直接继承 ``list`` 保存一份副本，否则 ``append`` / ``extend`` 等写操作会
    落到父类副本上，导致运行时看似成功但不会真正写回底层 ItemStore。
    """

    def __init__(self, data: list, store: object):
        """初始化 ClearableList。

        Args:
            data (list): 底层列表数据。
            store (object): 要同步的 ItemStore 对象。
        """
        self._data = data
        self._store = store

    def append(self, value: object) -> None:
        """向底层列表追加一项。"""
        self._data.append(value)

    def extend(self, values: list[object]) -> None:
        """向底层列表追加多项。"""
        self._data.extend(values)

    def clear(self) -> None:
        """同时清除列表和同步的 ItemStore。"""
        self._data.clear()
        if hasattr(self._store, "clear"):
            self._store.clear()  # type: ignore

    def insert(self, index: int, value: object) -> None:
        """在指定位置插入一项。"""
        self._data.insert(index, value)

    def __getitem__(self, index: int | slice):
        return self._data[index]

    def __setitem__(self, index: int | slice, value: object) -> None:
        self._data[index] = value

    def __delitem__(self, index: int | slice) -> None:
        del self._data[index]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)


def register_core_actions(
    registry: ToolRegistry,
    state: CodeAgentState,
    run_dir: str,
) -> None:
    """将核心辅助操作注册到工具注册表中。

    这些操作会自动包含在系统 Prompt 的功能列表中。

    Args:
        registry (ToolRegistry): 要添加操作的注册表。
        state (CodeAgentState): 当前代理状态。
        run_dir (str): 当前运行的目录。
    """
    # 提示：核心动作必须带文档字符串，以便 build_capabilities_list 提取

    fn_append = _make_append_items(state, run_dir)
    fn_append.__doc__ = "保存数据。追加到 all_items（自动去重），这是保存数据的唯一方式。"
    registry.register(fn_append, name="append_items", category="数据保存")

    fn_checkpoint = _make_save_checkpoint(state, run_dir)
    fn_checkpoint.__doc__ = "将当前 all_items 进度保存到磁盘（防止任务意外中断丢失数据）。不要使用 result.json 作为文件名；该文件名保留给最终结果。"
    registry.register(fn_checkpoint, name="save_checkpoint", category="数据保存")

    fn_verify = _make_verify_downloads(state, run_dir)
    fn_verify.__doc__ = "核对磁盘上的文件下载状态。在确认交付前调用，确保所有 download.file 都真实存在且不为空。"
    registry.register(fn_verify, name="verify_downloads", category="数据保存")

    fn_observe = _make_observe(state)

    fn_observe.__doc__ = "向大模型写入本步 Observation。只用它反馈数量、样本、状态；不要用 print 作为正式观察通道。"
    registry.register(fn_observe, name="observe", category="其他工具")

    fn_final = _make_final_answer(state)
    fn_final.__doc__ = "提交最终答案并强制结束任务。必须在确认完成时单独调用一次。"
    registry.register(fn_final, name="final_answer", category="数据保存")


def build_namespace(
    state: CodeAgentState,
    tools_dict: dict[str, Callable],
    run_dir: str,
) -> dict:
    """构建只读的系统工具字典。

    Args:
        state (CodeAgentState): 当前代理状态。
        tools_dict (dict[str, Callable]): 外部工具字典。
        run_dir (str): 当前运行的目录。

    Returns:
        dict: 为执行准备好的系统工具字典。
    """
    sys_dict: dict = {}

    # 注入外部工具（包括通过 register_core_actions 注入的核心动作）
    for name, fn in tools_dict.items():
        if name in {"browser_download", "obs_stream_download"}:
            # 为涉及下载的工具自动注入运行上下文并记录证据
            should_inject_run_dir = _supports_kwarg(fn, "run_dir")
            supports_ref = _supports_kwarg(fn, "ref")
            supports_entity_key = _supports_kwarg(fn, "entity_key")
            if inspect.iscoroutinefunction(inspect.unwrap(fn)):
                @functools.wraps(fn)
                async def wrapped_download_tool(
                    *args: object,
                    fn_to_call=fn,
                    name_of_tool=name,
                    inject_run_dir=should_inject_run_dir,
                    inject_ref=supports_ref,
                    inject_entity_key=supports_entity_key,
                    **kwargs: object,
                ) -> object:
                    if inject_run_dir and "run_dir" not in kwargs:
                        kwargs["run_dir"] = run_dir
                    call_kwargs = dict(kwargs)
                    if not inject_ref:
                        call_kwargs.pop("ref", None)
                    if not inject_entity_key:
                        call_kwargs.pop("entity_key", None)
                    result = await fn_to_call(*args, **call_kwargs)
                    if isinstance(result, dict) and result.get("ok"):
                        # 转换结果为证据项
                        if name_of_tool == "browser_download":
                            evidence = _build_download_evidence_item(result, run_dir)
                        else:
                            evidence = result
                        explicit_entity_key = kwargs.get("entity_key") or result.get("entity_key")
                        explicit_ref = kwargs.get("ref") or result.get("ref")
                        if isinstance(evidence, dict):
                            if isinstance(explicit_entity_key, str) and explicit_entity_key.strip():
                                evidence["entity_key"] = explicit_entity_key.strip()
                            elif isinstance(explicit_ref, str) and explicit_ref.strip():
                                evidence["ref"] = explicit_ref.strip()
                        if evidence:
                            changed, _ = state.items.append(normalize_items([evidence]))
                            if changed:
                                state.mark_activity()
                    return result
                sys_dict[name] = _bind_callable_to_state(state, wrapped_download_tool)
            else:
                # 同步下载工具（如果有）
                sys_dict[name] = _bind_callable_to_state(state, fn)
        else:
            sys_dict[name] = _bind_callable_to_state(state, fn)

    # 注入数据辅助函数
    # 注意：这里注入是为了让代码能跑通，但由于它们没在 ToolRegistry 注册，所以不会出现在 Prompt 里
    sys_dict["sys_clean_items"] = clean_items
    sys_dict["ensure"] = ensure
    sys_dict["sys_parse_http_response"] = parse_http_response
    sys_dict["sys_summarize_json"] = summarize_json
    sys_dict["sys_analyze_json_structure"] = analyze_json_structure
    sys_dict["sys_normalize_items"] = normalize_items
    sys_dict["sys_save_file"] = save_file
    
    # 共享数据引用 (使用 ClearableList 包装)
    sys_dict["all_items"] = ClearableList(state.items.get_raw_list(), state.items)

    # 标准库
    sys_dict["asyncio"] = asyncio
    sys_dict["json"] = json
    sys_dict["re"] = re
    sys_dict["time"] = time
    sys_dict["base64"] = base64
    sys_dict["hashlib"] = hashlib
    sys_dict["urllib"] = urllib
    sys_dict["Path"] = Path

    return sys_dict


def build_system_dict(
    state: CodeAgentState,
    tools_dict: dict[str, Callable],
    run_dir: str,
) -> dict:
    """兼容旧测试/调用方的系统命名空间构建入口。"""
    registry_dict = dict(tools_dict)
    from hawker_agent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    for name, fn in registry_dict.items():
        registry.register(fn, name=name, expose_in_prompt=False)
    register_core_actions(registry, state, run_dir)
    return build_namespace(state, registry.as_namespace_dict(), run_dir)

def _record_observation(state: CodeAgentState, message: str) -> None:
    """Emit and retain a recent observation for later evaluation."""
    emit_observation(message)
    state.remember_observation(message)


def _project_public_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hide internal identity metadata from model-facing append_items output."""
    projected: list[dict[str, Any]] = []
    for item in items:
        row = dict(item)
        row.pop("entity_key", None)
        projected.append(row)
    return projected

def _build_download_evidence_item(result: dict[str, object], run_dir: str) -> dict[str, object] | None:
    """将 browser_download 的成功结果转换为带 download 嵌套键的证据项。

    技术元数据封装在 download 键内；实体身份应由上游显式通过 ref/entity_key 透传。
    """
    url = str(result.get("url") or "").strip()
    path_text = str(result.get("path") or "").strip()
    filename = str(result.get("filename") or "").strip()
    if not url or not path_text:
        return None

    path = Path(path_text)
    base = Path(run_dir)
    try:
        downloaded_file = str(path.relative_to(base))
    except ValueError:
        downloaded_file = path.name or filename

    return {
        "download_url": url,
        "download": {
            "status": "success",
            "file": downloaded_file,
            "path": str(path),
            "size": int(result.get("size") or 0),
        }
    }
def _make_append_items(state: CodeAgentState, run_dir: str) -> Callable:
    """创建 append_items 工具函数。

    Args:
        state (CodeAgentState): 当前代理状态。

    Returns:
        Callable: append_items 函数。
    """
    async def append_items(items: object) -> list[dict]:
        normalized = normalize_items(items)
        changed, unchanged = state.items.append(normalized)
        if changed:
            state.mark_activity()
        last_changed = state.items.get_last_changed()
        if isinstance(last_changed, dict):
            last_changed = _project_public_items([last_changed])[0]
        sample = json.dumps(last_changed, ensure_ascii=False)[:120] if last_changed else ""
        _record_observation(
            state,
            f"[append_items] changed={changed} -> total={len(state.items)}"
            + (f" | unchanged={unchanged}" if unchanged else "")
            + (f" | 样本: {sample}" if sample else "")
        )
        return _project_public_items(state.items.get_raw_list())
    return append_items


def _make_observe(state: CodeAgentState) -> Callable:
    """创建 observe 工具函数。

    Returns:
        Callable: observe 函数。
    """
    def observe(message: object) -> None:
        _record_observation(state, str(message))
    return observe


def _make_save_checkpoint(state: CodeAgentState, run_dir: str) -> Callable:
    """创建 save_checkpoint 工具函数。

    Args:
        state (CodeAgentState): 当前代理状态。
        run_dir (str): 当前运行的目录。

    Returns:
        Callable: save_checkpoint 函数。
    """
    async def save_checkpoint(filename: str = "checkpoint.json") -> str:
        target = filename or "checkpoint.json"
        if Path(target).name == "result.json":
            target = "checkpoint.json"
            _record_observation(state, "[save_checkpoint] 文件名 result.json 保留给最终结果，已自动改存为 checkpoint.json")
        state.checkpoint_files.add(target)
        result = save_file(
            json.dumps(state.items.get_raw_list(), ensure_ascii=False), target, run_dir
        )
        _record_observation(state, f"[save_checkpoint] total={len(state.items)} | file={target}")
        return result
    return save_checkpoint


def _make_final_answer(state: CodeAgentState) -> Callable:
    """创建 final_answer 工具函数。

    Args:
        state (CodeAgentState): 当前代理状态。

    Returns:
        Callable: final_answer 函数。
    """
    async def final_answer(answer: object) -> None:
        artifact = normalize_final_artifact(
            answer,
            expected_output_format=state.expected_output_format,
        )
        answer_text = artifact_to_answer_text(artifact)
        state.final_artifact_requested = artifact
        state.final_answer_requested = answer_text
        emit_observation(f"[final_answer] {answer_text}")
    return final_answer


def _make_verify_downloads(state: CodeAgentState, run_dir: str) -> Callable:
    """创建 verify_downloads 工具函数。"""
    async def verify_downloads() -> dict[str, Any]:
        from hawker_agent.tools.data_tools import check_files_on_disk
        items = state.items.to_list()
        result = check_files_on_disk(run_dir, items)
        _record_observation(state, f"[verify_downloads] {result['summary']}")
        return result
    return verify_downloads
