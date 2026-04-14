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
from collections.abc import Callable
from pathlib import Path
from typing import Any, TYPE_CHECKING

from hawker_agent.observability import emit_observation
from hawker_agent.tools.data_tools import (
    clean_items,
    ensure,
    normalize_items,
    parse_http_response,
    save_file,
    summarize_json,
)

if TYPE_CHECKING:
    from hawker_agent.models.state import CodeAgentState

logger = logging.getLogger(__name__)


class HawkerNamespace:
    """
    分层命名空间管理类。
    - system: 系统注入的工具和库 (对 LLM 可见，只读保护)
    - session: 跨步骤持久化变量 (对 LLM 可见，可序列化)
    - cell_local: 当前步骤的临时变量 (对 LLM 不可见，步骤结束后丢弃)
    """

    def __init__(self, system_dict: dict, run_dir: str):
        self.system = system_dict
        self.session: dict[str, Any] = {"run_dir": run_dir}
        self.cell_local: dict[str, Any] = {}
        
    @property
    def exec_view(self) -> ChainMap:
        """返回合并视图供 exec() 使用，优先级: Local > Session > System"""
        return ChainMap(self.cell_local, self.session, self.system)

    def commit(self):
        """执行成功后，将 cell_local 中符合条件的变量提升到 session"""
        added = []
        for k, v in self.cell_local.items():
            if self._should_persist(k, v):
                self.session[k] = v
                added.append(k)
        
        if added:
            logger.debug("Namespace commit: 提升变量 %s 到 session", added)
        self.cell_local.clear()

    def rollback(self):
        """执行失败时，清空临时变量"""
        self.cell_local.clear()

    def _should_persist(self, name: str, value: Any) -> bool:
        """
        基于协议的持久化判定
        规则：
        1. 系统保留字 -> 不持久化
        2. 下划线开头 (如 _tmp) -> 显式声明的临时变量，不持久化
        3. 单字符名 (如 i, j, x) -> 典型的循环/临时变量，不持久化
        4. 其余变量 -> 全部持久化
        """
        if name in self.system: return False
        if name.startswith("_"): return False
        if len(name) <= 1: return False
        if inspect.ismodule(value): return False
            
        return True

    def get_llm_view(self) -> dict:
        """返回 LLM 应该感知的变量视图 (即 session 层)"""
        return {k: v for k, v in self.session.items() if not k.startswith("_")}


def _bind_callable_to_state(state: CodeAgentState, fn: Callable) -> Callable:
    """为工具函数注入日志上下文。"""
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


class ClearableList(list):
    """包装原始列表，拦截 .clear() 以同步重置 ItemStore。"""

    def __init__(self, data: list, store: object):
        super().__init__(data)
        self._data = data
        self._store = store

    def clear(self) -> None:
        self._data.clear()
        if hasattr(self._store, "clear"):
            self._store.clear()  # type: ignore

    def __getitem__(self, i): return self._data[i]
    def __iter__(self): return iter(self._data)
    def __len__(self): return len(self._data)
    def __repr__(self): return repr(self._data)


def register_core_actions(
    registry: ToolRegistry,
    state: CodeAgentState,
    run_dir: str,
) -> None:
    """将核心辅助动作注册到工具注册表，使其在 System Prompt 中自动生成。"""
    # 提示：核心动作必须带文档字符串，以便 build_capabilities_list 提取
    
    fn_append = _make_append_items(state)
    fn_append.__doc__ = "保存数据。追加到 all_items（自动去重），这是保存数据的唯一方式。"
    registry.register(fn_append, name="append_items", category="数据保存")
    
    fn_checkpoint = _make_save_checkpoint(state, run_dir)
    fn_checkpoint.__doc__ = "将当前 all_items 进度保存到磁盘（防止任务意外中断丢失数据）。不要使用 result.json 作为文件名；正式结果会由系统自动写入 result.json。"
    registry.register(fn_checkpoint, name="save_checkpoint", category="数据保存")

    fn_observe = _make_observe()
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
    """构建只读的系统工具字典"""
    sys_dict: dict = {}

    # 注入外部工具（包括通过 register_core_actions 注入的核心动作）
    for name, fn in tools_dict.items():
        if name in ("browser_download", "download_file"):
            # 为这些涉及文件写入的工具自动注入 run_dir 参数
            # 注意：必须使用默认参数绑定 fn_to_call，否则闭包会捕获循环变量的最终值
            if inspect.iscoroutinefunction(inspect.unwrap(fn)):
                @functools.wraps(fn)
                async def wrapped_with_rundir(*args: object, fn_to_call=fn, **kwargs: object) -> object:
                    if "run_dir" not in kwargs:
                        kwargs["run_dir"] = run_dir
                    return await fn_to_call(*args, **kwargs)
                sys_dict[name] = _bind_callable_to_state(state, wrapped_with_rundir)
            else:
                @functools.wraps(fn)
                def wrapped_with_rundir_sync(*args: object, fn_to_call=fn, **kwargs: object) -> object:
                    if "run_dir" not in kwargs:
                        kwargs["run_dir"] = run_dir
                    return fn_to_call(*args, **kwargs)
                sys_dict[name] = _bind_callable_to_state(state, wrapped_with_rundir_sync)
        else:
            sys_dict[name] = _bind_callable_to_state(state, fn)

    # 注入数据辅助函数
    # 注意：这里注入是为了让代码能跑通，但由于它们没在 ToolRegistry 注册，所以不会出现在 Prompt 里
    sys_dict["clean_items"] = clean_items
    sys_dict["ensure"] = ensure
    sys_dict["parse_http_response"] = parse_http_response
    sys_dict["summarize_json"] = summarize_json
    sys_dict["normalize_items"] = normalize_items
    sys_dict["save_file"] = save_file
    
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


def _make_append_items(state: CodeAgentState) -> Callable:
    async def append_items(items: object) -> list[dict]:
        normalized = normalize_items(items)
        added, skipped = state.items.append(normalized)
        if added:
            state.mark_activity()
        sample = json.dumps(state.items.get_raw_list()[-1], ensure_ascii=False)[:120] if added else ""
        emit_observation(
            f"[append_items] +{added} -> total={len(state.items)}"
            + (f" | skipped={skipped}" if skipped else "")
            + (f" | 样本: {sample}" if sample else "")
        )
        return state.items.get_raw_list()
    return append_items


def _make_observe() -> Callable:
    def observe(message: object) -> None:
        emit_observation(str(message))
    return observe


def _make_save_checkpoint(state: CodeAgentState, run_dir: str) -> Callable:
    async def save_checkpoint(filename: str = "checkpoint.json") -> str:
        target = filename or "checkpoint.json"
        if Path(target).name == "result.json":
            target = "checkpoint.json"
            emit_observation("[save_checkpoint] 文件名 result.json 保留给最终结果，已自动改存为 checkpoint.json")
        state.checkpoint_files.add(target)
        result = save_file(
            json.dumps(state.items.get_raw_list(), ensure_ascii=False), target, run_dir
        )
        emit_observation(f"[save_checkpoint] total={len(state.items)} | file={target}")
        return result
    return save_checkpoint


def _make_final_answer(state: CodeAgentState) -> Callable:
    async def final_answer(answer: object) -> None:
        state.final_answer_requested = str(answer)
        emit_observation(f"[final_answer] {answer}")
    return final_answer
