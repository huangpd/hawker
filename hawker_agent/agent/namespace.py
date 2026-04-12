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
        
        # 预定义不应持久化的临时变量黑名单
        self._temp_var_names = {
            "i", "j", "k", "n", "x", "y", "el", "item", "node",
            "tmp", "temp", "row", "line", "result", "resp", "response",
            "data", "raw", "buf", "content", "m", "match"
        }

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
        """判定变量是否值得持久化"""
        # 1. 排除系统保留 key
        if name in self.system:
            return False
        # 2. 排除下划线开头的私有变量
        if name.startswith("_"):
            return False
        # 3. 排除已知的临时变量名
        if name.lower() in self._temp_var_names:
            return False
        # 4. 排除过短的变量名 (通常是循环索引)
        if len(name) <= 1:
            return False
        # 5. 排除模块、函数、类等不可序列化的结构 (可选，LLM 定义的函数建议保留)
        if inspect.ismodule(value):
            return False
            
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


def build_system_dict(
    state: CodeAgentState,
    tools_dict: dict[str, Callable],
    run_dir: str,
) -> dict:
    """构建只读的系统工具字典"""
    sys_dict: dict = {}

    # 注入外部工具
    for name, fn in tools_dict.items():
        sys_dict[name] = _bind_callable_to_state(state, fn)

    # 注入核心动作
    sys_dict["append_items"] = _bind_callable_to_state(state, _make_append_items(state))
    sys_dict["save_checkpoint"] = _bind_callable_to_state(state, _make_save_checkpoint(state, run_dir))
    sys_dict["final_answer"] = _bind_callable_to_state(state, _make_final_answer(state))

    # 注入数据辅助函数
    sys_dict["clean_items"] = clean_items
    sys_dict["ensure"] = ensure
    sys_dict["parse_http_response"] = parse_http_response
    sys_dict["summarize_json"] = summarize_json
    
    # 共享数据引用
    sys_dict["all_items"] = state.items.get_raw_list()

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


def _make_save_checkpoint(state: CodeAgentState, run_dir: str) -> Callable:
    async def save_checkpoint(filename: str = "checkpoint.json") -> str:
        target = filename or "checkpoint.json"
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
