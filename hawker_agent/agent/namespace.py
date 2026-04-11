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
from collections.abc import Callable
from pathlib import Path

from hawker_agent.models.state import CodeAgentState
from hawker_agent.observability import emit_observation
from hawker_agent.tools.data_tools import (
    clean_items,
    ensure,
    normalize_items,
    parse_http_response,
    save_file,
    summarize_json,
)

logger = logging.getLogger(__name__)


def _bind_callable_to_state(state: CodeAgentState, fn: Callable) -> Callable:
    """为 namespace 中的可调用对象注入日志上下文并处理 DomActionResult。"""
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapped(*args: object, **kwargs: object) -> object:
            with state.bind_log_context():
                result = await fn(*args, **kwargs)
                # 处理 DomActionResult: 侧道注入 DOM，只返回 summary
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


def _make_append_items(state: CodeAgentState) -> Callable:
    """创建异步的 append_items 函数。"""

    async def append_items(items: object) -> list[dict]:
        normalized = normalize_items(items)
        added, skipped = state.items.append(normalized)
        if added:
            state.mark_activity()
        
        # 使用专业的 emit_observation (直接输出给 LLM)
        sample = json.dumps(state.items.get_raw_list()[-1], ensure_ascii=False)[:120] if added else ""
        emit_observation(
            f"[append_items] +{added} -> total={len(state.items)}"
            + (f" | skipped={skipped}" if skipped else "")
            + (f" | 样本: {sample}" if sample else "")
        )
        return state.items.get_raw_list()

    return append_items


def _make_save_checkpoint(state: CodeAgentState, run_dir: str) -> Callable:
    """创建异步的 save_checkpoint 函数。"""

    async def save_checkpoint(filename: str = "checkpoint.json") -> str:
        target = filename or "checkpoint.json"
        state.checkpoint_files.add(target)
        result = save_file(
            json.dumps(state.items.get_raw_list(), ensure_ascii=False), target, run_dir
        )
        
        # 使用专业的 emit_observation
        emit_observation(f"[save_checkpoint] total={len(state.items)} | file={target}")
        return result

    return save_checkpoint


def _make_final_answer(state: CodeAgentState) -> Callable:
    """创建异步的 final_answer 函数。"""

    async def final_answer(answer: object) -> None:
        state.final_answer_requested = str(answer)
        # 使用专业的 emit_observation
        emit_observation(f"[final_answer] {answer}")

    return final_answer


def build_namespace(
    state: CodeAgentState,
    tools_dict: dict[str, Callable],
    run_dir: str,
) -> dict:
    """
    构建代码执行命名空间。所有核心功能统一异步化。
    """
    ns: dict = {}

    # 注入外部工具 (保持其原始异步性)
    for name, fn in tools_dict.items():
        ns[name] = _bind_callable_to_state(state, fn)

    # 注入数据辅助函数 (同步类工具)
    ns["clean_items"] = clean_items
    ns["ensure"] = ensure
    ns["parse_http_response"] = parse_http_response
    ns["summarize_json"] = summarize_json
    
    # 注入核心动作 (强制异步化并绑定 state)
    ns["append_items"] = _bind_callable_to_state(state, _make_append_items(state))
    ns["save_checkpoint"] = _bind_callable_to_state(state, _make_save_checkpoint(state, run_dir))
    ns["final_answer"] = _bind_callable_to_state(state, _make_final_answer(state))

    # 共享数据引用 (使用 get_raw_list 以实现 step 内实时同步)
    ns["all_items"] = state.items.get_raw_list()

    # 运行元数据
    ns["run_dir"] = run_dir
    
    # 标准库模块
    ns["asyncio"] = asyncio
    ns["json"] = json
    ns["re"] = re
    ns["time"] = time
    ns["base64"] = base64
    ns["hashlib"] = hashlib
    ns["urllib"] = urllib
    ns["Path"] = Path

    return ns
