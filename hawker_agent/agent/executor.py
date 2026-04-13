from __future__ import annotations

import ast
import asyncio
import contextlib
import copy
import inspect
import io
import logging
import traceback
from typing import TYPE_CHECKING

from hawker_agent.agent.compressor import truncate_output

if TYPE_CHECKING:
    from hawker_agent.models.state import CodeAgentState
    from hawker_agent.agent.namespace import HawkerNamespace

logger = logging.getLogger(__name__)

# Modules that LLM-generated code must never import
_BLOCKED_IMPORTS = frozenset({
    "os", "subprocess", "shutil", "sys", "socket", "ctypes",
    "signal", "multiprocessing", "threading", "_thread",
    "importlib", "runpy", "code", "codeop", "compileall",
    "webbrowser", "antigravity",
    "pickle", "shelve", "marshal",
})


def _check_imports(code: str) -> str | None:
    """Return an error message if *code* tries to import a blocked module, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # let compile() report the real error later
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BLOCKED_IMPORTS:
                    return f"[\u5b89\u5168\u9650\u5236] \u7981\u6b62\u5bfc\u5165\u6a21\u5757: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in _BLOCKED_IMPORTS:
                    return f"[\u5b89\u5168\u9650\u5236] \u7981\u6b62\u5bfc\u5165\u6a21\u5757: {node.module}"
    return None


def _clean_traceback(tb_str: str) -> str:
    """清理 Traceback，移除执行器内部的干扰行。"""
    lines = tb_str.split("\n")
    cleaned = []
    # 关注点是从 <hawker-cell> 开始的报错
    found_cell = False
    for line in lines:
        if '<hawker-cell>' in line:
            found_cell = True
        if found_cell:
            cleaned.append(line)
    
    if not cleaned:
        return tb_str
    return "Traceback (most recent call last):\n" + "\n".join(cleaned)


async def execute(
    code: str,
    namespace: HawkerNamespace,
    state: CodeAgentState | None = None,
    step: int | None = None,
) -> str:
    """
    使用 Native Top-level Await 执行代码。
    具备事务语义：成功则 commit 变量提升，失败则 rollback 回滚快照。
    """
    log_scope = state.bind_log_context(step) if state is not None else contextlib.nullcontext()
    
    with log_scope:
        buf = io.StringIO()
        logger.info("代码执行开始: chars=%d", len(code))

        # 事务快照：session 层深拷贝
        try:
            session_snapshot = copy.deepcopy(namespace.session)
        except Exception as e:
            logger.debug("Namespace 快照失败 (可能包含不可 pickle 的对象): %s", e)
            session_snapshot = namespace.session.copy() # 降级为浅拷贝

        # 0. Static import guard
        import_err = _check_imports(code)
        if import_err:
            namespace.rollback()
            return import_err

        try:
            # 1. 编译：允许顶层 await (Python 3.8+)
            compiled = compile(
                code, 
                "<hawker-cell>", 
                "exec", 
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
            )

            # 2. 执行：在统一视图中运行
            with contextlib.redirect_stdout(buf):
                # 合并为一个真正的 dict，因为 exec 不接受 ChainMap
                # 我们将这个 dict 同时作为 globals 和 locals 以模拟顶层执行环境
                view = dict(namespace.exec_view)
                
                # 核心：在 ALLOW_TOP_LEVEL_AWAIT 模式下，如果代码含 await，
                # 使用 eval() 执行编译后的代码块可以正确返回协程对象。
                # 提示：eval 支持执行以 'exec' 模式编译的代码对象。
                maybe_coro = eval(compiled, view)
                
                if inspect.isawaitable(maybe_coro):
                    await maybe_coro
                
                # 3. 同步回 cell_local
                for k, v in view.items():
                    if k not in namespace.system:
                        namespace.cell_local[k] = v


            # 4. 提交事务：变量提升
            namespace.commit()
            
            output = truncate_output(buf.getvalue().strip() or "[无输出]")
            logger.info("代码执行成功")
            return output

        except Exception:
            # 5. 回滚事务：彻底恢复 session 并清空 local
            namespace.session = session_snapshot
            namespace.rollback()
            
            tb = _clean_traceback(traceback.format_exc(limit=8).strip())
            
            # 错误诊断增强
            hints = []
            if "SyntaxError" in tb:
                if '"""' in code or "'''" in code:
                    hints.append("提示: 检查三引号字符串内的引号转义")
            if "NameError" in tb:
                hints.append("提示: 变量可能未定义或在本步骤回滚中被清除")
            
            error_msg = buf.getvalue().strip()
            if error_msg:
                error_msg += "\n"
            error_msg += f"[执行错误]\n{tb}"
            if hints:
                error_msg += "\n" + "\n".join(hints)
            
            final_output = truncate_output(error_msg)
            logger.warning("代码执行失败，已回滚 Namespace 状态")
            return final_output
