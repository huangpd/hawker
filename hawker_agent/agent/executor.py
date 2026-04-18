from __future__ import annotations

import ast
import contextlib
import copy
import inspect
import io
import logging
import re
import traceback
from typing import TYPE_CHECKING

from hawker_agent.agent.compressor import truncate_output
from hawker_agent.agent.healer import try_heal_code
from hawker_agent.observability import collect_observations, trace

if TYPE_CHECKING:
    from hawker_agent.models.state import CodeAgentState
    from hawker_agent.agent.namespace import HawkerNamespace

logger = logging.getLogger(__name__)

# LLM模型生成的代码绝对不能导入的模块
_BLOCKED_IMPORTS = frozenset({
    "os", "subprocess", "shutil", "sys", "socket", "ctypes",
    "signal", "multiprocessing", "threading", "_thread",
    "importlib", "runpy", "code", "codeop", "compileall",
    "webbrowser", "antigravity",
    "pickle", "shelve", "marshal",
})


def _check_imports(code: str) -> str | None:
    """检查代码是否尝试导入任何被禁止的模块。

    Args:
        code (str): 要检查的 Python 代码。

    Returns:
        str | None: 如果发现被禁止的模块，则返回错误消息；否则返回 None。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # 让 compile() 稍后报告真正的错误
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BLOCKED_IMPORTS:
                    return f"[安全限制] 禁止导入模块: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in _BLOCKED_IMPORTS:
                    return f"[安全限制] 禁止导入模块: {node.module}"
    return None


def _clean_traceback(tb_str: str) -> str:
    """通过移除执行器内部框架来清理回溯 (traceback) 字符串。

    重点关注从执行的代码块开始的框架。

    Args:
        tb_str (str): 原始的回溯字符串。

    Returns:
        str: 清理后的回溯字符串。
    """
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


def _log_legacy_stdout(stdout_text: str) -> None:
    """在剥离 ANSI 颜色代码后，记录非结构化的标准输出。

    Args:
        stdout_text (str): 原始的标准输出文本。
    """
    if not stdout_text:
        return
    
    # 正则清理 ANSI 转义序列 (Color Codes)
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned_text = ansi_escape.sub('', stdout_text).strip()
    
    if not cleaned_text:
        return

    preview = truncate_output(cleaned_text, 400).replace("\n", "\\n")
    logger.info("捕获未结构化 stdout，未注入 Observation: %s", preview)


async def execute(
    code: str,
    namespace: HawkerNamespace,
    state: CodeAgentState | None = None,
    step: int | None = None,
    _healing_depth: int = 0,
) -> str:
    """使用原生的顶层 await 执行 Python 代码。

    执行具有事务语义：
    - 成功时：变量被提升（提交）。
    - 失败时：会话回滚到之前的状态。

    Args:
        code (str): 要执行的 Python 代码。
        namespace (HawkerNamespace): 执行所需的命名空间。
        state (CodeAgentState | None): 可选的代理状态，用于日志记录。
        step (int | None): 可选的步骤编号，用于追踪。

    Returns:
        str: 执行的输出（观察结果或错误消息）。
    """
    log_context = state.bind_log_context(step) if state else contextlib.nullcontext()
    with log_context:
        with trace(f"execute_code_{step or 'anon'}") as span:
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
                with collect_observations() as observations:
                    with contextlib.redirect_stdout(buf):
                        # 合并为一个真正的 dict，因为 exec 不接受 ChainMap
                        # 我们将这个 dict 同时作为 globals 和 locals 以模拟顶层执行环境
                        view = dict(namespace.exec_view)

                        # 核心：在 ALLOW_TOP_LEVEL_AWAIT 模式下，如果代码含 await，
                        # 使用 eval() 执行编译后的代码块可以正确返回协程对象。
                        # 即使模型忘了写 await，如果返回的是协程，我们也帮它完成。
                        maybe_coro = eval(compiled, view)

                        if inspect.isawaitable(maybe_coro):
                            # 情况 A: 显式使用了 top-level await
                            result = await maybe_coro
                        else:
                            # 情况 B: 模型可能忘了写 await，导致返回了一个 coroutine 对象
                            # 或者代码块最后一行是一个 async 函数调用但没写 await
                            result = maybe_coro

                        # 3. 自动救治：如果最后执行出的 result 仍然是个协程对象，说明调用没完成
                        if inspect.isawaitable(result):
                            result = await result

                        # 4. 同步回 cell_local
                        for k, v in view.items():
                            if k not in namespace.system:
                                namespace.cell_local[k] = v
                # 4. 提交事务：变量提升
                namespace.commit()

                legacy_stdout = buf.getvalue().strip()
                _log_legacy_stdout(legacy_stdout)
                observations_str = "\n".join(observations).strip()
                output = truncate_output(observations_str or "[无输出]")

                span.data.update({
                    "observations": observations_str,
                    "legacy_stdout": legacy_stdout,
                })

                logger.info("代码执行成功")
                return output

            except Exception:
                # 5. 回滚事务：彻底恢复 session 并清空 local
                namespace.session = session_snapshot
                namespace.rollback()

                legacy_stdout = buf.getvalue().strip()
                _log_legacy_stdout(legacy_stdout)
                tb = _clean_traceback(traceback.format_exc(limit=8).strip())

                # 错误诊断增强
                hints = []
                if "SyntaxError" in tb:
                    if '"""' in code or "'''" in code:
                        hints.append("提示: 检查三引号字符串内的引号转义")
                if "NameError" in tb:
                    hints.append("提示: 变量可能未定义或在本步骤回滚中被清除")

                error_msg = f"[执行错误]\n{tb}"
                if hints:
                    error_msg += "\n" + "\n".join(hints)

                if state is not None:
                    healed_code = await try_heal_code(
                        code=code,
                        error=error_msg,
                        namespace=namespace,
                        state=state,
                    )
                    if healed_code and _healing_depth < 3:
                        logger.info("代码执行失败，进入 Healing 重试: step=%s depth=%d", step, _healing_depth + 1)
                        return await execute(
                            healed_code,
                            namespace,
                            state=state,
                            step=step,
                            _healing_depth=_healing_depth + 1,
                        )

                final_output = truncate_output(error_msg)

                span.status = "error"
                span.data.update({
                    "error": tb,
                    "legacy_stdout": legacy_stdout,
                })

                logger.warning("代码执行失败，已回滚 Namespace 状态")
                return final_output
