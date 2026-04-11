from __future__ import annotations

import ast
import contextlib
import io
import logging
import traceback
from typing import TYPE_CHECKING

from hawker_agent.agent.compressor import truncate_output

if TYPE_CHECKING:
    from hawker_agent.models.state import CodeAgentState

logger = logging.getLogger(__name__)


def _has_async_constructs(tree: ast.AST) -> bool:
    """检测 AST 树中是否包含异步关键字。"""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Await, ast.AsyncWith, ast.AsyncFor)):
            return True
    return False


def _get_assigned_names(tree: ast.AST) -> set[str]:
    """收集代码中所有被赋值的变量名。"""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            target = getattr(node, "target", None)
            if isinstance(target, ast.Name):
                names.add(target.id)
        elif isinstance(node, ast.Global):
            names.update(node.names)
    return names


def _clean_traceback(tb_str: str) -> str:
    """清理 Traceback，移除执行器内部的干扰行。"""
    lines = tb_str.split("\n")
    cleaned = []
    skip_next = False
    for line in lines:
        if "exec(compile(" in line or "exec(code" in line or "__code_exec__" in line:
            skip_next = True
            continue
        if skip_next and line.startswith("    ") and not line.strip().startswith("File"):
            skip_next = False
            continue
        skip_next = False
        cleaned.append(line)
    return "\n".join(cleaned)


async def execute(
    code: str,
    namespace: dict,
    state: CodeAgentState | None = None,
    step: int | None = None,
) -> str:
    """
    在隔离的 namespace 中执行 LLM 生成的代码。
    支持自动识别并运行顶层 await。
    """
    log_scope = state.bind_log_context(step) if state is not None else contextlib.nullcontext()
    
    with log_scope:
        buf = io.StringIO()
        logger.info("代码执行开始: chars=%d", len(code))
        logger.debug("--- 执行代码 ---\n%s\n---------------", code)
        
        try:
            # 1. 解析 AST
            try:
                tree = ast.parse(code, mode="exec")
            except SyntaxError:
                # 语法错误直接抛出，走下方的统一错误处理
                raise

            # 2. 判断是否需要异步包装
            if _has_async_constructs(tree):
                # 收集所有被赋值的变量名
                assigned_names = _get_assigned_names(tree)
                # 与 namespace 中已有的变量取交集，注入 global 声明，确保能读写外部变量
                existing_vars = {n for n in assigned_names if n in namespace}
                global_decl = f"    global {', '.join(sorted(existing_vars))}\n" if existing_vars else ""
                
                # 缩进原始代码并包装进异步函数
                indented_code = "\n".join(
                    "    " + line if line.strip() else line for line in code.split("\n")
                )
                # 核心逻辑：异步函数返回 locals()，以便我们抓取新定义的变量
                wrapped_code = (
                    f"async def __code_exec__():\n"
                    f"{global_decl}"
                    f"{indented_code}\n"
                    f"    return locals()\n"
                )
                
                with contextlib.redirect_stdout(buf):
                    # 在 namespace 中定义这个协程函数
                    exec(compile(wrapped_code, "<code>", "exec"), namespace, namespace)
                    coro_func = namespace.pop("__code_exec__", None)
                    if coro_func:
                        # 运行协程并获取其局部变量字典
                        result_locals = await coro_func()
                        if result_locals:
                            # 将新定义的局部变量同步回全局 namespace
                            for k, v in result_locals.items():
                                if not k.startswith("_"):
                                    namespace[k] = v
            else:
                # 纯同步代码，直接在 namespace 中执行
                with contextlib.redirect_stdout(buf):
                    exec(code, namespace, namespace)

        except Exception:
            tb = _clean_traceback(traceback.format_exc(limit=8).strip())
            # 错误诊断增强 (Hints)
            hints = []
            if "SyntaxError" in tb:
                if '"""' in code or "'''" in code:
                    hints.append("提示: 检查三引号字符串内的引号转义")
                if "{" in code and "}" in code and "f'" in code:
                    hints.append("提示: f-string 内的大括号需要用 {{ }} 转义")
            if "NameError" in tb:
                hints.append("提示: 变量可能在之前的步骤中未被赋值或已被覆盖")
            
            error_msg = buf.getvalue().strip()
            if error_msg:
                error_msg += "\n"
            error_msg += f"[执行错误]\n{tb}"
            if hints:
                error_msg += "\n" + "\n".join(hints)
            
            final_output = truncate_output(error_msg)
            logger.info("代码执行结束: success=false output_chars=%d", len(final_output))
            return final_output

        # 执行成功，返回捕获的 stdout
        output = truncate_output(buf.getvalue().strip() or "[无输出]")
        logger.info("代码执行结束: success=true output_chars=%d", len(output))
        logger.debug("--- 执行输出 ---\n%s\n---------------", output)
        return output
