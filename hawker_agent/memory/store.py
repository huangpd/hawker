from __future__ import annotations

import ast
import builtins
import hashlib
import re
import sqlite3
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


# 模块职责：
# 1) 从任务文本/执行记录中提取可复用的结构化记忆；
# 2) 将记忆持久化到本地 SQLite；
# 3) 基于站点 + 意图做轻量打分召回。


_BEIJING_TZ = timezone(timedelta(hours=8))


def _utcnow_iso() -> str:
    """返回当前北京时间的 ISO 字符串。"""
    return datetime.now(_BEIJING_TZ).isoformat()


def normalize_site_key(value: str) -> str:
    """将 URL 或域名规范化为可检索的站点键。"""
    candidate = value.strip()
    if not candidate:
        return ""
    if "://" in candidate:
        parsed = urlparse(candidate)
        host = parsed.hostname or ""
    else:
        host = candidate
    host = host.lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def extract_site_keys(text: str) -> list[str]:
    """从任务文本中提取 URL 或裸域名。"""
    site_keys: list[str] = []
    seen: set[str] = set()

    for url in re.findall(r"https?://[^\s'\"<>]+", text):
        site_key = normalize_site_key(url)
        if site_key and site_key not in seen:
            site_keys.append(site_key)
            seen.add(site_key)

    for host in re.findall(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b", text):
        site_key = normalize_site_key(host)
        if site_key and site_key not in seen:
            site_keys.append(site_key)
            seen.add(site_key)

    return site_keys


def infer_task_intent(task: str) -> str:
    """根据任务文本推断任务意图。"""
    lowered = task.lower()
    rules = [
        ("sort", ("排序", "sort", "order by", "按下载量", "rank")),
        ("login", ("登录", "login", "cookie", "验证码", "captcha")),
        ("download", ("下载", "download", "文件", "pdf", "附件")),
        ("pagination", ("下一页", "翻页", "page=", "分页", "page ")),
        ("search", ("搜索", "search", "query", "检索")),
        ("api", ("api", "xhr", "fetch", "接口", "抓包", "network")),
        ("extract", ("提取", "采集", "列表", "title", "url", "字段")),
    ]
    for intent, keywords in rules:
        if any(keyword in lowered for keyword in keywords):
            return intent
    return "general"


def infer_page_kind(task: str, source_url: str = "") -> str:
    """根据任务文本和 URL 粗略推断页面类型。"""
    lowered = f"{task} {source_url}".lower()
    rules = [
        ("login_page", ("登录", "login")),
        ("search_page", ("搜索", "search", "query")),
        ("download_page", ("下载", "download", ".pdf", ".zip")),
        ("list_page", ("列表", "list", "explore", "trending", "排行", "next", "page")),
        ("detail_page", ("详情", "detail", "article", "post")),
    ]
    for kind, keywords in rules:
        if any(keyword in lowered for keyword in keywords):
            return kind
    return "general_page"


@dataclass
class MemoryEntry:
    """单条结构化记忆。"""

    memory_type: str
    site_key: str
    task_intent: str
    page_kind: str
    summary: str
    detail: str
    success: bool
    negative: bool
    confidence: float
    source_run_id: str
    source_step: int
    source_url: str
    fingerprint: str = ""
    seen_count: int = 1

    def ensure_fingerprint(self) -> str:
        """生成并缓存当前记忆的去重指纹。"""
        if not self.fingerprint:
            raw = "|".join(
                [
                    self.memory_type,
                    self.site_key,
                    self.task_intent,
                    self.page_kind,
                    self.summary,
                    str(self.negative),
                ]
            )
            self.fingerprint = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        return self.fingerprint


@dataclass
class MemoryMatch:
    """单条命中的记忆及其召回分数。"""

    entry: MemoryEntry
    score: float
    reason: str

    def render(self) -> str:
        """渲染命中的记忆为工作区摘要。"""
        if self.entry.memory_type.startswith("raw_"):
            tag = "成功代码"
        elif self.entry.memory_type.endswith("_recipe"):
            tag = "可执行配方"
        else:
            tag = "失败约束" if self.entry.negative else "站点经验"
        return f"- {tag}: {self.entry.summary} (score={self.score:.1f}, {self.reason})"


def compact_code(code: str, limit: int = 1200) -> str:
    """压缩代码文本并保留可读性。

    Args:
        code (str): 原始代码文本。
        limit (int): 结果最大字符长度。

    Returns:
        str: 清理并按需截断后的代码文本。
    """
    collapsed = "\n".join(line.rstrip() for line in code.strip().splitlines() if line.strip())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _infer_memory_type(code: str) -> str | None:
    """根据代码内容推断成功代码记忆类型。

    Args:
        code (str): 待分析的代码文本。

    Returns:
        str | None: 命中的记忆类型；若无法识别则返回 None。
    """
    lowered = code.lower()
    if "js(" in lowered:
        return "raw_extract_code"
    if "get_network_log(" in lowered or "http_json(" in lowered or "http_request(" in lowered:
        return "raw_network_code"
    if "click_index(" in lowered or "click(" in lowered:
        return "raw_pagination_code"
    return None


def _build_summary(site_key: str, memory_type: str, task_intent: str) -> str:
    """生成简短的人类可读摘要。"""
    if memory_type == "raw_extract_code":
        return f"{site_key} 成功提取代码 ({task_intent})"
    if memory_type == "raw_pagination_code":
        return f"{site_key} 成功翻页代码 ({task_intent})"
    if memory_type == "raw_network_code":
        return f"{site_key} 成功抓包/请求代码 ({task_intent})"
    return f"{site_key} 成功代码 ({task_intent})"


_RUNTIME_SYMBOLS: set[str] = {
    # modules / common helpers
    "json", "asyncio", "csv", "re", "datetime", "Path", "requests", "httpx",
    # data holders
    "all_items", "run_dir",
    # browser + network tools
    "nav", "dom_state", "nav_search", "js", "click", "click_index", "fill_input",
    "http_request", "get_network_log", "http_json", "get_cookies", "get_selector_from_index",
    "browser_download", "download_file",
    # data tools + core actions
    "parse_http_response", "clean_items", "ensure", "append_items",
    "save_checkpoint", "final_answer", "observe",
    # async variants
    "async_nav", "async_dom_state", "async_nav_search", "async_js",
    "async_click", "async_click_index", "async_fill_input", "async_get_network_log",
}


def _extract_external_dependencies(code: str) -> list[str]:
    """提取代码片段依赖的外部变量名。

    Args:
        code (str): 待分析的代码片段。

    Returns:
        list[str]: 外部依赖变量名列表（去重后按字母序）。
    """
    wrapped = "async def __mem_probe__():\n" + textwrap.indent(code, "    ")
    try:
        tree = ast.parse(wrapped)
    except SyntaxError:
        return []

    loaded: set[str] = set()
    assigned: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
            elif isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assigned.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assigned.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assigned.add(alias.asname or alias.name)

    builtin_names = set(dir(builtins))
    external = loaded - assigned - builtin_names - _RUNTIME_SYMBOLS - {"__mem_probe__"}
    return sorted(name for name in external if not name.startswith("_"))


def _collect_dependency_context(llm_records: list[dict[str, Any]], current_idx: int, deps: list[str]) -> str:
    """从前序步骤中提取依赖变量定义行。

    Args:
        llm_records (list[dict[str, Any]]): 全量步骤记录。
        current_idx (int): 当前命中步骤在 llm_records 中的索引。
        deps (list[str]): 外部依赖变量名列表。

    Returns:
        str: 依赖变量定义行文本；若无可用内容则返回空字符串。
    """
    lines: list[str] = []
    for dep in deps:
        pattern = re.compile(rf"^\s*{re.escape(dep)}\s*=")
        for prev in range(current_idx - 1, -1, -1):
            prev_parsed = llm_records[prev].get("parsed_output") or {}
            prev_code = str(prev_parsed.get("code") or "")
            for raw_line in prev_code.splitlines():
                if pattern.match(raw_line):
                    snippet = raw_line.strip()
                    if len(snippet) > 220:
                        snippet = snippet[:217] + "..."
                    lines.append(snippet)
                    break
            else:
                continue
            break
    return "\n".join(lines)


def build_raw_code_memories(task: str, state: Any) -> list[MemoryEntry]:
    """从成功步骤中直接提取原始代码记忆。

    仅抽取实际产生进展的步骤，并过滤收尾或纯保底动作。

    Args:
        task (str): 当前任务文本。
        state (Any): 运行时状态对象，需包含 llm_records 等字段。

    Returns:
        list[MemoryEntry]: 可持久化的原始代码记忆列表。
    """
    source_url = ""
    if getattr(state, "last_dom_snapshot", None):
        source_url = str(state.last_dom_snapshot.get("url") or "")
    site_keys = extract_site_keys(task)
    site_key = normalize_site_key(source_url) or (site_keys[0] if site_keys else "")
    if not site_key:
        return []

    task_intent = infer_task_intent(task)
    page_kind = infer_page_kind(task, source_url)
    results: list[MemoryEntry] = []
    seen_types: set[str] = set()
    llm_records: list[dict[str, Any]] = list(getattr(state, "llm_records", []))

    # 逆序扫描，优先保留最近一次成功策略。
    for idx in range(len(llm_records) - 1, -1, -1):
        record = llm_records[idx]
        parsed = record.get("parsed_output") or {}
        execution = record.get("execution") or {}
        code = str(parsed.get("code") or "").strip()
        if not code or not execution.get("progress_made"):
            continue
        lowered = code.lower()
        # 过滤不具备可迁移价值的动作。
        if "final_answer(" in lowered:
            continue
        if "save_checkpoint(" in lowered and "js(" not in lowered and "click" not in lowered:
            continue

        memory_type = _infer_memory_type(code)
        if not memory_type or memory_type in seen_types:
            continue

        deps = _extract_external_dependencies(code)
        observation = str(execution.get("observation") or "").strip()
        detail = compact_code(code)
        if deps:
            dep_text = ", ".join(deps[:6])
            detail = (
                f"# 依赖前置变量: {dep_text}\n"
                "# 复用前请先补齐这些变量定义。\n\n"
                + detail
            )
            context = _collect_dependency_context(llm_records, idx, deps[:6])
            if context:
                detail += "\n\n# 依赖变量示例\n" + context
        if observation:
            detail += "\n\n# 上次成功证据\n" + observation[:300]

        results.append(
            MemoryEntry(
                memory_type=memory_type,
                site_key=site_key,
                task_intent=task_intent,
                page_kind=page_kind,
                summary=_build_summary(site_key, memory_type, task_intent),
                detail=detail,
                success=True,
                negative=False,
                confidence=0.82,
                source_run_id=state.run_id,
                source_step=int(record.get("step") or 0),
                source_url=source_url,
            )
        )
        seen_types.add(memory_type)
        # 控制记忆数量，避免同一次 run 产生过多冗余条目。
        if len(results) >= 3:
            break

    return list(reversed(results))


class MemoryStore:
    """本地 SQLite 记忆库。"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """创建数据库连接并启用按列名访问。

        Returns:
            sqlite3.Connection: 已配置 row_factory 的数据库连接。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """初始化表结构与索引。"""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT NOT NULL UNIQUE,
                    memory_type TEXT NOT NULL,
                    site_key TEXT NOT NULL,
                    task_intent TEXT NOT NULL,
                    page_kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    negative INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    source_run_id TEXT NOT NULL,
                    source_step INTEGER NOT NULL,
                    source_url TEXT NOT NULL,
                    seen_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_site_intent
                ON memory_entries(site_key, task_intent, updated_at)
                """
            )

    def upsert_entries(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """批量写入或合并记忆条目。

        Args:
            entries (list[MemoryEntry]): 待写入或合并的记忆条目。

        Returns:
            list[MemoryEntry]: 原样返回输入条目，便于调用链继续处理。
        """
        if not entries:
            return []

        now = _utcnow_iso()
        with self._connect() as conn:
            for entry in entries:
                entry.ensure_fingerprint()
                conn.execute(
                    """
                    INSERT INTO memory_entries (
                        fingerprint, memory_type, site_key, task_intent, page_kind,
                        summary, detail, success, negative, confidence,
                        source_run_id, source_step, source_url, seen_count,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                    ON CONFLICT(fingerprint) DO UPDATE SET
                        detail=excluded.detail,
                        success=excluded.success,
                        negative=excluded.negative,
                        confidence=MAX(memory_entries.confidence, excluded.confidence),
                        source_run_id=excluded.source_run_id,
                        source_step=excluded.source_step,
                        source_url=excluded.source_url,
                        seen_count=memory_entries.seen_count + 1,
                        updated_at=excluded.updated_at
                    """,
                    (
                        entry.fingerprint,
                        entry.memory_type,
                        entry.site_key,
                        entry.task_intent,
                        entry.page_kind,
                        entry.summary,
                        entry.detail,
                        int(entry.success),
                        int(entry.negative),
                        entry.confidence,
                        entry.source_run_id,
                        entry.source_step,
                        entry.source_url,
                        now,
                        now,
                    ),
                )
        return entries

    def search(self, task: str, *, site_keys: list[str] | None = None, limit: int = 5) -> list[MemoryMatch]:
        """按站点和任务意图召回记忆。

        Args:
            task (str): 当前任务文本。
            site_keys (list[str] | None): 外部指定的站点键列表，未提供时自动提取。
            limit (int): 召回条目上限。

        Returns:
            list[MemoryMatch]: 按分数降序排列的召回结果。
        """
        resolved_site_keys = site_keys or extract_site_keys(task)
        if not resolved_site_keys:
            return []

        task_intent = infer_task_intent(task)
        placeholders = ",".join("?" for _ in resolved_site_keys)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    fingerprint,
                    memory_type,
                    site_key,
                    task_intent,
                    page_kind,
                    summary,
                    detail,
                    success,
                    negative,
                    confidence,
                    source_run_id,
                    source_step,
                    source_url,
                    seen_count,
                    (
                        CASE WHEN site_key IN ({placeholders}) THEN 100 ELSE 0 END +
                        CASE WHEN task_intent = ? THEN 25
                             WHEN task_intent = 'general' THEN 10
                             ELSE 0 END +
                        confidence * 10 +
                        MIN(seen_count, 5)
                    ) AS score
                FROM memory_entries
                WHERE site_key IN ({placeholders})
                  AND (task_intent = ? OR task_intent = 'general')
                ORDER BY score DESC, updated_at DESC
                LIMIT ?
                """,
                [*resolved_site_keys, task_intent, *resolved_site_keys, task_intent, limit],
            ).fetchall()

        matches: list[MemoryMatch] = []
        for row in rows:
            entry = MemoryEntry(
                memory_type=row["memory_type"],
                site_key=row["site_key"],
                task_intent=row["task_intent"],
                page_kind=row["page_kind"],
                summary=row["summary"],
                detail=row["detail"],
                success=bool(row["success"]),
                negative=bool(row["negative"]),
                confidence=float(row["confidence"]),
                source_run_id=row["source_run_id"],
                source_step=int(row["source_step"]),
                source_url=row["source_url"],
                fingerprint=row["fingerprint"],
                seen_count=int(row["seen_count"]),
            )
            reason_parts = [f"site={entry.site_key}"]
            if entry.task_intent == task_intent:
                reason_parts.append(f"intent={task_intent}")
            if entry.negative:
                reason_parts.append("negative")
            matches.append(MemoryMatch(entry=entry, score=float(row["score"]), reason=", ".join(reason_parts)))
        return matches
