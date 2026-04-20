from __future__ import annotations

import re
import sqlite3
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse


_BEIJING_TZ = timezone(timedelta(hours=8))


def _utcnow_iso() -> str:
    return datetime.now(_BEIJING_TZ).isoformat()


def normalize_site_key(value: str) -> str:
    """将 URL 或域名规范化为可检索域名。"""
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
    """从任务文本提取 URL 或裸域名。"""
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


def extract_urls(text: str) -> list[str]:
    """从任务文本中提取 URL。"""
    seen: set[str] = set()
    urls: list[str] = []
    for url in re.findall(r"https?://[^\s'\"<>]+", text):
        if url not in seen:
            urls.append(url)
            seen.add(url)
    return urls


def normalize_page_pattern(value: str) -> str:
    """将 URL/path 规范化为页面模式。"""
    candidate = value.strip()
    if not candidate:
        return ""
    if "://" in candidate:
        parsed = urlparse(candidate)
        path = parsed.path or "/"
    else:
        path = candidate
    path = path.strip() or "/"
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 1:
        path = path.rstrip("/")
    return path


@dataclass
class SiteSOP:
    domain: str
    sop_markdown: str
    golden_rule: str
    page_pattern: str = ""
    workflow_kind: str = "generic"
    should_inspect_first: bool = True
    preferred_entry: str = ""
    field_contract: list[str] | None = None
    confidence: float = 0.0
    quality_status: str = "active"
    update_reason: str = "manual"
    source_run_id: str = ""
    source_url: str = ""
    version: int = 1
    evidence_hash: str = ""
    proof_summary: str = ""
    last_generated_at: str = ""
    created_at: str = ""
    updated_at: str = ""

class SiteSOPStore:
    """SQLite 站点 SOP 存储。"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS site_sops (
                    domain TEXT PRIMARY KEY,
                    page_pattern TEXT NOT NULL DEFAULT '',
                    workflow_kind TEXT NOT NULL DEFAULT 'generic',
                    should_inspect_first INTEGER NOT NULL DEFAULT 1,
                    preferred_entry TEXT NOT NULL DEFAULT '',
                    field_contract TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL DEFAULT 0,
                    sop_markdown TEXT NOT NULL,
                    golden_rule TEXT NOT NULL,
                    quality_status TEXT NOT NULL DEFAULT 'active',
                    update_reason TEXT NOT NULL,
                    source_run_id TEXT NOT NULL,
                    source_url TEXT NOT NULL DEFAULT '',
                    version INTEGER NOT NULL DEFAULT 1,
                    evidence_hash TEXT NOT NULL DEFAULT '',
                    proof_summary TEXT NOT NULL DEFAULT '',
                    last_generated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS site_sop_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    page_pattern TEXT NOT NULL DEFAULT '',
                    workflow_kind TEXT NOT NULL DEFAULT 'generic',
                    should_inspect_first INTEGER NOT NULL DEFAULT 1,
                    preferred_entry TEXT NOT NULL DEFAULT '',
                    field_contract TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL,
                    sop_markdown TEXT NOT NULL,
                    update_reason TEXT NOT NULL,
                    source_run_id TEXT NOT NULL,
                    evidence_hash TEXT NOT NULL DEFAULT '',
                    proof_summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_site_sop_versions_domain_created
                ON site_sop_versions(domain, created_at);

                """
            )
            self._ensure_column(conn, "site_sops", "page_pattern", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "site_sops", "workflow_kind", "TEXT NOT NULL DEFAULT 'generic'")
            self._ensure_column(conn, "site_sops", "should_inspect_first", "INTEGER NOT NULL DEFAULT 1")
            self._ensure_column(conn, "site_sops", "preferred_entry", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "site_sops", "field_contract", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "site_sops", "confidence", "REAL NOT NULL DEFAULT 0")
            self._ensure_column(conn, "site_sop_versions", "page_pattern", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "site_sop_versions", "workflow_kind", "TEXT NOT NULL DEFAULT 'generic'")
            self._ensure_column(conn, "site_sop_versions", "should_inspect_first", "INTEGER NOT NULL DEFAULT 1")
            self._ensure_column(conn, "site_sop_versions", "preferred_entry", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "site_sop_versions", "field_contract", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(conn, "site_sop_versions", "confidence", "REAL NOT NULL DEFAULT 0")
            conn.commit()

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def get_active_sop(self, domain: str) -> SiteSOP | None:
        normalized = normalize_site_key(domain)
        if not normalized:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT domain, page_pattern, workflow_kind, should_inspect_first, preferred_entry,
                       field_contract, confidence, sop_markdown, golden_rule, quality_status, update_reason,
                       source_run_id, source_url, version, evidence_hash, proof_summary,
                       last_generated_at, created_at, updated_at
                FROM site_sops
                WHERE domain = ? AND quality_status = 'active'
                """,
                (normalized,),
            ).fetchone()
        return self._row_to_sop(row) if row else None

    def find_for_task(self, task: str) -> SiteSOP | None:
        urls = extract_urls(task)
        task_paths = [normalize_page_pattern(url) for url in urls]
        for site_key in extract_site_keys(task):
            sop = self.get_active_sop(site_key)
            if sop:
                if sop.page_pattern and task_paths:
                    if any(path.startswith(sop.page_pattern) for path in task_paths):
                        return sop
                    continue
                return sop
        return None

    def upsert_sop(self, sop: SiteSOP) -> SiteSOP:
        now = _utcnow_iso()
        normalized = normalize_site_key(sop.domain)
        if not normalized:
            raise ValueError("domain is required")

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT version, created_at FROM site_sops WHERE domain = ?",
                (normalized,),
            ).fetchone()
            next_version = (int(existing["version"]) + 1) if existing else 1
            created_at = str(existing["created_at"]) if existing else now
            last_generated_at = sop.last_generated_at or now
            conn.execute(
                """
                INSERT INTO site_sops(
                    domain, page_pattern, workflow_kind, should_inspect_first, preferred_entry,
                    field_contract, confidence, sop_markdown, golden_rule, quality_status, update_reason,
                    source_run_id, source_url, version, evidence_hash, proof_summary,
                    last_generated_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    page_pattern=excluded.page_pattern,
                    workflow_kind=excluded.workflow_kind,
                    should_inspect_first=excluded.should_inspect_first,
                    preferred_entry=excluded.preferred_entry,
                    field_contract=excluded.field_contract,
                    confidence=excluded.confidence,
                    sop_markdown=excluded.sop_markdown,
                    golden_rule=excluded.golden_rule,
                    quality_status=excluded.quality_status,
                    update_reason=excluded.update_reason,
                    source_run_id=excluded.source_run_id,
                    source_url=excluded.source_url,
                    version=excluded.version,
                    evidence_hash=excluded.evidence_hash,
                    proof_summary=excluded.proof_summary,
                    last_generated_at=excluded.last_generated_at,
                    updated_at=excluded.updated_at
                """,
                (
                    normalized,
                    normalize_page_pattern(sop.page_pattern),
                    sop.workflow_kind,
                    1 if sop.should_inspect_first else 0,
                    sop.preferred_entry,
                    json.dumps(sop.field_contract or [], ensure_ascii=False),
                    sop.confidence,
                    sop.sop_markdown,
                    sop.golden_rule,
                    sop.quality_status,
                    sop.update_reason,
                    sop.source_run_id,
                    sop.source_url,
                    next_version,
                    sop.evidence_hash,
                    sop.proof_summary,
                    last_generated_at,
                    created_at,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO site_sop_versions(
                    domain, page_pattern, workflow_kind, should_inspect_first, preferred_entry,
                    field_contract, confidence, version, sop_markdown, update_reason,
                    source_run_id, evidence_hash, proof_summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized,
                    normalize_page_pattern(sop.page_pattern),
                    sop.workflow_kind,
                    1 if sop.should_inspect_first else 0,
                    sop.preferred_entry,
                    json.dumps(sop.field_contract or [], ensure_ascii=False),
                    sop.confidence,
                    next_version,
                    sop.sop_markdown,
                    sop.update_reason,
                    sop.source_run_id,
                    sop.evidence_hash,
                    sop.proof_summary,
                    now,
                ),
            )
            conn.commit()

        stored = self.get_active_sop(normalized)
        if stored is None:
            raise RuntimeError("site SOP upsert succeeded but reload failed")
        return stored

    def recent_accepted_update_count(self, domain: str, *, hours: int = 24) -> int:
        normalized = normalize_site_key(domain)
        if not normalized:
            return 0
        cutoff = (datetime.now(_BEIJING_TZ) - timedelta(hours=hours)).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM site_sop_versions
                WHERE domain = ? AND created_at >= ?
                """,
                (normalized, cutoff),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def replace_active_sop(self, sop: SiteSOP) -> SiteSOP:
        """直接替换当前生效 SOP，并写入新版本，用于人工修正。"""
        return self.upsert_sop(sop)

    @staticmethod
    def _row_to_sop(row: sqlite3.Row) -> SiteSOP:
        return SiteSOP(
            domain=str(row["domain"]),
            page_pattern=str(row["page_pattern"] or ""),
            workflow_kind=str(row["workflow_kind"] or "generic"),
            should_inspect_first=bool(row["should_inspect_first"]),
            preferred_entry=str(row["preferred_entry"] or ""),
            field_contract=json.loads(str(row["field_contract"] or "[]")),
            confidence=float(row["confidence"] or 0.0),
            sop_markdown=str(row["sop_markdown"]),
            golden_rule=str(row["golden_rule"]),
            quality_status=str(row["quality_status"]),
            update_reason=str(row["update_reason"]),
            source_run_id=str(row["source_run_id"]),
            source_url=str(row["source_url"]),
            version=int(row["version"]),
            evidence_hash=str(row["evidence_hash"]),
            proof_summary=str(row["proof_summary"]),
            last_generated_at=str(row["last_generated_at"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
