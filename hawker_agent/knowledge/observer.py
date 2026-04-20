from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from hawker_agent.agent.prompts import render_template
from hawker_agent.browser import actions
from hawker_agent.browser.session import BrowserSession
from hawker_agent.config import Settings
from hawker_agent.knowledge.store import SiteSOP, SiteSOPStore, extract_site_keys, extract_urls, normalize_page_pattern
from hawker_agent.llm.client import LLMClient
from hawker_agent.models.cell import CellStatus, CodeCell
from hawker_agent.models.state import CodeAgentState

logger = logging.getLogger(__name__)

_BEIJING_TZ = timezone(timedelta(hours=8))
_EXAMPLE_DIR = Path(__file__).parent.parent / "templates" / "observer_examples"


@dataclass
class ObserverEvidence:
    domain: str
    execution_log: str
    network_summary: str
    source_url: str


@dataclass
class SOPValidationResult:
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class ObserverExample:
    key: str
    title: str
    content: str


def _today_text() -> str:
    return datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d")


def load_observer_examples() -> dict[str, ObserverExample]:
    """加载内置 few-shot 案例库。"""
    examples: dict[str, ObserverExample] = {}
    file_map = {
        "api_only": "api_only.md",
        "hybrid": "hybrid.md",
        "browser_required": "browser_required.md",
    }
    for key, file_name in file_map.items():
        path = _EXAMPLE_DIR / file_name
        content = path.read_text(encoding="utf-8").strip()
        title = content.splitlines()[0].lstrip("# ").strip() if content else key
        examples[key] = ObserverExample(key=key, title=title, content=content)
    return examples


def classify_observer_evidence(execution_log: str, network_summary: str) -> str:
    """根据主 Agent 证据判断更适合的 few-shot 类型。"""
    exec_lower = execution_log.lower()
    net_lower = network_summary.lower()

    api_signals = sum(
        token in exec_lower or token in net_lower
        for token in ("http_json(", "http_request(", "application/json", "/api/", "response_sample:")
    )
    browser_signals = sum(
        token in exec_lower
        for token in ("await nav(", "await click(", "await fill_input(", "await js(", "dom_state(")
    )

    if api_signals >= 2 and browser_signals == 0:
        return "api_only"
    if browser_signals >= 2 and api_signals == 0:
        return "browser_required"
    if api_signals >= 1 and browser_signals >= 1:
        return "hybrid"
    if api_signals > browser_signals:
        return "api_only"
    return "browser_required"


def select_observer_examples(execution_log: str, network_summary: str, *, max_examples: int = 2) -> list[ObserverExample]:
    """按证据类型选择 few-shot 参考案例。"""
    examples = load_observer_examples()
    primary_key = classify_observer_evidence(execution_log, network_summary)
    order = [primary_key]
    for fallback in ("hybrid", "api_only", "browser_required"):
        if fallback not in order:
            order.append(fallback)
    return [examples[key] for key in order[:max_examples] if key in examples]


def _extract_source_url(cells: list[CodeCell]) -> str:
    for cell in reversed(cells):
        if cell.url:
            return cell.url
    return ""


def infer_observer_domain(task: str, cells: list[CodeCell]) -> str:
    """尽可能早地从 task / cells 推断目标域名。"""
    domains = extract_site_keys(task)
    if domains:
        return domains[0]
    source_url = _extract_source_url(cells)
    fallback = extract_site_keys(source_url) if source_url else []
    return fallback[0] if fallback else ""


def _format_cell(cell: CodeCell) -> str:
    parts = [f"Step {cell.step} | status={cell.status.value} | items={cell.items_count}"]
    if cell.thought.strip():
        parts.append(f"Thought: {cell.thought.strip()[:240]}")
    if cell.source.strip():
        parts.append(f"```python\n{cell.source.strip()}\n```")
    if cell.error:
        parts.append(f"Error: {cell.error.strip()[:300]}")
    elif cell.output:
        parts.append(f"Observation: {cell.output.strip()[:500]}")
    return "\n".join(parts)


def build_execution_log(cells: list[CodeCell], items: list[dict[str, Any]], *, max_cells: int = 4) -> str:
    """从成功路径和关键失败构建紧凑执行日志。"""
    successful = [cell for cell in cells if cell.status == CellStatus.SUCCESS and cell.source.strip()]
    failed = [
        cell for cell in cells
        if cell.status == CellStatus.ERROR and (cell.error or "").strip() and "SyntaxError" not in (cell.error or "")
    ]

    selected: list[CodeCell] = []
    if failed:
        selected.extend(failed[-1:])
    if successful:
        selected.extend(successful[-max_cells:])
    if not selected:
        selected = cells[-max_cells:]

    sections = [_format_cell(cell) for cell in selected]
    if items:
        sample = json.dumps(items[:2], ensure_ascii=False, indent=2)
        sections.append(f"Confirmed data sample:\n{sample}")
    return "\n\n".join(sections).strip() or "No execution evidence available."


def build_network_summary(netlog_result: dict[str, Any]) -> str:
    """从 netlog 结果构建适合 SOP 蒸馏的摘要。"""
    entries = list(netlog_result.get("entries") or [])
    if not entries:
        return "No useful network evidence."

    lines: list[str] = []
    for entry in entries[:10]:
        url = str(entry.get("url") or "")
        method = str(entry.get("method") or "GET").upper()
        status = entry.get("status")
        content_type = ""
        headers = entry.get("headers")
        if isinstance(headers, dict):
            for key, value in headers.items():
                if str(key).lower() == "content-type":
                    content_type = str(value)
                    break
        req_body = str(entry.get("reqBody") or entry.get("requestBody") or "")
        body = str(entry.get("body") or "")
        line = f"- {method} {url} | status={status}"
        if content_type:
            line += f" | content_type={content_type}"
        if req_body:
            line += f"\n  request_body: {req_body[:300]}"
        if body:
            line += f"\n  response_sample: {body[:500]}"
        lines.append(line)
    return "\n".join(lines)


def extract_requested_fields(task: str) -> list[str]:
    """从任务文本中提取显式要求的字段名。"""
    fields: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"[-•]\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:：]", task):
        field = match.group(1)
        lowered = field.lower()
        if lowered not in seen:
            fields.append(field)
            seen.add(lowered)
    return fields


def infer_page_pattern(task: str, source_url: str) -> str:
    """根据 task/source_url 推断页面路径模式。"""
    urls = extract_urls(task)
    if urls:
        return normalize_page_pattern(urls[0])
    return normalize_page_pattern(source_url)


def infer_should_inspect_first(workflow_kind: str, page_pattern: str, execution_log: str) -> bool:
    """判断命中 SOP 后是否仍需先侦察。"""
    log_lower = execution_log.lower()
    if workflow_kind == "api_only":
        return False
    if page_pattern == "/trending" and "await js(" in log_lower:
        return False
    if "await js(" in log_lower and "confirmed data sample" in log_lower.lower():
        return False
    return True


def infer_preferred_entry(workflow_kind: str, page_pattern: str) -> str:
    if workflow_kind == "api_only":
        return "api_direct"
    if page_pattern == "/trending":
        return "nav_summary_then_extract"
    if workflow_kind == "hybrid":
        return "nav_summary_then_network"
    return "inspect_then_extract"


def extract_golden_rule(markdown: str) -> str:
    """从 SOP Markdown 中抽取 Golden Rule。"""
    bold_match = re.search(r"\*\*Golden Rule:\*\*\s*(.+)", markdown)
    if bold_match:
        return bold_match.group(1).strip()
    plain_match = re.search(r"Golden Rule:\s*(.+)", markdown)
    if plain_match:
        return plain_match.group(1).strip()
    return "Prefer the shortest verified path supported by live evidence."


def _extract_section(markdown: str, heading: str) -> str:
    pattern = rf"{re.escape(heading)}\n(.*?)(?=\n## |\Z)"
    match = re.search(pattern, markdown, re.S)
    return match.group(1).strip() if match else ""


def _is_placeholder_api_reference(section_text: str) -> bool:
    lowered = section_text.lower().strip()
    if not lowered:
        return True
    placeholders = [
        "暂无稳定 api",
        "暂无 api",
        "no stable api",
        "no stable api reference",
    ]
    return any(token in lowered for token in placeholders)


def _merge_bullets(old_text: str, new_text: str, *, limit: int = 8) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for block in (new_text, old_text):
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line.startswith("- "):
                continue
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            lines.append(line)
            if len(lines) >= limit:
                return "\n".join(lines)
    return "\n".join(lines)


def smart_merge_sop(existing_markdown: str, candidate_markdown: str) -> str:
    """对新旧 SOP 做最小智能合并，避免无限追加和空占位回退。"""
    if not existing_markdown.strip():
        return candidate_markdown.strip()

    merged = candidate_markdown.strip()
    reference_headings = [
        "## API reference",
        "## URL reference",
        "## URL and ID reference",
        "## Endpoint reference",
    ]
    for heading in reference_headings:
        old_ref = _extract_section(existing_markdown, heading)
        new_ref = _extract_section(candidate_markdown, heading)
        if new_ref and _is_placeholder_api_reference(new_ref) and not _is_placeholder_api_reference(old_ref):
            merged = merged.replace(f"{heading}\n\n{new_ref}", f"{heading}\n\n{old_ref}")

    old_gotchas = _extract_section(existing_markdown, "## Gotchas")
    new_gotchas = _extract_section(candidate_markdown, "## Gotchas")
    merged_gotchas = _merge_bullets(old_gotchas, new_gotchas)
    if merged_gotchas:
        merged = re.sub(
            r"## Gotchas\n(.*?)(?=\n## |\Z)",
            f"## Gotchas\n\n{merged_gotchas}\n",
            merged,
            flags=re.S,
        )
    return merged.strip()


def validate_browser_harness_style_sop(markdown: str, domain: str) -> SOPValidationResult:
    """校验 SOP 是否符合 browser-harness 风格的最小录入标准。"""
    stripped = markdown.strip()
    expected_title = f"# {domain} — Scraping & Data Extraction"
    if not stripped.startswith(expected_title):
        return SOPValidationResult(False, "title does not match browser-harness style")

    header_body = stripped.split("\n", 1)[1].strip() if "\n" in stripped else ""
    if not header_body:
        return SOPValidationResult(False, "missing intro summary under title")

    required_sections = [
        "## Do this first",
        "## Common workflows",
        "## Gotchas",
    ]
    for section in required_sections:
        if section not in stripped:
            return SOPValidationResult(False, f"missing required section: {section}")

    intro_end = min((stripped.find(section) for section in required_sections if stripped.find(section) != -1), default=-1)
    intro_block = stripped[len(expected_title):intro_end].strip() if intro_end > 0 else header_body
    if "http" not in intro_block.lower() and domain not in intro_block.lower():
        return SOPValidationResult(False, "intro summary should mention target URL/domain")

    if "# Confirmed output (" not in stripped:
        return SOPValidationResult(False, "missing confirmed output proof")

    code_block_count = stripped.count("```python")
    if code_block_count == 0:
        return SOPValidationResult(False, "missing python workflow blocks")

    secret_patterns = [
        r"(?i)\bcookie:\s*\S+",
        r"(?i)\bbearer\s+[a-z0-9._\-]+",
        r"(?i)\bsession(id)?\s*[:=]\s*[a-z0-9._\-]{8,}",
        r"(?i)\bapi[-_ ]?key\s*[:=]\s*\S+",
    ]
    for pattern in secret_patterns:
        if re.search(pattern, stripped):
            return SOPValidationResult(False, "contains sensitive secret-like material")

    diary_patterns = [
        r"第一步.*第二步.*第三步",
        r"first we .* then we .* finally",
        r"首先打开.*然后点击",
    ]
    for pattern in diary_patterns:
        if re.search(pattern, stripped, re.I | re.S):
            return SOPValidationResult(False, "looks like execution diary instead of SOP")

    return SOPValidationResult(True)


async def collect_observer_evidence(
    *,
    task: str,
    cells: list[CodeCell],
    items: list[dict[str, Any]],
    browser: BrowserSession,
) -> ObserverEvidence | None:
    domains = extract_site_keys(task)
    if not domains:
        source_url = _extract_source_url(cells)
        domains = extract_site_keys(source_url) if source_url else []
    if not domains:
        return None

    source_url = _extract_source_url(cells)
    try:
        netlog = await actions.get_network_log(
            browser,
            only_new=False,
            content_type_contains="json",
            max_entries=20,
        )
    except Exception as exc:
        logger.info("Observer netlog 收集失败，继续使用纯执行证据: %s", exc)
        netlog = {"entries": []}

    return ObserverEvidence(
        domain=domains[0],
        execution_log=build_execution_log(cells, items),
        network_summary=build_network_summary(netlog),
        source_url=source_url,
    )


async def maybe_generate_and_store_site_sop(
    *,
    task: str,
    state: CodeAgentState,
    cells: list[CodeCell],
    browser: BrowserSession,
    sop_store: SiteSOPStore,
    cfg: Settings,
) -> SiteSOP | None:
    """在任务成功后生成并写入站点 SOP。"""
    if not cfg.model_name:
        return None
    if not state.done:
        return None
    items = state.items.to_list()
    if not items and not state.final_artifact:
        return None

    domain = infer_observer_domain(task, cells)
    if domain:
        recent_updates = sop_store.recent_accepted_update_count(domain, hours=24)
        if recent_updates >= 2:
            logger.warning("Observer 命中更新熔断，已跳过生成: domain=%s", domain)
            return None

    evidence = await collect_observer_evidence(task=task, cells=cells, items=items, browser=browser)
    if evidence is None:
        return None

    existing = sop_store.get_active_sop(evidence.domain)
    examples = select_observer_examples(evidence.execution_log, evidence.network_summary)
    messages = [
        {
            "role": "system",
            "content": render_template(
                "observer_prompt.jinja2",
                domain=evidence.domain,
                today=_today_text(),
                example_sops="\n\n".join(
                    f"<!-- example: {example.key} -->\n{example.content}" for example in examples
                ),
                existing_sop=existing.sop_markdown if existing else "",
                execution_log=evidence.execution_log,
                network_summary=evidence.network_summary,
                update_reason="success_distillation",
            ),
        }
    ]

    client = LLMClient(cfg)
    try:
        response = await client.complete_with_model(
            messages,
            model_name=cfg.model_name,
            reasoning_effort=cfg.observer_reasoning_effort,
            trace_name="observer_sop_generation",
        )
    except Exception as exc:
        logger.warning("Observer 生成 SOP 失败，已跳过: %s", exc)
        return None

    markdown = response.text.strip()
    if not markdown.startswith("#"):
        logger.warning("Observer 返回非 Markdown SOP，已拒绝写入")
        return None

    validation = validate_browser_harness_style_sop(markdown, evidence.domain)
    if not validation.ok:
        logger.warning("Observer 返回的 SOP 未通过录入校验，已拒绝写入: %s", validation.reason)
        return None

    merged_markdown = smart_merge_sop(existing.sop_markdown, markdown) if existing else markdown
    merged_validation = validate_browser_harness_style_sop(merged_markdown, evidence.domain)
    if not merged_validation.ok:
        logger.warning("Observer 合并后的 SOP 未通过录入校验，已拒绝写入: %s", merged_validation.reason)
        return None

    page_pattern = infer_page_pattern(task, evidence.source_url)
    workflow_kind = classify_observer_evidence(evidence.execution_log, evidence.network_summary)
    should_inspect_first = infer_should_inspect_first(workflow_kind, page_pattern, evidence.execution_log)
    preferred_entry = infer_preferred_entry(workflow_kind, page_pattern)
    field_contract = extract_requested_fields(task)
    confidence = 0.9 if page_pattern else 0.6
    golden_rule = extract_golden_rule(merged_markdown)
    try:
        stored = sop_store.upsert_sop(
            SiteSOP(
                domain=evidence.domain,
                page_pattern=page_pattern,
                workflow_kind=workflow_kind,
                should_inspect_first=should_inspect_first,
                preferred_entry=preferred_entry,
                field_contract=field_contract,
                confidence=confidence,
                sop_markdown=merged_markdown,
                golden_rule=golden_rule,
                update_reason="success_distillation",
                source_run_id=state.run_id,
                source_url=evidence.source_url,
                proof_summary=evidence.execution_log[:400],
                last_generated_at=datetime.now(_BEIJING_TZ).isoformat(),
            )
        )
    except Exception as exc:
        logger.warning("Observer 写入 SOP 失败，已跳过: %s", exc)
        return None

    logger.info(
        "Observer 已写入站点 SOP: domain=%s page_pattern=%s version=%d workflow=%s",
        stored.domain,
        stored.page_pattern,
        stored.version,
        stored.workflow_kind,
    )
    return stored
