"""Observer pipeline for distilling successful runs into site SOPs.

This module turns a finished task run into a compact, reusable site-specific SOP
document. It gathers execution evidence, selects few-shot examples, prompts the
observer model, validates the generated markdown, and persists the result to the
knowledge store.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from hawker_agent.agent.prompts import render_template
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
    """Structured evidence bundle used for SOP generation.

    Attributes:
        domain: Canonical target domain inferred from the task or executed code.
        execution_log: Condensed step history showing successful and failed cells.
        data_access_summary: Compact summary of explicit data access evidence.
        source_url: Best-effort source URL extracted from the run.
    """

    domain: str
    execution_log: str
    data_access_summary: str
    source_url: str


@dataclass
class SOPValidationResult:
    """Validation result for a generated SOP document.

    Attributes:
        ok: Whether the candidate SOP passes the validation checks.
        reason: Human-readable rejection reason when ``ok`` is ``False``.
    """

    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class ObserverExample:
    """Few-shot example used to steer observer generation.

    Attributes:
        key: Stable example identifier.
        title: Human-readable example title.
        content: Full markdown body injected into the observer prompt.
    """

    key: str
    title: str
    content: str


def _today_text() -> str:
    """Returns today's date string in Beijing time."""
    return datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d")


def load_observer_examples() -> dict[str, ObserverExample]:
    """Loads built-in few-shot examples from the template directory.

    Returns:
        A mapping from example key to parsed :class:`ObserverExample`.
    """
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


def classify_observer_evidence(execution_log: str, data_access_summary: str) -> str:
    """Classifies the run into a coarse evidence type.

    The classifier is intentionally heuristic and low-cost. It only needs to
    choose a suitable few-shot neighborhood, not a perfect workflow label.

    Args:
        execution_log: Condensed code execution history.
        data_access_summary: Condensed explicit data access summary.

    Returns:
        One of ``"api_only"``, ``"hybrid"``, or ``"browser_required"``.
    """
    exec_lower = execution_log.lower()
    data_lower = data_access_summary.lower()

    api_signals = sum(
        token in exec_lower or token in data_lower
        for token in ("http_json(", "http_request(", "fetch(", "application/json", "/api/", "explicit_url:")
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


def select_observer_examples(
    execution_log: str,
    data_access_summary: str,
    *,
    max_examples: int = 2,
) -> list[ObserverExample]:
    """Selects the most relevant few-shot examples for the observer prompt.

    Args:
        execution_log: Condensed code execution history.
        data_access_summary: Condensed explicit data access summary.
        max_examples: Maximum number of examples to return.

    Returns:
        Ordered few-shot examples, starting from the best evidence-type match.
    """
    examples = load_observer_examples()
    primary_key = classify_observer_evidence(execution_log, data_access_summary)
    order = [primary_key]
    for fallback in ("hybrid", "api_only", "browser_required"):
        if fallback not in order:
            order.append(fallback)
    return [examples[key] for key in order[:max_examples] if key in examples]


def _extract_source_url(cells: list[CodeCell]) -> str:
    """Extracts the best-effort source URL from executed cells.

    Args:
        cells: Executed code cells in chronological order.

    Returns:
        The most recent explicit cell URL, or the first URL found in source
        code, or an empty string if none can be inferred.
    """
    for cell in reversed(cells):
        if cell.url:
            return cell.url
        urls = extract_urls(cell.source)
        if urls:
            return urls[0]
    return ""


def infer_observer_domain(task: str, cells: list[CodeCell]) -> str:
    """Infers the target domain for observer distillation.

    Args:
        task: Original user task.
        cells: Executed code cells.

    Returns:
        The first inferred domain, or an empty string if no domain can be found.
    """
    domains = extract_site_keys(task)
    if domains:
        return domains[0]
    source_url = _extract_source_url(cells)
    fallback = extract_site_keys(source_url) if source_url else []
    return fallback[0] if fallback else ""


def _format_cell(cell: CodeCell) -> str:
    """Formats a single code cell into a compact observer-facing block."""
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
    """Builds a compact execution log for observer prompting.

    Args:
        cells: Executed code cells.
        items: Structured items collected during the run.
        max_cells: Maximum number of successful cells to include.

    Returns:
        A condensed markdown-like execution trace suitable for prompt injection.
    """
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


def build_data_access_summary(cells: list[CodeCell], items: list[dict[str, Any]]) -> str:
    """Builds a prompt-friendly summary from explicit code and confirmed items."""
    lines: list[str] = []
    url_pattern = re.compile(r"https?://[^\s'\"\\)]+")
    for cell in cells:
        if cell.status != CellStatus.SUCCESS:
            continue
        source = cell.source or ""
        if not any(token in source for token in ("fetch(", "http_json(", "http_request(", "search_web(", "js(")):
            continue
        urls = url_pattern.findall(source)
        tool_hits = [
            token.rstrip("(")
            for token in ("fetch(", "http_json(", "http_request(", "search_web(", "js(")
            if token in source
        ]
        detail = f"- step={cell.step} tools={','.join(tool_hits)}"
        if urls:
            detail += " explicit_url=" + ", ".join(urls[:3])
        if cell.output:
            detail += " output=" + str(cell.output).replace("\n", " ")[:300]
        lines.append(detail)
    if items:
        sample = json.dumps(items[:1], ensure_ascii=False)
        lines.append(f"- confirmed_items={len(items)} sample={sample[:500]}")
    return "\n".join(lines) if lines else "No explicit data access evidence."


def extract_requested_fields(task: str) -> list[str]:
    """Extracts explicitly requested fields from the task text.

    Args:
        task: Original user task.

    Returns:
        Field names found in bullet-like ``- field: ...`` patterns, preserving
        first-seen order and removing case-insensitive duplicates.
    """
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
    """Infers a normalized page pattern from task text or source URL.

    Args:
        task: Original user task.
        source_url: Best-effort URL extracted from the run.

    Returns:
        A normalized page pattern string suitable for SOP matching.
    """
    urls = extract_urls(task)
    if urls:
        return normalize_page_pattern(urls[0])
    return normalize_page_pattern(source_url)


def infer_should_inspect_first(workflow_kind: str, page_pattern: str, execution_log: str) -> bool:
    """Determines whether execution should inspect before extraction.

    Args:
        workflow_kind: Coarse workflow classification.
        page_pattern: Normalized page pattern for the run.
        execution_log: Condensed code execution history.

    Returns:
        ``True`` when a future run should still inspect first, ``False`` when a
        known direct path appears reliable enough to skip initial reconnaissance.
    """
    log_lower = execution_log.lower()
    if workflow_kind == "api_only":
        return False
    if page_pattern == "/trending" and "await js(" in log_lower:
        return False
    if "await js(" in log_lower and "confirmed data sample" in log_lower.lower():
        return False
    return True


def infer_preferred_entry(workflow_kind: str, page_pattern: str) -> str:
    """Infers a preferred entry strategy for future executions.

    Args:
        workflow_kind: Coarse workflow classification.
        page_pattern: Normalized page pattern for the run.

    Returns:
        A short strategy label injected into the stored SOP metadata.
    """
    if workflow_kind == "api_only":
        return "api_direct"
    if page_pattern == "/trending":
        return "nav_summary_then_extract"
    if workflow_kind == "hybrid":
        return "nav_summary_then_explicit_fetch"
    return "inspect_then_extract"


def extract_golden_rule(markdown: str) -> str:
    """Extracts the golden rule line from SOP markdown.

    Args:
        markdown: Generated SOP markdown.

    Returns:
        The extracted golden rule, or a conservative default rule if missing.
    """
    bold_match = re.search(r"\*\*Golden Rule:\*\*\s*(.+)", markdown)
    if bold_match:
        return bold_match.group(1).strip()
    plain_match = re.search(r"Golden Rule:\s*(.+)", markdown)
    if plain_match:
        return plain_match.group(1).strip()
    return "Prefer the shortest verified path supported by live evidence."


def _extract_section(markdown: str, heading: str) -> str:
    """Extracts the body of a level-2 markdown section."""
    pattern = rf"{re.escape(heading)}\n(.*?)(?=\n## |\Z)"
    match = re.search(pattern, markdown, re.S)
    return match.group(1).strip() if match else ""


def _is_placeholder_api_reference(section_text: str) -> bool:
    """Checks whether an API reference section is only placeholder text."""
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
    """Merges unique bullet lines from new and old sections.

    Args:
        old_text: Existing stored section body.
        new_text: Newly generated section body.
        limit: Maximum number of merged bullet lines to keep.

    Returns:
        A newline-separated bullet list with duplicates removed.
    """
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
    """Performs a minimal merge between existing and candidate SOP markdown.

    The merge intentionally stays conservative. It only preserves stronger
    reference sections and merges gotcha bullets to avoid endless appendix
    growth or regression to placeholder content.

    Args:
        existing_markdown: Currently stored SOP markdown.
        candidate_markdown: Newly generated SOP markdown.

    Returns:
        The merged SOP markdown.
    """
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
    """Validates a generated SOP against the minimum storage contract.

    Args:
        markdown: Candidate SOP markdown.
        domain: Expected target domain used in the title.

    Returns:
        A :class:`SOPValidationResult` describing whether the SOP is acceptable.
    """
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
    """Collects evidence required for observer generation.

    Args:
        task: Original user task.
        cells: Executed code cells.
        items: Structured items collected during the run.
        browser: Active browser session. It is not inspected by Observer.

    Returns:
        An :class:`ObserverEvidence` bundle, or ``None`` when no target domain
        can be inferred.
    """
    domains = extract_site_keys(task)
    if not domains:
        source_url = _extract_source_url(cells)
        domains = extract_site_keys(source_url) if source_url else []
    if not domains:
        logger.info("Observer 未能从任务或成功代码中推断站点域名，已跳过 SOP 生成")
        return None

    source_url = _extract_source_url(cells)
    return ObserverEvidence(
        domain=domains[0],
        execution_log=build_execution_log(cells, items),
        data_access_summary=build_data_access_summary(cells, items),
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
    """Generates and stores a site SOP after a successful task run.

    Args:
        task: Original user task.
        state: Mutable run state.
        cells: Executed code cells.
        browser: Active browser session.
        sop_store: SOP persistence layer.
        cfg: Runtime settings.

    Returns:
        The stored :class:`SiteSOP` on success, otherwise ``None`` when SOP
        generation is skipped, rejected, or fails.
    """
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
    stored = await generate_and_store_site_sop_from_evidence(
        evidence=evidence,
        existing_sop_markdown=existing.sop_markdown if existing else "",
        sop_store=sop_store,
        cfg=cfg,
        source_run_id=state.run_id,
        task=task,
    )
    return stored


async def generate_and_store_site_sop_from_evidence(
    *,
    evidence: ObserverEvidence,
    existing_sop_markdown: str,
    sop_store: SiteSOPStore,
    cfg: Settings,
    source_run_id: str,
    task: str = "",
) -> SiteSOP | None:
    """Generates and stores a site SOP from pre-collected evidence.

    This variant is suitable for background execution after the user-facing
    result has already been returned. It does not depend on a live browser
    session or mutable run state.

    Args:
        evidence: Pre-collected observer evidence bundle.
        existing_sop_markdown: Current active SOP markdown for the domain.
        sop_store: SOP persistence layer.
        cfg: Runtime settings.
        source_run_id: Run identifier used for lineage metadata.
        task: Original task text when available. Used to enrich metadata such as
            page pattern and requested field contract.

    Returns:
        The stored :class:`SiteSOP` on success, otherwise ``None`` when SOP
        generation is rejected or fails.
    """
    examples = select_observer_examples(evidence.execution_log, evidence.data_access_summary)
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
                existing_sop=existing_sop_markdown,
                execution_log=evidence.execution_log,
                data_access_summary=evidence.data_access_summary,
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

    merged_markdown = smart_merge_sop(existing_sop_markdown, markdown) if existing_sop_markdown else markdown
    merged_validation = validate_browser_harness_style_sop(merged_markdown, evidence.domain)
    if not merged_validation.ok:
        logger.warning("Observer 合并后的 SOP 未通过录入校验，已拒绝写入: %s", merged_validation.reason)
        return None

    page_pattern = infer_page_pattern(task, evidence.source_url)
    workflow_kind = classify_observer_evidence(evidence.execution_log, evidence.data_access_summary)
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
                source_run_id=source_run_id,
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
