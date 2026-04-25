from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hawker_agent.agent.runner import run
from hawker_agent.config import get_settings
from hawker_agent.models.result import CodeAgentResult


TASKS: list[tuple[str, str]] = [
    (
        "web_agent_papers_top5",
        """
1. 检索web agent paper 获取前5条论文
2. 返回前5条论文标题，下载链接（可下载的pdf链接），摘要，研究领域
3. 并下载这5个文论到本地
""".strip(),
    ),
    (
        "wikipedia_company_profiles",
        """
1. 访问维基百科网站：https://en.wikipedia.org
2. 分别搜索 OpenAI、APPLE 二家公司
3. 打开维基百科的相关文章
4. 从信息框和文章中提取以下内容：名称、成立日期、总部、现任首席执行官/领导者、所属行业、主要产品/服务（列出）、收入（如有则提供）、员工人数（如有则提供），以及 2-3 句概括
提取字段:
- name: "公司名称"
- founded: "成立时间"
- headquarters: "总部地址"
- ceo: "现任 CEO"
- industry: "所属行业"
- products: "主要产品/服务列表"
- revenue: "营收（如有）"
- employees: "员工数量（如有）"
- summary: "公司简介（2-3句话）"
""".strip(),
    ),
    (
        "ahnews_hss_three_pages",
        '步骤1: 打开 https://www.ahnews.com.cn/df/hss/pc/lay/node_525.html,点击"下一页"，获取3页数据,获取列表页URL和title',
    ),
    (
        "mcp_aibase_search_tools_python",
        """
步骤1: 打开网址 https://mcp.aibase.com/zh/explore
步骤2: 找到下一页按钮，获取前10页数据 ，分类点击 "搜索工具",认证状态点击“不限”，编程语言是“python”，类型 “MCP Server”，点击按“按下载量”排序
步骤3:提取所有项目名称、简介、url
""".strip(),
    ),
    (
        "specific_reactree_llamar_papers",
        """
1.
获取下面里提到的的论文，获取下载链接，摘要，研究领域,并下载论文
[87] Jae-Woo Choi, Hyungmin Kim, Hyobin Ong, Youngwoo Yoon, Minsu Jang, Jaehong Kim, et al.
Reactree: Hierarchical task planning with dynamic tree expansion using llm agent nodes. 2025.
[88] Siddharth Nayak, Adelmo Morrison Orozco, Marina Ten Have, Vittal Thirumalai, Jackson
Zhang, Darren Chen, Aditya Kapoor, Eric Robinson, Karthik Gopalakrishnan, James Harrison, et al. LLaMAR: Long-horizon planning for multi-agent robots in partially observable
environments. arXiv preprint arXiv:2407.10031, 2024.
'''
""".strip(),
    ),
    (
        "x_openai_sama_latest",
        "去 https://x.com/OpenAI，https://x.com/sama 查看这些账号最新10条动态，用中文总结一下，有AI有什么新的洞察",
    ),
    (
        "huggingface_nemotron_climbmix_files",
        "步骤1: 获取 https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix 仓库信息,提取这个仓库所有的文件名和下载URL",
    ),
    ("iphone_17_china_price", "看看 iPhone 17 在中国的价格"),
    (
        "github_trending",
        """
1.打开 https://github.com/trending
2.获取当前页面的项目URL、start、fork、today_start
提取字段:
- URL: 项目链接
- start: start数
- fork： fork数
- today_start: today_start数
""".strip(),
    ),
    (
        "arxiv_llm_agent_planning_15",
        '打开 https://arxiv.org/search/advanced 搜索关于 "LLM agent planning" 的 15 篇论文，返回标题、链接、摘要、研究领域，并下载论文',
    ),
    (
        "arxiv_web_agent_date_range",
        """
1. 打开 https://arxiv.org/search/advanced
2. 搜素标题为 web agent 论文，查找2026年2月12到2026年4月30之间的论文
3. 返回该条件下所有论文的链接、标题、摘要、发表时间，json格式
""".strip(),
    ),
]


@dataclass
class FileRecord:
    path: str
    size: int
    role: str


@dataclass
class TaskRunSummary:
    index: int
    name: str
    task: str
    success: bool
    stop_reason: str
    items_count: int
    total_steps: int
    duration_s: float
    cost: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    run_id: str = ""
    run_dir: str = ""
    notebook_path: str = ""
    result_json_path: str = ""
    llm_io_path: str = ""
    answer_preview: str = ""
    error: str = ""
    files: list[FileRecord] = field(default_factory=list)


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}分{secs}秒" if minutes else f"{secs}秒"


def _file_role(path: Path, result: CodeAgentResult) -> str:
    resolved = path.resolve()
    if result.notebook_path and resolved == result.notebook_path.resolve():
        return "notebook"
    if result.result_json_path and resolved == result.result_json_path.resolve():
        return "result_json"
    if result.llm_io_path and resolved == result.llm_io_path.resolve():
        return "llm_io"
    if path.suffix.lower() in {".pdf", ".csv", ".jsonl", ".parquet", ".zip"}:
        return "download_or_data"
    if "result" in path.parts:
        return "result_artifact"
    return "artifact"


def _collect_files(result: CodeAgentResult) -> list[FileRecord]:
    if not result.run_dir or not result.run_dir.exists():
        return []
    records: list[FileRecord] = []
    for path in sorted(result.run_dir.rglob("*")):
        if not path.is_file():
            continue
        records.append(
            FileRecord(
                path=str(path),
                size=path.stat().st_size,
                role=_file_role(path, result),
            )
        )
    return records


def _summarize_result(index: int, name: str, task: str, result: CodeAgentResult) -> TaskRunSummary:
    stats = result.token_stats
    return TaskRunSummary(
        index=index,
        name=name,
        task=task,
        success=result.success,
        stop_reason=result.stop_reason,
        items_count=result.items_count,
        total_steps=result.total_steps,
        duration_s=round(result.total_duration, 3),
        cost=round(stats.cost, 6),
        total_tokens=stats.total_tokens,
        input_tokens=stats.input_tokens,
        output_tokens=stats.output_tokens,
        cached_tokens=stats.cached_tokens,
        run_id=result.run_id,
        run_dir=str(result.run_dir or ""),
        notebook_path=str(result.notebook_path or ""),
        result_json_path=str(result.result_json_path or ""),
        llm_io_path=str(result.llm_io_path or ""),
        answer_preview=(result.answer or "").replace("\n", " ")[:300],
        files=_collect_files(result),
    )


def _summarize_error(index: int, name: str, task: str, exc: BaseException, started_at: float) -> TaskRunSummary:
    return TaskRunSummary(
        index=index,
        name=name,
        task=task,
        success=False,
        stop_reason="crashed",
        items_count=0,
        total_steps=0,
        duration_s=round(time.time() - started_at, 3),
        cost=0.0,
        total_tokens=0,
        input_tokens=0,
        output_tokens=0,
        cached_tokens=0,
        error=f"{type(exc).__name__}: {exc}",
    )


def _select_tasks(args: argparse.Namespace) -> list[tuple[int, str, str]]:
    selected = [(idx, name, task) for idx, (name, task) in enumerate(TASKS, start=1)]
    if args.only:
        wanted = {int(part) for part in args.only.split(",") if part.strip()}
        selected = [item for item in selected if item[0] in wanted]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def _write_reports(output_dir: Path, summaries: list[TaskRunSummary]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks_count": len(summaries),
        "totals": {
            "duration_s": round(sum(item.duration_s for item in summaries), 3),
            "cost": round(sum(item.cost for item in summaries), 6),
            "total_tokens": sum(item.total_tokens for item in summaries),
            "items_count": sum(item.items_count for item in summaries),
        },
        "runs": [asdict(item) for item in summaries],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_render_markdown(summaries, payload["totals"]), encoding="utf-8")


def _render_markdown(summaries: list[TaskRunSummary], totals: dict[str, Any]) -> str:
    lines = [
        "# Hawker Task Suite Summary",
        "",
        f"- tasks: {len(summaries)}",
        f"- total_items: {totals['items_count']}",
        f"- total_duration: {_format_duration(float(totals['duration_s']))} ({totals['duration_s']}s)",
        f"- total_cost: ${totals['cost']:.4f}",
        f"- total_tokens: {totals['total_tokens']:,}",
        "",
        "| # | task | status | items | steps | duration | cost | tokens | run_dir |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summaries:
        lines.append(
            "| {index} | {name} | {status} | {items} | {steps} | {duration} | ${cost:.4f} | {tokens:,} | {run_dir} |".format(
                index=item.index,
                name=item.name,
                status=item.stop_reason,
                items=item.items_count,
                steps=item.total_steps,
                duration=_format_duration(item.duration_s),
                cost=item.cost,
                tokens=item.total_tokens,
                run_dir=item.run_dir,
            )
        )
    lines.append("")
    for item in summaries:
        lines.extend([
            f"## {item.index}. {item.name}",
            "",
            f"- 任务状态: {item.stop_reason}",
            f"- 采集数据: {item.items_count} 条",
            f"- 迭代步数: {item.total_steps} 步",
            f"- 总计耗时: {_format_duration(item.duration_s)} ({item.duration_s:.1f}s)",
            f"- 消耗费用: ${item.cost:.4f}",
            f"- Token 统计: {item.total_tokens:,} (in:{item.input_tokens:,} out:{item.output_tokens:,} cache:{item.cached_tokens:,})",
            f"- run_dir: {item.run_dir or '(none)'}",
            f"- result_json: {item.result_json_path or '(none)'}",
            f"- notebook: {item.notebook_path or '(none)'}",
            f"- llm_io: {item.llm_io_path or '(none)'}",
        ])
        if item.error:
            lines.append(f"- error: {item.error}")
        lines.append("- 具体文件:")
        if item.files:
            for record in item.files:
                lines.append(f"  - [{record.role}] {record.path} ({record.size} bytes)")
        else:
            lines.append("  - (none)")
        if item.answer_preview:
            lines.extend(["", f"> {item.answer_preview}"])
        lines.append("")
    return "\n".join(lines)


async def _run_suite(args: argparse.Namespace) -> list[TaskRunSummary]:
    cfg = get_settings()
    max_steps = args.max_steps or cfg.max_steps
    summaries: list[TaskRunSummary] = []
    selected = _select_tasks(args)

    for ordinal, (index, name, task) in enumerate(selected, start=1):
        print(f"\n[{ordinal}/{len(selected)}] START task#{index} {name}", flush=True)
        started_at = time.time()
        try:
            result = await run(task, max_steps=max_steps)
            summary = _summarize_result(index, name, task, result)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            summary = _summarize_error(index, name, task, exc, started_at)
            if not args.continue_on_error:
                summaries.append(summary)
                _write_reports(args.output_dir, summaries)
                raise
        summaries.append(summary)
        print(
            "[{idx}] {name}: status={status} items={items} steps={steps} "
            "duration={duration:.1f}s cost=${cost:.4f} tokens={tokens:,} run_dir={run_dir}".format(
                idx=index,
                name=name,
                status=summary.stop_reason,
                items=summary.items_count,
                steps=summary.total_steps,
                duration=summary.duration_s,
                cost=summary.cost,
                tokens=summary.total_tokens,
                run_dir=summary.run_dir or "(none)",
            ),
            flush=True,
        )
        _write_reports(args.output_dir, summaries)
    return summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hawker benchmark task suite sequentially.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override MAX_STEPS for every task.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected tasks.")
    parser.add_argument("--only", default="", help="Comma-separated 1-based task indexes, e.g. 1,3,8.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "hawker_file" / "task_suite",
        help="Directory for summary.json and summary.md.",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="Continue when one task crashes.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected tasks without running them.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected = _select_tasks(args)
    if args.dry_run:
        for index, name, task in selected:
            print(f"{index}. {name}: {task.splitlines()[0][:120]}")
        return
    summaries = asyncio.run(_run_suite(args))
    _write_reports(args.output_dir, summaries)
    print(f"\nSummary JSON: {args.output_dir / 'summary.json'}")
    print(f"Summary Markdown: {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
