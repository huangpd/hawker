from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table

from hawker_agent import run
from hawker_agent.config import get_settings


@dataclass(frozen=True)
class CaseDef:
    case_id: str
    name: str
    task: str
    expected_total: int | None = None


@dataclass
class CaseResultRow:
    case_id: str
    name: str
    status: str
    steps: int
    duration: float
    tokens: int
    cost: float
    crawl_data: int
    total_data: int | None
    percent: str
    run_dir: str
    stop_reason: str
    answer: str


CASES: list[CaseDef] = [
    CaseDef(
        case_id="case1",
        name="HuggingFace Dataset Files",
        expected_total=220,
        task="""
1. 打开网址 https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix
2. 找到并点击 "Files and versions" 标签页
3. 下滑页面到最底部，加载更多 "Load more files"
4. 遍历所有子文件夹
5. 提取页面的文件名和下载URL
样本数据:
{"file_name":".gitattributes","download_url":"https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix/resolve/main/.gitattributes?download=true"}
""".strip(),
    ),
    CaseDef(
        case_id="case2",
        name="AHNews List Pages",
        expected_total=72,
        task="""
1. 打开 https://www.ahnews.com.cn/df/hss/pc/lay/node_525.html
2. 点击"下一页"，获取3页数据
3. 获取列表页URL和title
样本数据:
{"title":"专人守护、一树一策 黄山多措并举保护古松树","URL":"http://www.ahnews.com.cn/anhui/pc/con/2026-01/23/562_1662432.html"}
""".strip(),
    ),
    CaseDef(
        case_id="case3",
        name="MCP AIBase Explore",
        expected_total=343,
        task="""
1. 打开网址 https://mcp.aibase.com/zh/explore
2. 找到下一页按钮，获取前10页数据 ，分类点击 "搜索工具",认证状态点击"不限"，编程语言是"python"，类型 "MCP Server"，点击按"按下载量"排序
3. 提取所有name、desc(简介)、url
样本数据:
{"name":"Klavis","desc":"Klavis AI是一个开源项目，提供在Slack、Discord和Web平台上简单易用的MCP（模型上下文协议）服务，包括报告生成、YouTube工具、文档转换等多种功能，支持非技术用户和开发者使用AI工作流。","url":"https://mcp.aibase.com/zh/server/1528363509283561529"}
""".strip(),
    ),
    CaseDef(
        case_id="case4",
        name="Wikipedia Research",
        expected_total=2,
        task="""
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
    CaseDef(
        case_id="case5",
        name="GitHub Trending",
        expected_total=14,
        task="""
1. 打开 https://github.com/trending
2. 获取当前页面的项目URL、start、fork、today_start
提取字段:
- URL: 项目链接
- start: start数
- fork： fork数
- today_start: today_start数
""".strip(),
    ),
    CaseDef(
        case_id="case6",
        name="arXiv Paper Search",
        expected_total=57,
        task="""
1. 打开 https://arxiv.org/search/advanced
2. 搜素标题 web agent 论文，查找2026年1月14到2026年3月20之间的
3. 如果有"Next",获取下一页
4. 返回该条件下论文的下载链接(pdf格式)
""".strip(),
    ),
    CaseDef(
        case_id="case7",
        name="论文引用附录",
        expected_total=7,
        task="""
从以下content引用中找每篇论文的下载链接并下载PDF到本地，同时返回每篇论文的摘要和研究领域、下载链接、编号[81]返回JSON。
1. content='''
[86] Lutfi Eren Erdogan, Hiroki Furuta, Sehoon Kim, Nicholas Lee, Suhong Moon, Gopala Anumanchipalli, Kurt Keutzer, and Amir Gholami. Plan-and-Act: Improving planning of agents
for long-horizon tasks. In Forty-second International Conference on Machine Learning, 2025.
URL https://openreview.net/forum?id=ybA4EcMmUZ.
[87] Jae-Woo Choi, Hyungmin Kim, Hyobin Ong, Youngwoo Yoon, Minsu Jang, Jaehong Kim, et al.
Reactree: Hierarchical task planning with dynamic tree expansion using llm agent nodes. 2025.
[88] Siddharth Nayak, Adelmo Morrison Orozco, Marina Ten Have, Vittal Thirumalai, Jackson
Zhang, Darren Chen, Aditya Kapoor, Eric Robinson, Karthik Gopalakrishnan, James Harrison, et al. LLaMAR: Long-horizon planning for multi-agent robots in partially observable
environments. arXiv preprint arXiv:2407.10031, 2024.
[89] Anthropic. Model Context Protocol (MCP), 2024. URL https://www.anthropic.com/news
/model-context-protocol.
[90] Yingxuan Yang, Huacan Chai, Yuanyi Song, Siyuan Qi, Muning Wen, Ning Li, Junwei Liao,
Haoyi Hu, Jianghao Lin, Gaowei Chang, Weiwen Liu, Ying Wen, Yong Yu, and Weinan Zhang.
A survey of AI agent protocols, 2025. URL https://arxiv.org/abs/2504.16736.
[91] Yu Wang and Xi Chen. Mirix: Multi-agent memory system for llm-based agents, 2025. URL
https://arxiv.org/abs/2507.07957.
[92] Kai Mei, Xi Zhu, Wujiang Xu, Wenyue Hua, Mingyu Jin, Zelong Li, Shuyuan Xu, Ruosong
Ye, Yingqiang Ge, and Yongfeng Zhang. Aios: Llm agent operating system. arXiv preprint
arXiv:2403.16971, 2024.
'''
""".strip(),
    ),
    CaseDef(
        case_id="case8",
        name="Google Search Thesis",
        expected_total=5,
        task="""
1. 打开google搜索，检索web agent 获取5篇高质量论文，论文标题，下载链接（pdf链接），摘要，研究领域等，返回json
""".strip(),
    ),
]


def _find_cases(case_ids: list[str] | None) -> list[CaseDef]:
    if not case_ids:
        return CASES
    wanted = {case_id.strip() for case_id in case_ids}
    return [case for case in CASES if case.case_id in wanted]


def _format_percent(crawl_data: int, total_data: int | None) -> str:
    if not total_data or total_data <= 0:
        return "-"
    percent = round(crawl_data / total_data * 100)
    return f"{percent}%"


def _build_table(rows: list[CaseResultRow]) -> Table:
    table = Table(title="Case Run Summary")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Steps", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("$", justify="right")
    table.add_column("crawl_Data", justify="right")
    table.add_column("total_data", justify="right")
    table.add_column("percent", justify="right")

    for row in rows:
        table.add_row(
            row.case_id,
            row.name,
            row.status,
            str(row.steps),
            f"{row.duration:.1f}s",
            f"{row.tokens:,}",
            f"${row.cost:.4f}",
            str(row.crawl_data),
            str(row.total_data or "-"),
            row.percent,
        )
    return table


def _save_report(run_dir: Path, rows: list[CaseResultRow]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    (run_dir / "case_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "| ID | Name | Status | Steps | Duration | Tokens | $ | crawl_Data | total_data | percent |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.case_id} | {row.name} | {row.status} | {row.steps} | {row.duration:.1f}s | "
            f"{row.tokens:,} | ${row.cost:.4f} | "
            f"{row.crawl_data} | {row.total_data or '-'} | {row.percent} |"
        )
    (run_dir / "case_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_report_subdir(selected: list[CaseDef]) -> str:
    if len(selected) == 1:
        return selected[0].case_id
    joined = "_".join(case.case_id for case in selected)
    return f"multi_{joined}"


async def _run_one(case: CaseDef, max_steps: int, console: Console) -> CaseResultRow:
    console.rule(f"[bold cyan]{case.case_id}[/bold cyan] {case.name}")
    started_at = time.time()
    try:
        result = await run(case.task, max_steps=max_steps)
        crawl_data = result.items_count
        status = "PASS" if result.success else "FAIL"
        row = CaseResultRow(
            case_id=case.case_id,
            name=case.name,
            status=status,
            steps=result.total_steps,
            duration=result.total_duration,
            tokens=result.token_stats.total_tokens,
            cost=result.token_stats.cost,
            crawl_data=crawl_data,
            total_data=case.expected_total,
            percent=_format_percent(crawl_data, case.expected_total),
            run_dir=str(result.run_dir or ""),
            stop_reason=result.stop_reason,
            answer=result.answer,
        )
        console.print(
            f"[green]{case.case_id} 完成[/green] "
            f"status={row.status} steps={row.steps} items={row.crawl_data} "
            f"duration={row.duration:.1f}s tokens={row.tokens:,}"
        )
        return row
    except Exception as exc:
        duration = time.time() - started_at
        console.print(f"[red]{case.case_id} 崩溃[/red] {exc}")
        return CaseResultRow(
            case_id=case.case_id,
            name=case.name,
            status="ERROR",
            steps=0,
            duration=duration,
            tokens=0,
            cost=0.0,
            crawl_data=0,
            total_data=case.expected_total,
            percent=_format_percent(0, case.expected_total),
            run_dir="",
            stop_reason="exception",
            answer=str(exc),
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hawker regression cases.")
    parser.add_argument("--case", action="append", dest="cases", help="Run specific case id, e.g. --case case3")
    parser.add_argument("--max-steps", type=int, default=20, help="Max steps for each case")
    parser.add_argument(
        "--report-dir",
        default="case_reports",
        help="Directory to save markdown/json summary",
    )
    args = parser.parse_args()

    console = Console()
    cfg = get_settings()
    if not cfg.openai_api_key or "sk-xxxx" in cfg.openai_api_key:
        console.print("[red]❌ 错误: 请在 .env 文件中设置有效的 OPENAI_API_KEY[/red]")
        return

    selected = _find_cases(args.cases)
    if not selected:
        console.print("[red]没有匹配到任何 case[/red]")
        return

    rows: list[CaseResultRow] = []
    for case in selected:
        row = await _run_one(case, args.max_steps, console)
        rows.append(row)

    table = _build_table(rows)
    console.print()
    console.print(table)

    report_dir = Path(args.report_dir)
    report_subdir = _build_report_subdir(selected)
    output_dir = report_dir / report_subdir
    _save_report(output_dir, rows)
    console.print(f"\n[bold green]报告已保存[/bold green]: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
