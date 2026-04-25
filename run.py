import asyncio
import sys
from pathlib import Path

# 确保可以导入当前目录下的 hawker_agent
sys.path.append(str(Path(__file__).parent))

from hawker_agent import run
from hawker_agent.config import get_settings
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner

# =================================================================
# 📝 在这里编写你的任务描述 (支持多行)
# =================================================================


TASK="""
1. 检索web agent paper 获取前5条论文
2. 返回前5条论文标题，下载链接（可下载的pdf链接），摘要，研究领域
3. 并下载这5个文论到本地
"""

TASK="""
1. 维基百科分别搜索 OpenAI、APPLE 二家公司
2. 从文章中提取以下内容：名称、成立日期、总部、现任首席执行官/领导者、所属行业、主要产品/服务（列出）、收入（如有则提供）、员工人数（如有则提供），以及 2-3 句概括
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
"""


TASK="""
步骤1: 打开 https://www.ahnews.com.cn/df/hss/pc/lay/node_525.html,点击"下一页"，获取3页数据,获取列表页URL和title 
"""

TASK="""
步骤1: 打开网址 https://mcp.aibase.com/zh/explore 
步骤2: 分类点击 "搜索工具",认证状态点击“不限”，编程语言是“python”，类型 “MCP Server”，点击按“按下载量”排序
步骤3: 提取所有项目名称、简介、url
"""


TASK="""
1.
获取下面里提到的的论文，获取下载链接，摘要，研究领域,题号,返回json，并下载论文
[87] Jae-Woo Choi, Hyungmin Kim, Hyobin Ong, Youngwoo Yoon, Minsu Jang, Jaehong Kim, et al.
Reactree: Hierarchical task planning with dynamic tree expansion using llm agent nodes. 2025.
[88] Siddharth Nayak, Adelmo Morrison Orozco, Marina Ten Have, Vittal Thirumalai, Jackson
Zhang, Darren Chen, Aditya Kapoor, Eric Robinson, Karthik Gopalakrishnan, James Harrison, et al. LLaMAR: Long-horizon planning for multi-agent robots in partially observable
environments. arXiv preprint arXiv:2407.10031, 2024.
"""


TASK="""
去 https://x.com/sama 查看这些账号最新10条动态，用中文总结一下，有AI有什么新的洞察
"""

TASK="""
步骤1: 获取 https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix 仓库信息,提取这个仓库所有的文件名和下载URL
"""

TASK="看看 iPhone 17 在中国的价格"

TASK="""
1.打开 https://github.com/trending
2.获取当前页面的项目URL、start、fork、today_start
提取字段: 
- URL: 项目链接 
- start: start数 
- fork： fork数
- today_start: today_start数
"""



TASK="""在arxiv中 搜索关于 "RNA" 的 5 篇论文，返回标题、链接、摘要、研究领域，并下载论文，上传obs 路径:hawker/paper"""

TASK="""
1. 打开 https://arxiv.org/search/advanced
2. 搜素标题为 web agent 论文，查找2026年2月12到2026年4月30之间的论文
3. 返回该条件下所有论文的链接、标题、摘要、发表时间，json格式
"""

# =================================================================

async def main():
    console = Console()
    cfg = get_settings()

    # 简单检查
    if not cfg.openai_api_key or "sk-xxxx" in cfg.openai_api_key:
        console.print("[red]❌ 错误: 请在 .env 文件中设置有效的 OPENAI_API_KEY[/red]")
        return

    console.print(Panel(
        f"[bold green]🚀 正在启动 HawkerAgent[/bold green]\n"
        f"[dim]模型: {cfg.model_name}[/dim]\n\n"
        f"[bold blue]当前任务:[/bold blue]\n{TASK.strip()}",
        title="HawkerAgent Runner",
        expand=False
    ))

    # 使用 Live 效果展示运行状态
    with Live(Spinner("dots", text="[yellow] Agent 正在思考并执行中...[/yellow]"), refresh_per_second=10, console=console):
        try:
            # 调用重构后的核心入口
            result = await run(TASK, max_steps=cfg.max_steps)
        except Exception as e:
            console.print(f"\n[red]💥 运行过程中发生崩溃: {e}[/red]")
            import traceback
            console.print(traceback.format_exc())
            return

    # 运行结束展示
    status_color = "green" if result.success else "yellow"
    console.print(f"\n[bold {status_color}]✨ 任务执行完毕 ({result.stop_reason})[/bold {status_color}]")
    console.print(Panel(result.answer, title="最终回答", border_style=status_color))
    
    # 耗时格式化
    m, s = divmod(int(result.total_duration), 60)
    duration_str = f"{m}分{s}秒" if m > 0 else f"{s}秒"

    console.print(
        f"\n📊 [bold]运行统计汇总:[/bold]\n"
        f"  - 任务结果: [bold {status_color}]{'成功' if result.success else '未完全完成'}[/bold {status_color}]\n"
        f"  - 使用模型: [dim]{result.model_name}[/dim]\n"
        f"  - 采集数据: [cyan]{result.items_count}[/cyan] 条\n"
        f"  - 迭代步数: [cyan]{result.total_steps}[/cyan] 步\n"
        f"  - 总计耗时: [bold cyan]{duration_str}[/bold cyan] ({result.total_duration:.1f}s)\n"
        f"  - 消耗费用: [cyan]${result.token_stats.cost:.4f}[/cyan]\n"
        f"  - 消耗 Token: [dim]{result.token_stats.total_tokens:,}[/dim]\n"
        f"  - 运行产物: [underline]{result.run_dir}[/underline]"
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] 用户中断")
