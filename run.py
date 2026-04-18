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
1. 打开 https://arxiv.org/search/advanced
2. 搜素标题为 web agent 论文，查找2026年1月12到2026年3月30之间的论文
3. 返回前3篇论文的链接、标题、摘要、发表时间，然后下载本地
"""




TASK="""
1. 检索web agent paper 获取前5条论文
2. 返回前5条论文标题，下载链接（可下载的pdf链接），摘要，研究领域
3. 并下载这5个文论到本地
"""
TASK="""
步骤1: 打开 https://www.ahnews.com.cn/df/hss/pc/lay/node_525.html
步骤2: 点击"下一页"，获取3页数据
步骤3: 获取列表页URL和title 
"""


TASK="""
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
"""


TASK="""
步骤1: 打开 https://www.ahnews.com.cn/df/hss/pc/lay/node_525.html,点击"下一页"，获取3页数据,获取列表页URL和title 
"""
TASK="""
1.打开 https://github.com/trending
2.获取当前页面的项目URL、start、fork、today_start，获取第一页直接返回
提取字段: 
- URL: 项目链接 
- start: start数 
- fork： fork数
- today_start: today_start数
"""
TASK="""
步骤1: 打开网址 https://mcp.aibase.com/zh/explore 
步骤2: 找到下一页按钮，获取前10页数据 ，分类点击 "搜索工具",认证状态点击“不限”，编程语言是“python”，类型 “MCP Server”，点击按“按下载量”排序
步骤3:提取所有项目名称、简介、url
"""


TASK='''
1.打开 https://kns.cnki.net/kns8s/AdvSearch，点击专业检索
2.检索词：SU%=航空+航天+飞行器设计+航空发动机+航天器+航空器+卫星+卫星遥感+航空遥感+航天遥感+飞行试验+航空无人机+航天无人机+空中交通+航空安全+航空电子+航空制造 and SU%=数据+数据科学+大数据+人工智能+机器学习+深度学习+数据挖掘+数据分析+数据融合+联邦学习+预测性维护+数据驱动+统计+自然语言处理+计算机视觉
3.点击搜索
4.需要爬取的字段：
文献名称:
作者:
期刊页链接:
发布时间:
引用量:
下载量:
期刊影响因子
摘要:
关键词:
发布单位：
6.爬10页
'''

TASK="""
1.打开 https://github.com/trending
2.获取当前页面的项目URL、start、fork、today_start，获取数据就返回
提取字段: 
- URL: 项目链接 
- start: start数 
- fork： fork数
- today_start: today_start数
"""
TASK="""
步骤1: 打开 https://www.ahnews.com.cn/df/hss/pc/lay/node_525.html,点击"下一页"，获取30页数据,获取列表页URL和title 
"""
TASK="""
1.获取content里提到的的论文，论文下载链接（缺少部分应该网络搜索补全，最好pdf），摘要，研究领域,返回json
content='''
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
'''
"""

TASK="""
去 https://x.com/gregpr07(最好从google跳转，不然容易风控)  查看这个人最新10条动态，用中文总结一下，让我知道他最近的在干什么，有什么新的洞察
"""
TASK="""
去 https://x.com/OpenAI，https://x.com/sama，https://x.com/claudeai，https://x.com/bcherny  查看这些账号最新10条动态，用中文总结一下，有AI有什么新的洞察
"""

TASK="""
步骤1: 获取 https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix 仓库信息
步骤5：提取这个仓库所有的文件名和下载URL
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
