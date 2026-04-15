from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console
from rich.panel import Panel

from hawker_agent.agent.runner import run
from hawker_agent.config import get_settings

app = typer.Typer(help="HawkerAgent — LLM 驱动的自主网络爬虫")
console = Console()


@app.command()
def main(
    task: str = typer.Argument(..., help="描述你要爬取的任务"),
    max_steps: int = typer.Option(25, "--max-steps", "-s", help="最大允许迭代步数"),
) -> None:
    """运行 HawkerAgent 爬取任务。"""
    cfg = get_settings()
    if not cfg.openai_api_key:
        console.print("[red]错误: OPENAI_API_KEY 未设置。请检查 .env 文件。[/red]")
        sys.exit(1)

    console.print(Panel(f"[bold blue]任务开始:[/bold blue]\n{task}", title="HawkerAgent", expand=False))

    try:
        result = asyncio.run(run(task, max_steps=max_steps))
        
        # 结果输出
        status_color = "green" if result.success else "yellow"
        console.print(f"\n[bold {status_color}]运行结束 ({result.stop_reason}):[/bold {status_color}]")
        console.print(Panel(result.answer, title="回答", border_style=status_color))
        
        # 耗时格式化
        m, s = divmod(int(result.total_duration), 60)
        duration_str = f"{m}分{s}秒" if m > 0 else f"{s}秒"

        summary = (
            f"📊 [bold]统计汇总:[/bold]\n"
            f"  - 任务状态: [bold {status_color}]{result.stop_reason}[/bold {status_color}]\n"
            f"  - 采集数据: [cyan]{result.items_count}[/cyan] 条\n"
            f"  - 迭代步数: [cyan]{result.total_steps}[/cyan] 步\n"
            f"  - 总计耗时: [bold cyan]{duration_str}[/bold cyan] ({result.total_duration:.1f}s)\n"
            f"  - 消耗费用: [cyan]${result.token_stats.cost:.4f}[/cyan]\n"
            f"  - Token 统计: [dim]{result.token_stats.total_tokens:,} (in:{result.token_stats.input_tokens:,} out:{result.token_stats.output_tokens:,} cache:{result.token_stats.cached_tokens:,})[/dim]"
        )
        console.print(summary)
        
        if result.run_dir:
            console.print(f"\n📁 运行产物保存至: [underline]{result.run_dir}[/underline]")
            if result.notebook_path:
                console.print(f"📓 Notebook: {result.notebook_path.relative_to(result.run_dir)}")
            if result.result_json_path:
                console.print(f"📊 JSON 结果: {result.result_json_path.relative_to(result.run_dir)}")
            if result.llm_io_path:
                console.print(f"🧠 LLM I/O: {result.llm_io_path.relative_to(result.run_dir)}")

    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断运行。[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]运行出错: {e}[/red]")
        logger = sys.modules.get("logging")
        if logger:
            logger.getLogger("hawker_agent.cli").exception("CLI 异常")
        sys.exit(1)


if __name__ == "__main__":
    app()
