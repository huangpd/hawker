from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置，替换原文件顶部的 os.getenv() 全局常量。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    openai_api_key: str
    openai_base_url: str | None = None
    model_name: str
    reasoning_effort: str = ""

    # 预算控制
    max_total_tokens: int = 120_000
    max_no_progress_steps: int = 10
    message_compression_tokens: int = 12_000

    # 文件系统
    scrape_dir: Path = Path("crawler_agent")

    # 浏览器
    headless: bool = False

    # 日志
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """延迟加载配置单例，避免 import 时因缺少环境变量而失败。"""
    return Settings()
