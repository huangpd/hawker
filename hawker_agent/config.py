from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用程序配置类，替代 os.getenv() 全局常量。

    该类使用 pydantic-settings 从环境变量和可选的 .env 文件中加载配置。

    Attributes:
        openai_api_key (str): OpenAI 兼容 LLM 服务的 API 密钥。
        openai_base_url (str | None): LLM API 的基础 URL。默认为 None。
        model_name (str): 要使用的 LLM 模型名称。
        reasoning_effort (str): 推理模型的努力程度。默认为 ""。
        langfuse_public_key (str | None): Langfuse 公钥。默认为 None。
        langfuse_secret_key (str | None): Langfuse 私钥。默认为 None。
        langfuse_base_url (str | None): Langfuse 服务地址。默认为 None。
        langfuse_environment (str): Langfuse 环境标签。默认为 "development"。
        langfuse_release (str): Langfuse 版本标签。默认为 ""。
        max_total_tokens (int): 任务允许的最大总 token 数。默认为 120,000。
        max_no_progress_steps (int): 停止前允许的最大无进展步数。默认为 10。
        message_compression_tokens (int): 触发消息压缩的 token 阈值。默认为 12,000。
        scrape_dir (Path): 存储抓取数据的目录。默认为 "crawler_agent"。
        memory_db_path (Path): 本地记忆数据库路径。默认为 "memory_db_path/memory.db"。
        headless (bool): 是否以无头模式运行浏览器。默认为 False。
        log_level (str): 日志级别（如 "INFO", "DEBUG"）。默认为 "INFO"。
    """

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
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str | None = None
    langfuse_environment: str = "development"
    langfuse_release: str = ""

    # 预算控制
    max_total_tokens: int = 120_000
    max_no_progress_steps: int = 10
    message_compression_tokens: int = 12_000

    # 文件系统
    scrape_dir: Path = Path("hawker_file")
    memory_db_path: Path = scrape_dir / Path("memory.db")

    # 浏览器
    headless: bool = False

    # 日志
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """延迟加载配置单例。

    这可以防止在缺少环境变量时发生导入阶段的失败。

    Returns:
        Settings: 应用程序配置实例。
    """
    return Settings()
