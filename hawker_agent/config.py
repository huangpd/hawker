from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
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
    small_model_name: str | None = None
    healer_enabled: bool = True
    healer_reasoning_effort: str = ""
    healer_max_attempts: int = 3
    final_evaluator_enabled: bool = True
    final_evaluator_reasoning_effort: str = ""
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str | None = None
    langfuse_environment: str = "development"
    langfuse_release: str = ""

    # 预算控制
    max_steps: int = 30
    max_total_tokens: int = 200_000
    max_no_progress_steps: int = 10
    message_compression_tokens: int = 12_000

    # 文件系统
    scrape_dir: Path = Path("hawker_file")
    memory_db_path: Path = scrape_dir / Path("memory.db")

    # 浏览器
    headless: bool = False
    browser_executable_path: Path | None = None
    browser_user_data_dir: Path | None = None
    browser_profile_directory: str = "Default"
    browser_storage_state: Path | None = None
    browser_channel: str | None = None
    browser_cdp_url: str | None = None

    # 日志
    log_level: str = "INFO"

    @field_validator(
        "browser_executable_path",
        "browser_user_data_dir",
        "browser_storage_state",
        mode="before",
    )
    @classmethod
    def _empty_path_as_none(cls, value: object) -> object:
        """将 .env 中空的可选路径配置视为未配置。"""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _sync_paths(self) -> "Settings":
        """在用户仅覆盖 `scrape_dir` 时，同步默认的 memory.db 位置。"""
        default_memory_path = Path("hawker_file") / "memory.db"
        if self.memory_db_path == default_memory_path:
            self.memory_db_path = self.scrape_dir / "memory.db"
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """延迟加载配置单例。

    这可以防止在缺少环境变量时发生导入阶段的失败。

    Returns:
        Settings: 应用程序配置实例。
    """
    return Settings()
