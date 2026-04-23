from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
 
APP_NAME = "hawker"


def default_global_config_path() -> Path:
    return default_data_dir() / "config.env"


def default_data_dir() -> Path:
    return Path.home() / APP_NAME


def default_scrape_dir() -> Path:
    return default_data_dir()


def default_knowledge_db_path() -> Path:
    return default_data_dir() / "knowledge.db"


def resolve_env_files() -> list[Path]:
    return [default_global_config_path()]


class Settings(BaseSettings):
    """应用程序配置类，替代 os.getenv() 全局常量。

    该类使用 pydantic-settings 从环境变量和 `~/hawker/config.env` 加载配置。

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
        scrape_dir (Path): 存储抓取数据的目录。默认为 ``~/hawker``。
        knowledge_db_path (Path): SQLite 站点 SOP 数据库路径。默认为 ``~/hawker/knowledge.db``。
        headless (bool): 是否以无头模式运行浏览器。默认为 False。
        log_level (str): 日志级别（如 "INFO", "DEBUG"）。默认为 "INFO"。
    """

    model_config = SettingsConfigDict(
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
    observer_enabled: bool = True
    observer_reasoning_effort: str = ""
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
    scrape_dir: Path = default_scrape_dir()
    knowledge_db_path: Path = default_knowledge_db_path()

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

    @field_validator("scrape_dir", mode="before")
    @classmethod
    def _normalize_scrape_dir(cls, value: object) -> object:
        """将空的 SCRAPE_DIR 视为默认值，并规范化用户目录。

        注意：这里不做 resolve()，以保留相对路径语义（相对当前工作目录）。
        """
        if isinstance(value, str):
            v = value.strip()
            if not v:
                return default_scrape_dir()
            return Path(v).expanduser()
        if isinstance(value, Path):
            return value.expanduser()
        return value

    @field_validator("knowledge_db_path", mode="before")
    @classmethod
    def _normalize_knowledge_db_path(cls, value: object) -> object:
        """允许将 KNOWLEDGE_DB_PATH 配成空值或目录，并自动规范为数据库文件路径。

        例如：
        - KNOWLEDGE_DB_PATH=
        - KNOWLEDGE_DB_PATH=.
        - KNOWLEDGE_DB_PATH=hawker_file
        """
        if isinstance(value, str):
            v = value.strip()
            if not v:
                return default_knowledge_db_path()
            p = Path(v).expanduser()
        elif isinstance(value, Path):
            p = value.expanduser()
        else:
            return value

        # 如果显式给了目录（存在且为目录），或写成 "."/".."，则自动补齐文件名。
        if p.exists() and p.is_dir():
            return p / "knowledge.db"

        # 末尾有分隔符时，按目录处理（即使目录还不存在）。
        v_str = str(p)
        if v_str.endswith(os.sep) or v_str.endswith("/"):
            return p / "knowledge.db"

        # 对 "." / ".." 做兜底（通常它们存在且为目录，上面已覆盖，但这里更明确）。
        if v_str in {".", ".."}:
            return p / "knowledge.db"

        return p

    @field_validator(
        "browser_executable_path",
        "browser_user_data_dir",
        "browser_storage_state",
        mode="before",
    )
    @classmethod
    def _empty_path_as_none(cls, value: object) -> object:
        """将配置文件中空的可选路径配置视为未配置。"""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("browser_channel", mode="before")
    @classmethod
    def _empty_channel_as_none(cls, value: object) -> object:
        """将空的浏览器 channel 配置视为未配置。"""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _sync_paths(self) -> "Settings":
        """在用户仅覆盖 `scrape_dir` 时，同步默认的知识库路径。"""
        default_knowledge_path = default_knowledge_db_path()
        if self.knowledge_db_path == default_knowledge_path:
            self.knowledge_db_path = self.scrape_dir / "knowledge.db"
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """延迟加载配置单例。

    这可以防止在缺少环境变量时发生导入阶段的失败。

    Returns:
        Settings: 应用程序配置实例。
    """
    return Settings(_env_file=resolve_env_files())
