# Hawker

[English](./README.md) | [简体中文](./README.zh-CN.md)

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](./LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)
[![Mypy](https://img.shields.io/badge/types-mypy-blue.svg)](https://mypy-lang.org/)

Hawker 是一个面向浏览器场景的自主 Web 智能体框架，用来把浏览器交互、网络流量和 LLM 推理过程，转换成可审计、可复现、可交付的结构化结果。

它适合这类任务：

- 页面有复杂交互，普通爬虫不够用
- 需要浏览器操作与接口重放结合
- 需要完整运行产物用于调试、审计和复盘
- 需要最终交付结构化结果，而不是只要一段自然语言

## 项目状态

Hawker 目前处于 **积极开发中**，可用于实验性和内部场景，但默认应视为 **实验性项目**。

推荐场景：

- 内部研究和自动化任务
- Agent 工程实验
- 浏览器优先的数据采集流程

当前不建议未经额外加固直接用于：

- 受监管的高合规生产环境
- 对密钥隔离、依赖治理和安全审计要求极高的场景

## 为什么是 Hawker

很多 Web 自动化系统只解决“点到了页面”或“代码跑完了”，Hawker 更关注真正的交付闭环：

- **结构化交付**：最终结果落盘为 `result.json`，包含 items 和交付 artifact
- **运行可追踪**：每次运行都有日志、Notebook 风格执行记录和 LLM IO 记录
- **浏览器与流量协同**：如果存在真实数据 API，会优先走接口重放，而不是脆弱 DOM 抓取
- **长任务友好**：支持历史压缩、checkpoint 和记忆增强，降低上下文膨胀和重复试错

## 架构概览

Hawker 主要由四层组成：

- **Agent Runtime**：step 编排、执行、最终交付校验和终止逻辑
- **Browser Layer**：基于 Playwright / browser-use 的浏览器操作、DOM 快照和网络日志
- **LLM Layer**：模型调用、token 统计、evaluator / healer 侧路能力
- **Storage Layer**：结果导出、运行产物归档、记忆持久化

每次运行的核心产物目录：

- `hawker_file/<run_id>/result/result.json`：面向用户的最终交付结果
- `hawker_file/<run_id>/run.ipynb`：Notebook 风格运行轨迹
- `hawker_file/<run_id>/llm_io.json`：模型输入输出记录
- `log/<run_id>/`：应用和运行日志

## 快速开始

### 1. 安装依赖

项目使用 [`uv`](https://github.com/astral-sh/uv) 进行环境管理。

```bash
git clone <your-fork-or-repo-url>
cd hawker
uv sync
uv run playwright install chromium
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

典型配置：

```ini
OPENAI_API_KEY=...
MODEL_NAME=gemini/gemini-2.0-flash-thinking-preview-01-21
HEADLESS=false
```

如果目标网站需要登录态，Hawker 支持通过以下方式复用浏览器状态：

- 本机 Chrome 用户目录
- 导出的 `storage_state.json`
- 已运行浏览器的 CDP 端点

具体变量请参考 `.env.example`。

### 3. 运行任务

编辑 `run.py`：

```bash
uv run run.py
```

CLI 模式：

```bash
uv run python -m hawker_agent.cli "采集 GitHub Trending 数据" --max-steps 15
```

## 使用模型

Hawker 适合需要策略判断而不是只靠选择器的任务。

例如：

- “下载 Web Agent 相关论文 PDF，并归档摘要”
- “打开列表页，找出真实 JSON API，再重放它”
- “读取公开账号最近动态，并返回结构化 items 和总结”

标准执行流程通常是：

1. 侦察页面和网络请求
2. 优先选择可重放 API
3. 必要时回退到 DOM 提取
4. 清洗和校验结构化 items
5. 提交并通过最终交付校验

## 开发

安装开发依赖：

```bash
uv sync --extra dev
```

常用命令：

```bash
uv run ruff check hawker_agent tests run.py
uv run mypy hawker_agent
uv run pytest -q
```

当前项目主要使用：

- `ruff` 做 lint
- `mypy` 做严格类型检查
- `pytest` 与 `pytest-asyncio` 做测试

## 安全

如果发现安全问题，请不要公开提 issue。

安全漏洞提交流程见 [SECURITY.md](./SECURITY.md)。

## 贡献

欢迎贡献。在提交 PR 之前，请至少：

1. 本地跑通 lint 和测试
2. 控制改动范围，保持 review 友好
3. 如果行为发生变化，同步更新文档

详见 [CONTRIBUTING.md](./CONTRIBUTING.md)。

## 许可证

本项目采用 [Apache License 2.0](./LICENSE)。

第三方依赖仍遵循各自原始许可证，详见：

- [NOTICE](./NOTICE)
- [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)

## 致谢

Hawker 建立在许多优秀开源项目之上，特别感谢：

- [browser-use](https://github.com/browser-use/browser-use)：提供面向 Agent 的浏览器自动化能力
- [Playwright for Python](https://github.com/microsoft/playwright-python)：提供浏览器控制基础设施
- [LiteLLM](https://github.com/BerriAI/litellm)：统一多模型提供方调用接口
- [Langfuse](https://github.com/langfuse/langfuse)：提供 tracing / observability 集成能力
- [HTTPX](https://github.com/encode/httpx)、[Typer](https://github.com/fastapi/typer)、[Rich](https://github.com/Textualize/rich)、[Jinja](https://github.com/pallets/jinja)、[Pydantic Settings](https://github.com/pydantic/pydantic-settings)、[nbformat](https://github.com/jupyter/nbformat)、[Boto3](https://github.com/boto/boto3)

感谢这些项目的维护者和社区贡献者。
