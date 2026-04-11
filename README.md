# HawkerAgent: Autonomous LLM-Driven Web Intelligence

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Package Manager: uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Framework: Browser-use](https://img.shields.io/badge/Framework-Browser--use-orange.svg)](https://github.com/browser-use/browser-use)

**HawkerAgent** 是一个面向工业级需求的自主网络智能体框架。它基于先进的 LLM 推理模型，通过自动化浏览器、流量分析和动态代码生成，将复杂的 Web 交互转化为高可靠、可观测的结构化数据流。

---

## ⚡ 极速安装 (Recommended)

本项目推荐使用 [uv](https://github.com/astral-sh/uv) 进行环境管理。只需三行命令，即可完成从环境到浏览器的全部准备：

```bash
# 1. 克隆并进入项目
git clone https://github.com/your-repo/hawker-agent.git && cd hawker

# 2. 使用 uv 自动创建环境、同步依赖并安装
uv sync

# 3. 安装浏览器核心
uv run playwright install chromium
```

> 没有安装 `uv`？只需运行 `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux)

---

## 🌟 核心工程特性

- 🧠 **Multi-Agent 推理适配**：针对 Gemini 2.0 Flash Thinking、OpenAI o1/o3 及 DeepSeek-R1 优化，原生支持逻辑复杂的长链推理。
- ⚡ **Async-Native 架构**：全链路基于 `asyncio` 设计，彻底告别 `nest_asyncio` 等同步黑盒，确保在生产环境中的极致稳定性。
- 🔍 **Traffic-Aware 策略**：内置 CDP 拦截器，自动通过 `get_network_log` 分析接口流量，优先实现 **API 重放** 而非脆弱的 DOM 爬取。
- 🛡️ **分布式追踪 (Observability)**：内置 `trace_id` 传播机制，所有 LLM 请求、代码执行和文件 IO 均可跨协程追踪。
- 📦 **自动化产物收割**：智能识别浏览器下载行为，自动将 PDF、CSV 等文件移动并归档至任务专属目录。

---

## ⚙️ 配置环境

复制 `.env.example` 并配置您的模型参数：

```bash
cp .env.example .env
```

核心变量参考：
```ini
OPENAI_API_KEY=AIza...  # 支持 Google 官方 Key 或 OpenAI 兼容 Key
MODEL_NAME=gemini/gemini-2.0-flash-thinking-preview-01-21
HEADLESS=false          # 调试建议设为 false 以观察行为
```

---

## 🚀 运行与开发

### A. 交互式开发 (run.py)
最适合编写复杂任务。直接编辑根目录下的 `run.py` 顶部的 `TASK` 变量：

```python
TASK = "到 Arxiv 下载 2026 年关于 Web Agent 的 PDF 论文，并归档摘要。"
```
执行：`uv run run.py`

### B. 生产级 CLI
```bash
uv run python -m hawker_agent.cli "采集 GitHub Trending 数据" --max-steps 15
```

---

## 📂 观测与产物 (Artifacts)

每次运行产生的 `run_id` 都会在 `crawler_agent/` 下拥有独立空间：

- **`run.ipynb`**: 交互式复盘文档。完美还原 Agent 每一秒的思考与动作。
- **`result.json`**: 最终提交的结构化结果。
- **`app.log`**: 带有 `trace_id` 的彩色全链路日志。
- **归档文件**: 浏览器下载的 PDF 或数据表会自动出现在该文件夹内。

---

## 🧠 深度对齐 (Engineering Specs)

- **AST Sandbox**: 动态分析 LLM 生成的代码，智能补全 `await` 包装，防止协程泄露。
- **Memory Compression**: 自动根据 Token 阈值对中间历史进行摘要压缩，确保长程任务不爆上下文。
- **429 Backoff**: 指数级退避重试策略，优雅应对 API 频率限制。

---
*Powered by HawkerAgent Engineering Team - 2026*
