# HawkerAgent 技术设计文档 (Design Document)

## 1. 项目概览
HawkerAgent 是一个基于大语言模型 (LLM) 驱动的自主网络爬虫框架。它能够将用户的自然语言任务转化为可执行的 Python 代码，通过模拟浏览器操作、流量分析和 API 重放，全自动地完成复杂的数据采集任务。

### 1.1 设计目标
- **工程化 (Engineering)**：从脚本向框架演进，具备高内聚、低耦合的模块化设计。
- **稳定性 (Stability)**：具备完善的异常处理、自动重试（429 错误）和状态恢复机制。
- **可观测性 (Observability)**：全链路日志追踪（Trace ID），支持导出运行记录为 Jupyter Notebook。
- **异步优先 (Async-first)**：基于原生 `asyncio` 架构，彻底消除同步阻塞，支持高性能并发。

---

## 2. 系统架构

### 2.1 逻辑分层
系统采用模块化分层设计，确保各组件可独立测试与替换：

- **数据模型层 (Models)**: 定义 `State`（状态）、`Result`（结果）、`History`（记忆）和 `ItemStore`（去重容器）。
- **执行引擎层 (Agent)**: 
    - `Runner`: 主循环协调者，控制“思考-执行-观测”的迭代。
    - `Executor`: 基于 AST 的 Python 异步沙箱，自动识别并运行异步代码。
    - `Parser`: LLM 响应解析器，支持多语言块混合提取。
- **浏览器能力层 (Browser)**: 封装 `browser-use` 和 `Playwright`，处理会话生命周期、CDP 交互和自动下载归档。
- **工具集 (Tools)**: 提供 `nav`（导航）、`nav_search`（搜索）、`js`（执行）、`http_json`（API 重放）等核心原子能力。
- **存储与导出 (Storage)**: 负责全链路日志记录、结果持久化及 Notebook 导出。

### 2.2 核心数据流
1. **输入**: 自然语言任务 -> `CodeAgentHistoryList`。
2. **推理**: `LLMClient` 调用模型 -> 返回 `CodeAgentModelOutput` (Thought + Code)。
3. **执行**: `Executor` 在隔离的 `Namespace` 中运行代码。
4. **观测**: 工具产生 `Observation` -> 更新 `State` -> 注入 `History`。
5. **结束**: 任务完成或预算耗尽 -> `Exporter` 产出运行报告。

---

## 3. 关键技术特性

### 3.1 全链路日志追踪 (Observability)
引入 `trace_id` 和 `step_id` 机制。通过 `ContextVar` 实现日志在协程间的自动传播。
- `app.log`: 记录系统级的详细工程日志。
- `run.log`: 记录面向业务的 Step 摘要。
- 支持 `Rich` 彩色终端输出，提升开发体验。

### 3.2 异步沙箱执行器 (Executor)
`Executor` 通过对 LLM 生成的代码进行 AST (抽象语法树) 分析：
- 自动检测 `await` 关键字。
- 如果代码包含异步操作，自动将其包装进 `async def __code_exec__():` 函数中运行。
- 实现了变量在不同步骤间的 Jupyter 式持久化。

### 3.3 自动化产物归档
`BrowserSession` 监控系统临时目录：
- 自动收割浏览器产生的下载文件（如 PDF、CSV）。
- 将文件重命名并归档至当前任务的专属日志目录。
- 任务结束时彻底清理临时垃圾，保证磁盘整洁。

### 3.4 爬虫核心策略
- **API 优先**: Prompt 强制引导 Agent 优先检查网络请求（Network Log），实现高效的 API 重放。
- **自动去重**: `ItemStore` 通过语义 Key (URL, ID 等) 自动拦截重复数据。
- **反思机制**: 在任务关键节点（第一步、获取数据、进度过半）自动注入系统提示词，纠正 Agent 的偏差。

---

## 4. 环境要求
- **Python**: 3.11+
- **核心依赖**: `litellm`, `browser-use`, `playwright`, `pydantic-settings`, `rich`, `nbformat`

---

## 5. 快速启动
1. **配置环境**: 复制 `.env.example` 为 `.env` 并填入 API Key。
2. **编写任务**: 在 `run.py` 的 `TASK` 变量中写入目标。
3. **运行**: `python run.py`。

---

## 6. 未来展望
- **云端沙箱**: 对接 `sandbox-runtime` 实现更安全的代码隔离。
- **多代理协作**: 引入 `Supervisor` 角色，协同多个模型共同处理超长链路任务。
- **视觉增强**: 进一步整合模型的多模态能力，直接基于截图进行动作预测。

---
*文档版本: 1.0.0 | 最后更新: 2026-04-11*
