# Hawker 旁路 Observer 与 SQLite SOP 库设计草案

## 1. 这次设计变更的结论

这次方向与之前不同，结论非常明确：

1. **废弃 Hawker 现有“长期记忆”体系。**
2. **不再把跨任务知识写成 `MemoryStore` 的 summary / raw code / recipe。**
3. **不再把知识库存成项目内本地 Markdown 文件。**
4. **跨任务唯一保留的长期知识形态，是站点级 SOP，并统一存入 SQLite。**

这里要区分两个概念：

- `CodeAgentHistoryList`
  - 这是单次任务执行期的对话上下文，不是长期记忆库。
  - 它短期内仍然需要保留，否则主 Agent 没法完成多步执行。
- `MemoryStore` / `Memory Workspace`
  - 这是跨任务的“所谓记忆”系统。
  - 这条线应被替换。

所以这次真正要移除的是：

- `MemoryStore`
- `memory/store.py` 的经验抽取和召回逻辑
- `runner.py` 启动时的 Memory Workspace 注入
- “站点经验 / 失败约束 / 原始成功代码片段” 这种记忆化设计

替代方案是：

- 旁路 `Observer Agent`
- 单一知识形态：`Standard SOP`
- 单一存储底座：`SQLite`


## 2. 为什么要废掉“记忆”体系

从效果上看，现有设计的问题不是“实现得不够复杂”，而是范式本身不对。

### 2.1 现有记忆形态的天然缺陷

当前 MemoryStore 保存的是：

- 站点经验 summary
- 失败约束
- 成功代码片段
- task_intent / page_kind / site_key 等检索标签

这套设计的问题是：

1. 记忆条目太碎
   - 召回后只能得到几条片段化提示，不能直接执行。
2. 记忆不是 SOP
   - 它告诉模型“可能该怎么做”，但没有把整条成功路径蒸馏成稳定工作流。
3. 记忆质量不稳定
   - 很依赖当次 run 的 summary 质量。
4. 记忆容易污染 prompt
   - 注入到 `Memory Workspace` 后，本质上还是在给模型塞提示，而不是给它一份工业化的执行手册。

### 2.2 更合理的长期知识应该是什么

跨任务长期知识最应该保留的是：

- 这个站点该优先走 Browser 还是 API
- 第一入口是什么
- 搜索 / 列表 / 详情 / 翻页 的标准工作流是什么
- 关键 URL / 参数规律是什么
- 有哪些坑

这本质上就是一份 **站点标准 SOP**。

所以长期知识的正确单位不是 `MemoryEntry`，而是：

- `Site SOP`


## 3. 新目标架构

### 3.1 新的知识链路

```text
主 Agent 完成任务
    ->
旁路 Observer 被异步唤醒
    ->
对成功路径和 netlog 做剪枝
    ->
生成/修订站点标准 SOP
    ->
写入 SQLite 的 site_sops 表
    ->
下一次任务启动时，按 domain 召回 SOP
    ->
把 SOP 作为站点知识块注入主 Agent
```

### 3.2 明确废弃的模块职责

以下职责应被删除，而不是继续增强：

- `MemoryStore.search()` 这类“片段化经验召回”
- `build_raw_code_memories()` 这类“从代码抽取记忆”
- `history.set_memory_workspace()` 这类“把记忆塞回工作区”

长期来看，`Memory Workspace` 这一块可以整体删除。


## 4. SQLite 中应该存什么

不再存“记忆条目”，而是存“站点 SOP 文档”。

### 4.1 建议表结构

建议新增一张核心表：

```sql
CREATE TABLE site_sops (
    domain TEXT PRIMARY KEY,
    sop_markdown TEXT NOT NULL,
    golden_rule TEXT NOT NULL,
    quality_status TEXT NOT NULL DEFAULT 'active',
    update_reason TEXT NOT NULL,
    source_run_id TEXT NOT NULL,
    source_url TEXT NOT NULL DEFAULT '',
    version INTEGER NOT NULL DEFAULT 1,
    evidence_hash TEXT NOT NULL,
    proof_summary TEXT NOT NULL,
    last_generated_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

再配一张版本表：

```sql
CREATE TABLE site_sop_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,
    version INTEGER NOT NULL,
    sop_markdown TEXT NOT NULL,
    update_reason TEXT NOT NULL,
    source_run_id TEXT NOT NULL,
    evidence_hash TEXT NOT NULL,
    proof_summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

如果还要支持熔断与人工审查，可以再加：

```sql
CREATE TABLE site_sop_update_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,
    event_type TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    reason TEXT NOT NULL,
    source_run_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

### 4.2 为什么 SOP 仍然存 Markdown

虽然底座是 SQLite，但正文仍建议保存 Markdown，而不是拆成十几个 JSON 字段。

原因：

- 最终消费者是 Agent
- 高质量 SOP 的最佳阅读形态就是 Markdown
- 版本 diff 也更清晰

SQLite 存的是：

- Markdown 正文
- 少量可检索字段
- 严格的元数据和版本信息


## 5. Observer 的职责

Observer 不再是“总结记忆”的模型，而是：

- 站点 SOP 蒸馏器
- 站点 SOP 修订器

### 5.1 输入

Observer 的输入应被严格裁剪为两块：

1. `execution_log`
   - 成功代码路径
   - 关键失败重试
   - 少量 selector / DOM 证据
   - 真实输出样本

2. `network_summary`
   - 成功请求的 API 路径
   - query / body / header 规律
   - 响应结构摘要

### 5.2 输出

Observer 只输出一份最终 SOP Markdown，不输出解释。

### 5.3 失败容忍

Observer 失败时：

- 只记日志
- 不影响主任务结果
- 不写脏数据进 SQLite


## 6. 触发时机

### 6.1 触发器 A：成功蒸馏

条件：

- `state.done is True`
- 本次任务确实拿到了有效结构化结果

动作：

- 异步生成或修订该域名 SOP

### 6.2 触发器 B：Healer 成功修复后的硬更新

条件：

- 旧 SOP 已经导致 Worker 明显失效
- Healer 产出的新代码被后续执行验证成功

动作：

- 把这次 run 标记为“高优先级 SOP 更新”

### 6.3 触发器 C：Evaluator 识别到数据漂移

条件：

- 代码没报错
- 但数据结果缺字段、空字段、结构错位
- 修复后恢复正常

动作：

- 局部修订 SOP 中的数据提取段与 Gotchas

### 6.4 触发器 D：路径优化发现

条件：

- 当前成功路径主要靠 Browser
- 但 netlog 暗示存在更短的 API 路径

动作：

- 允许 Observer 把 SOP 的 Golden Rule 改成 API 优先


## 7. 准入标准

不是每次成功都应该直接覆盖 SOP。

### 7.1 必须有活体证据

任何新增或修改的 Workflow 都必须来自真实执行证据。

最低要求：

- 对应代码真的跑过
- 有真实输出样本
- SOP 代码块末尾带 `# Confirmed output (日期): ...`

### 7.2 必须做 Smart Merge

Observer 必须合并旧规则和新证据，不能无限追加。

例如：

- 旧规则失效时要替换，而不是并列保留
- 新 API 已经证明更优时，要删除旧的 Browser-only 表述

### 7.3 冷却熔断

如果同一域名在 24 小时内已更新过 2 次：

- 不再自动覆盖
- 只写入候选版本或 update event
- 等人工审查


## 8. 主流程如何消费 SOP

下一次任务启动时，不再调用 `MemoryStore.search()`。

改为：

1. 从任务中解析目标 domain
2. 去 SQLite 的 `site_sops` 查主版本
3. 命中后，把 SOP 作为独立知识块注入 prompt

注入形态建议是：

```text
[Site SOP]
<完整或裁剪后的站点 SOP>
```

而不是：

```text
[Memory Workspace]
- 站点经验 ...
- 成功路径 ...
```

也就是说，主 Agent 消费的是一份标准手册，不是几条零散记忆。


## 9. 对现有代码的直接影响

### 9.1 需要逐步删除

- [store.py](/Users/hpd/code/hawker/hawker_agent/memory/store.py)
- [runner.py](/Users/hpd/code/hawker/hawker_agent/agent/runner.py) 中的：
  - `MemoryStore(...)`
  - `memory_store.search(...)`
  - `_build_memory_workspace_entries(...)`
  - `_finish()` 内的 `build_raw_code_memories(...)`
- [history.py](/Users/hpd/code/hawker/hawker_agent/models/history.py) 中的：
  - `_memory_workspace`
  - `set_memory_workspace()`
  - Workspace 中的 `Memory Workspace` 区块

### 9.2 需要新增

- `hawker_agent/knowledge/store.py`
  - 管理 SQLite 中的 `site_sops`
- `hawker_agent/knowledge/observer.py`
  - 负责构造 observer 输入、调模型、验证结果、写库
- `hawker_agent/templates/observer_prompt.jinja2`
  - Observer 的严格 SOP 生成 prompt


## 10. 为什么不把知识库存成本地文件

这次明确不建议落到项目内文件系统。

原因：

1. 项目目录不是数据库
   - 长期运行会产生大量站点文件和版本文件。
2. 并发写入与部署同步麻烦
   - 尤其未来如果迁到服务端、多实例运行，更不适合本地文件。
3. SQLite 更适合：
   - 原子更新
   - 版本表
   - 冷却熔断统计
   - 审计与检索

所以本轮设计明确采用：

- SQLite 作为 SOP 底座
- 不再写 `knowledge_base/*.md`


## 11. 当前研究分支的建议实施顺序

### 阶段 1：架构切换

目标：

- 在设计层面确认“去记忆化”
- 把长期知识模型从 `MemoryEntry` 切换到 `Site SOP`

### 阶段 2：最小实现

目标：

- 新增 `site_sops` SQLite 表
- 新增 `SiteSOPStore`
- 先实现读取与写入，不接主流程

### 阶段 3：接入旁路 Observer

目标：

- 在任务结束后异步生成 SOP
- 写入 `site_sops`

### 阶段 4：替换主流程召回

目标：

- 删除 `MemoryStore.search()`
- 改为启动时直接召回 SOP

### 阶段 5：删除旧记忆系统

目标：

- 删掉 `memory/store.py`
- 删掉 `Memory Workspace`
- 删掉旧测试


## 12. 最终结论

这次方向已经很清楚：

- Hawker 不应该再维护一套“站点经验记忆系统”。
- 长期知识的唯一正确单位，是“站点标准 SOP”。
- SOP 不该写本地文件，应该严格存入 SQLite。
- 旁路 Observer 负责蒸馏和修订 SOP。
- 主 Agent 启动时只消费 SOP，不再消费所谓记忆。

换句话说，新的原则是：

**去记忆化，改为 SOP 化。**

