# Hawker 记忆系统设计计划

## 1. 目标

Hawker 的记忆系统，不应该是一个泛化的“陪伴式聊天记忆库”，而应该是一个面向网页任务执行的经验系统。

它要解决的核心问题不是“记住用户说过什么”，而是：

1. 当 Agent 再次进入同一站点或相似站点时，能够带着成功经验出发，而不是从零开始摸索。
2. 当 Agent 曾经在某站点踩过坑时，能够提前避开已知失败路径。
3. 记忆必须是可审计、可检索、可折叠的，不能重新变成一条无限膨胀的聊天历史。
4. 记忆应当服务于当前的 Context Engine，而不是绕开它另起一套隐形状态机。

换句话说，Hawker 的记忆系统本质上是：

- 面向站点的经验记忆
- 面向任务的操作记忆
- 面向框架的策略记忆


## 2. 我们到底要记什么

不是所有成功执行过的内容都值得记忆。建议只保留高价值、可迁移、可复用的知识。

### 2.1 站点经验记忆

用于回答“这个站点以前怎么成功过”。

示例：

- 站点 `github.com` 的 Trending 页可直接通过 SSR DOM 提取，无需依赖接口
- 站点 `mcp.aibase.com` 的排序参数不能盲猜，必须先点击 UI 再看抓包
- 某站点翻页必须点击“下一页”，直接改 `?page=2` 会触发封禁或返回错误结果

这类记忆最重要，因为它最直接影响下一次同站点成功率。

### 2.2 提取策略记忆

用于回答“面对这类页面结构，什么策略更有效”。

示例：

- 商品列表页优先尝试网络请求和 SSR DOM，再考虑滚动加载
- 复杂弹窗站点先用 `dom_state(diff)` 而不是 `full`
- 下载站优先使用浏览器态 Cookie + HTTP 下载，不必强行走浏览器 click 下载

这类记忆不一定绑定具体站点，可以跨站点迁移。

### 2.3 负面经验记忆

用于回答“哪些做法不要再试”。

示例：

- 该站点点击排序按钮后会触发异步请求，不能根据页面视觉排序直接推断请求参数
- 某 API 参数 `download|desc` 实际无效，真实排序由页面点击触发的请求体决定
- 该站点使用 Shadow DOM，`querySelector` 直查会失效

这类记忆的价值很高，因为它直接节省 trial-and-error token。

### 2.4 结构化站点画像

用于快速识别站点和辅助检索。

示例字段：

- `site_key`: 规范化域名或子域名
- `url_pattern`: 路径模式
- `page_kind`: 列表页、详情页、搜索页、登录页、下载页
- `stack_hint`: SSR、CSR、Next.js、Vue、Shadow DOM、多 iframe
- `auth_required`: 是否依赖登录态

这部分不是经验本身，但它是经验被正确召回的索引基础。


## 3. 记忆系统的分层

建议把记忆分成三层，而不是搞一个统一大仓库。

### 3.1 Episode Memory

保存单次任务运行中沉淀下来的“经验片段”。

特点：

- 粒度小
- 可追溯到具体 run
- 可随时回放和删除

典型内容：

- 本次任务对某站点的成功策略
- 本次任务遇到的失败路径
- 本次任务提取出的 API 参数规律

### 3.2 Site Memory

把多个 Episode 聚合为站点级别的稳定经验。

特点：

- 一个站点对应多条记忆
- 有冲突时要保留版本和置信度
- 会随着更多任务积累而更新

典型内容：

- “推荐优先路径”
- “禁用路径”
- “常见页面类型”
- “典型字段提取方式”

### 3.3 Policy Memory

这是框架级策略知识，不绑定单站点。

特点：

- 更像长期经验规则
- 来源可以是人工沉淀，也可以是高频 Episode 总结

典型内容：

- 先抓包再猜参数
- 先 `summary/diff`，必要时再 `full`
- 搜索页先识别 SSR/CSR，再决定 DOM 或 API 路线


## 4. 记忆进入系统的方式

记忆不能直接把整段对话存进去。必须经过提纯。

### 4.1 触发时机

建议只在以下时机尝试写入记忆：

1. 任务成功结束时
2. 出现明确失败原因且具有可迁移性时
3. 发现稳定 API 规律时
4. 达成关键里程碑时

不建议每一步都写记忆，否则系统会迅速污染。

### 4.2 提纯流程

每次提纯应从以下输入生成候选记忆：

- `llm_io.json`
- `result.json`
- Notebook Workspace
- 关键工具调用轨迹
- 网络日志摘要

然后抽取成统一结构：

```json
{
  "memory_type": "site_lesson",
  "site_key": "mcp.aibase.com",
  "page_kind": "explore_list",
  "trigger": "sort_by_download",
  "lesson": "不要直接猜排序参数，必须先点击页面排序并抓取 querypage 请求体。",
  "evidence": {
    "run_id": "xxx",
    "step": 12,
    "tool": "get_network_log"
  },
  "confidence": 0.82,
  "success": true,
  "negative": false
}
```

### 4.3 写入原则

只保存以下三类内容：

- 可复用的成功策略
- 可复用的失败约束
- 可识别站点的结构特征

不保存以下内容：

- 一次性的页面文案
- 大段 DOM
- 大段日志原文
- 低置信度的猜测


## 5. 记忆如何被召回

这是设计里最重要的一层。写入不是难点，召回才是。

### 5.1 召回入口

建议在任务早期就做一次记忆检索，而不是走到第 10 步才想起来。

推荐召回点：

1. 任务开始，`nav()` 前
2. 首次进入站点后，识别到 `site_key + page_kind` 时
3. 关键动作前，例如登录、翻页、排序、下载、抓包
4. 连续无进展时，作为诊断增强

### 5.2 检索维度

不应只按 embedding 相似度检索。建议混合召回：

- 域名精确召回
- URL 路径模式召回
- 页面类型召回
- 任务意图召回
- 向量语义召回

最终按加权排序：

```text
score =
  0.35 * site_match +
  0.20 * page_kind_match +
  0.20 * task_intent_match +
  0.15 * vector_similarity +
  0.10 * recency_confidence
```

### 5.3 注入方式

不要把召回记忆直接拼成聊天消息。它应该进入 Notebook Workspace 的新分区：

```text
[Memory Workspace]
- 站点经验: 该站点排序参数不要猜，先点击 UI 再抓包
- 成功路径: 列表页优先检查是否 SSR，可直接 DOM 提取
- 失败约束: 直接改 query 参数曾导致错误结果
```

这样它依旧属于状态流，而不是旧式聊天历史。


## 6. 数据存储方案

### 6.1 推荐的一期方案：SQLite + FTS5 + 可选向量扩展

这是我最推荐的起步方案。

理由：

- 本地优先
- 部署极轻
- 易于调试
- 易于导出和审计
- 能与当前 Hawker 的 run 目录天然配合

最低配置：

- 一张 `memory_entries` 主表
- 一张 `memory_links` 关系表
- SQLite FTS5 做全文检索

如果后面需要更强语义检索，再增加：

- `sqlite-vec`
- 或单独的向量索引表

建议主表字段：

- `id`
- `memory_type`
- `site_key`
- `url_pattern`
- `page_kind`
- `task_intent`
- `summary`
- `detail`
- `success`
- `negative`
- `confidence`
- `source_run_id`
- `source_step`
- `created_at`
- `updated_at`
- `embedding`

### 6.2 Markdown 方案

也可以用 Markdown 文件加 frontmatter 的方式存储记忆，例如按站点写：

- `memory/sites/github.com.md`
- `memory/sites/mcp.aibase.com.md`

优点：

- 人类可读性很好
- 版本控制友好
- 便于手工修订

缺点：

- 检索和排序能力弱
- 结构化更新麻烦
- 并发写入和聚合成本高

结论：

- Markdown 适合做导出视图
- 不适合作为主存储

### 6.3 纯向量库方案

不建议一开始就上 Chroma、Qdrant 这类纯向量库作为核心存储。

原因：

- 你要检索的是站点经验，不只是语义相似
- 很多命中来自域名、URL 模式、页面类型等强结构信号
- 纯向量方案会让“为什么命中这条记忆”变得不透明

结论：

- 向量检索是加分项
- 不能替代结构化主索引


## 7. 自研还是用开源

### 7.1 我的判断

Hawker 应该自研“记忆内核”，但可以借用开源项目的局部能力。

也就是：

- 不建议把整套记忆系统外包给通用 Agent memory framework
- 建议自研记忆 schema、写入规则、召回策略、Workspace 注入方式
- 可以复用外部项目的 embedding、向量存储、事实提取思路

原因很直接：

1. Hawker 的核心不是“聊天记忆”，而是“站点执行经验”
2. Hawker 已经有自己的 Context Engine，外部框架往往会把记忆直接拼回 messages
3. 你真正的壁垒在记忆规则，而不是存储库

### 7.2 开源方案对比

以下判断基于官方资料和项目主页：

- Mem0
  - 官方定位是“Universal memory layer for AI Agents”，同时提供托管和开源模式
  - 优点是现成、生态广、支持自动捕获与搜索
  - 缺点是它更偏通用 memory layer，天然更靠近“通用对话记忆”
  - 适合作为参考对象，也可以借它的 OSS 检索/存储思路
  - 不建议直接成为 Hawker 的核心记忆引擎
  - 参考: https://github.com/mem0ai/mem0
  - 参考: https://docs.mem0.ai/open-source/overview

- Letta
  - 官方定位是“stateful agents”，强调 memory blocks、archival memory 和状态型 agent
  - 优点是理念很先进，和“状态化 Agent”方向接近
  - 缺点是它更像完整 agent runtime，不是一个薄记忆层
  - 适合借鉴分层记忆和 sleeptime consolidation 思路
  - 不适合直接嵌进 Hawker 作为子系统
  - 参考: https://github.com/letta-ai/letta
  - 参考: https://docs.letta.com/letta-code/memory/

- LangMem
  - 官方定位是帮助 agents 从交互中学习和适应
  - 优点是把 memory 管理和 background consolidation 讲得很清楚
  - 缺点是生态绑定 LangGraph 较深
  - 适合借鉴 API 设计与后台记忆提纯流程
  - 不适合直接作为 Hawker 的底座
  - 参考: https://github.com/langchain-ai/langmem

- Zep
  - 官方定位是 long-term memory service，强调长期记忆与结构化召回
  - 优点是长期记忆、事实提取、图谱方向有启发
  - 缺点是更偏服务型系统，基础设施比 Hawker 当前需要的更重
  - 适合参考它对 session memory 与 long-term memory 的分层
  - 不适合作为一期最小落地方案
  - 参考: https://github.com/getzep/zep-python
  - 参考: https://help.getzep.com/v2/memory

### 7.3 最现实的结论

一期最合理的路径是：

- 自研记忆规则和数据模型
- 用 SQLite 做主存储
- 用 FTS5 做检索底座
- 向量检索作为二期增强
- 从 Mem0、Letta、LangMem、Zep 借鉴提纯和召回机制，但不整体引入


## 8. 一期最小可行方案

### 8.1 目标

在不破坏现有 Context Engine 的前提下，让 Hawker 具备“同站点经验复用”能力。

### 8.2 一期功能

1. 新建本地记忆库
   - `memory.db`
2. 新增 `Memory Workspace`
3. 任务结束后自动生成候选 Episode Memory
4. 任务开始时按 `site_key + task_intent` 检索
5. 注入 top-k 记忆摘要到下一轮 prompt

### 8.3 一期不做

- 自动修改系统提示词
- 自动把记忆升级为全局策略
- 复杂知识图谱
- 多用户共享记忆
- 云端同步


## 9. 二期增强

1. 记忆 consolidation
   - 多个 Episode 合并成站点级稳定经验

2. 记忆冲突管理
   - 允许同一站点存在多条冲突经验
   - 通过置信度和新鲜度控制排序

3. 记忆评分闭环
   - 某条记忆被命中后是否真的帮助任务成功
   - 成功则升权，误导则降权

4. 记忆版本化
   - 同站点改版后，旧经验自动衰减

5. 记忆导出视图
   - 自动导出为 Markdown 站点档案，便于人工审阅


## 10. 我建议的最终路线

最推荐的路线不是“选一个记忆框架接进来”，而是：

1. 自研一层 Hawker Memory Kernel
2. 内核只做四件事
   - 记忆提纯
   - 结构化存储
   - 混合检索
   - Workspace 注入
3. 底层先用 SQLite + FTS5
4. 向量检索后补
5. 开源框架只参考，不托管核心逻辑

一句话总结：

Hawker 要做的不是一个“会聊天的长期记忆库”，而是一个“会积累站点经验的执行记忆系统”。

