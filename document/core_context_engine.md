# Hawker 上下文引擎设计

## 1. 背景

Hawker 的核心难题并不是“如何调用浏览器”或“如何执行 Python”，而是如何在长链路任务中，持续向大模型提供足够的信息，同时避免上下文失控膨胀。

传统 Agent 往往采用线性对话流：

`User -> Assistant -> User Observation -> Assistant -> ...`

这种结构在短任务中足够有效，但在 10 到 20 步以上的爬取任务中，会出现三个根本问题：

1. 历史不可逆增长。早期失败尝试、无效代码、临时 DOM 都会持续占用 token。
2. 状态表达混乱。模型需要从旧代码和旧输出中“反推”当前有哪些变量、页面处于什么状态。
3. 页面上下文过重。DOM 作为最长的 Observation，极易成为 token 消耗主因。

因此，Hawker 的上下文引擎不再把“对话历史”视为主载体，而是转向“工作区状态 + 短期日志”的表达方式。

---

## 2. 核心理念

这套技术的核心不是简单压缩字符串，而是重构模型看到的信息组织方式。

### 2.1 从对话流转向状态流

模型不再依赖完整聊天记录理解当前任务，而是基于一个显式工作区来决策。

工作区只保留高价值状态：

- 当前任务
- 当前运行时状态
- 已确认的成功里程碑
- 已知失败经验
- 当前可直接复用的变量摘要
- 当前页面的结构化上下文

换句话说，模型面向的是一个持续更新的 Notebook Workspace，而不是一个无限增长的聊天线程。

### 2.2 长短期记忆分离

上下文引擎将记忆分成两层：

- STM（短期记忆）：保留最近少量原始消息，用于处理当前调试、错误恢复和局部推理
- LTM（长期记忆）：将更早历史提纯为里程碑、经验教训和工作区状态

这样可以同时满足：

- 短距离调试时仍有足够细节
- 长任务中不会因为历史堆积而拖垮 prompt

### 2.3 DOM 不是长期正文，而是可衰减证据

DOM 是爬虫 Agent 中最昂贵的上下文类型，因此不应长期以原始文本形式保留。

Hawker 采用三层 DOM 表达：

- `summary`：轻量页面摘要，可长期保留
- `diff`：相对上一版页面快照的语义变化，可短期保留
- `full`：完整页面结构，只允许短期存在，随后自动折叠

这意味着完整 DOM 是短期证据，而不是长期状态。

---

## 3. Prompt 结构

当前 Hawker 向模型发送的 prompt，逻辑上由以下部分构成：

1. `system`
2. `task`
3. `Notebook Workspace`
4. 最近少量原始 assistant/user 消息

其中 `Notebook Workspace` 为核心载体，包含：

- `Runtime Snapshot`
- `Milestones`
- `Long-Term Memory`
- `Namespace Snapshot`
- `DOM Workspace`

这套结构的目标是让模型在单轮推理时，看到的是“当前真实工作区”，而不是漫长且噪音很高的聊天历史。

---

## 4. Notebook Workspace

### 4.1 Runtime Snapshot

用于表达当前任务的即时运行状态：

- 已采集条数
- 当前步骤
- token 消耗
- 连续无进展步数

它解决的问题是：模型不需要再从日志中推断当前进度。

### 4.2 Milestones

记录已经确认成功的重要节点，例如：

- 已进入列表页
- 已发现 API
- 已完成首批数据提取

Milestones 的作用不是复盘历史，而是为模型建立“已经成立的事实”。

### 4.3 Long-Term Memory

记录高纯度的失败经验和策略约束，例如：

- 某选择器已失效
- 某接口返回 403
- 某路径验证无效，不应重复尝试

Long-Term Memory 的价值在于阻止模型重复犯错。

### 4.4 Namespace Snapshot

记录当前可直接使用的持久化变量及其摘要，例如：

- 变量名
- 变量类型
- 列表长度 / 字典键数
- 简短样本

它解决的问题是：模型不再依赖“回忆旧代码”理解当前环境。

### 4.5 DOM Workspace

记录最近一次显式请求的页面结构证据。

不同于旧式 `[Browser State]` 的临时消息，DOM Workspace 被纳入工作区统一管理，并具备生命周期控制：

- `summary`：长期有效
- `diff`：短期有效
- `full`：仅短暂有效

---

## 5. DOM 上下文模型

### 5.1 语义快照而非原始 HTML 差分

Hawker 不做原始 DOM 字符串 diff，而是构建语义快照：

- 标题
- URL
- 可交互元素数
- 可交互元素预览
- 区域标签
- 滚动状态
- 待处理请求数

基于语义快照生成：

- `DOM Summary`
- `DOM Diff`

这样做的优点是：

- 输出更稳定
- token 更低
- 对模型更友好
- 不依赖原始 HTML 文本细节

### 5.2 DOM 生命周期控制

这是当前上下文引擎中的关键机制之一。

规则如下：

- `summary`：可长期驻留在 `DOM Workspace`
- `diff`：保留 2 轮后折叠为 `summary`
- `full`：保留 1 轮后折叠为 `summary`

该设计的工程目的很明确：

- 允许模型在短时间内利用完整 DOM 做定位或调试
- 避免完整 DOM 在后续所有步骤中持续消耗 token

这也是解决“某一步调用 full DOM 后，后续所有步骤输入 token 持续高位”的核心技术。

---

## 6. 自动 DOM 策略

Hawker 不要求模型频繁手动指定 DOM 模式，而是通过 `mode="auto"` 执行自动决策。

当前策略为：

- `nav` / `nav_search`
  - 默认降到 `summary`
- `click` / `click_index`
  - 若已有前序快照，默认使用 `diff`
  - 若连续无进展达到阈值，自动升级到 `full`
- `dom_state`
  - 若已有前序快照，优先使用 `diff`
  - 若无前序快照，使用 `full`
- 点击失败时
  - 自动补充一次诊断 DOM

这套策略的设计目标不是完全替代模型判断，而是在不牺牲稳定性的前提下，降低 DOM 上下文的使用频率和长度。

---

## 7. 观测与可审计性

这套技术不仅要“压上下文”，还必须可观测、可回放、可审计。

因此 Hawker 额外导出 `llm_io.json`，按 step 保存：

- 实际发送给模型的 messages
- prompt 的拆分结构
- 被折叠 / 省略的部分
- 模型原始返回对象
- 解析后的 thought / code
- 执行结果 observation

这意味着：

- 可以审查任一步真实 prompt 长什么样
- 可以分析 token 为何上升
- 可以区分“工作区内容”和“传输层原始响应”

这对后续继续优化上下文引擎是必要基础设施。

---

## 8. Execution Model

上下文引擎之所以成立，不只是因为 prompt 结构被重写了，还因为执行模型本身被设计成适合状态流。

Hawker 采用 Notebook 式执行模型：

- 模型每一步只生成一段 Python
- 代码在统一命名空间中增量执行
- 执行结果回流为 Observation
- 变量按照持久化协议进入下一步工作区

这意味着上下文并不需要记住“模型过去写过什么代码”，而只需要表达：

- 当前有哪些已持久化变量
- 当前页面处于什么状态
- 当前有哪些已确认事实和失败经验

这是上下文引擎能够从聊天流切换为状态流的前提。

### 8.1 增量执行优于一次性脚本生成

对于 Web 任务而言，页面结构、登录状态、异步加载、接口返回都带有强烈的不确定性。

因此 Hawker 并不要求模型一次性生成最终脚本，而是采用：

- 观察
- 试探
- 执行
- 校正
- 固化状态

这种模式本质上更接近 Notebook，而不是传统编译式程序生成。

### 8.2 事务式执行语义

执行引擎不仅负责运行代码，还负责保护状态质量。

核心原则是：

- 成功执行后，临时变量按协议提升到持久层
- 执行异常时，回滚当前步骤对命名空间的污染
- 失败只留下错误信息，不留下脏状态

这意味着上下文引擎中的 `Namespace Snapshot` 可以被视为可信环境，而不是“可能被异常步骤破坏过的半成品”。

### 8.3 状态流依赖执行层的纯化

如果执行层允许：

- 任意污染全局变量
- 异常后状态残留
- 无边界地引入不可控副作用

那么任何 Workspace 结构都会迅速失真。

因此，Execution Model 并不是上下文引擎的外部条件，而是它的内部支撑。

---

## 9. Tool Boundary

上下文引擎不是直接驱动浏览器、网络和文件系统，而是通过工具边界与外部世界交互。

这条边界的核心理念是：

- 模型看到的是能力接口，而不是任意运行环境
- 工具返回的是受控 Observation，而不是任意噪音输出
- 每类工具只暴露最必要的操作抽象

### 9.1 工具是受控能力，不是任意函数集合

Hawker 的工具系统不是为了“让模型能做更多事”，而是为了把可做的事情约束成高价值能力。

例如：

- 浏览器工具关注导航、点击、页面状态和网络日志
- HTTP 工具关注接口请求和下载
- 数据工具关注清洗、去重、保存和结构化摘要
- 核心动作关注 Observation、结果写入和任务结束

模型因此面向的是一组明确语义的动作，而不是一个无限开放的 Python 运行时。

### 9.2 工具返回必须服务于状态流

从上下文引擎角度看，一个好工具不只是“完成动作”，还必须：

- 输出足够验证当前动作是否成功的 Observation
- 避免把无关细节塞回 prompt
- 能与 Workspace 结构协作

这也是为什么 Hawker 的 DOM 工具会返回 `summary / diff / full`，而不是一律返回整页 DOM。

### 9.3 工具边界也是上下文边界

如果工具返回不受控：

- 一次 JS 提取返回全量 100 条对象
- 一次浏览器操作返回整页 DOM
- 一次异常返回完整长栈

那么上下文引擎很快就会失去效果。

因此 Tool Boundary 的真正作用之一，就是把外部世界输入压成适合状态流消费的高密度信息。

---

## 10. API-first Browser Intelligence

上下文引擎并不默认把浏览器 DOM 当作第一信息源。

Hawker 的浏览器智能核心原则是：

- API 优先
- 浏览器为观察与交互媒介
- DOM 是结构证据与最后兜底，而不是默认数据源

### 10.1 为什么不是 DOM-first

单纯 DOM-first 会带来两个问题：

1. token 成本过高。DOM 天生是最重的上下文类型。
2. 稳定性不足。选择器和结构容易变化，且难以直接表达数据生成路径。

因此，如果任务可以通过网络请求或接口重放完成，那么上下文引擎应尽量把“页面结构问题”转化为“接口和数据结构问题”。

### 10.2 浏览器的职责

在 Hawker 中，浏览器更多承担以下职责：

- 建立真实会话与登录状态
- 发现接口与网络请求
- 提供页面定位和交互能力
- 在接口不可用时提供 DOM 兜底抓取

这意味着浏览器不是上下文的终点，而是获取结构化事实的中间层。

### 10.3 上下文引擎如何约束浏览器智能

上下文引擎通过以下方式约束浏览器信息：

- 页面上下文分级为 `summary / diff / full`
- DOM 只能短期存在
- 失败时自动补诊断 DOM，而不是持续保留 full DOM
- 网络日志和页面结构共同构成策略选择依据

因此，API-first Browser Intelligence 的价值并不只是提效，更是从源头上控制上下文成本。

---

## 11. Observability and Auditability

上下文引擎要想可持续演进，必须可观测、可分析、可回放。

否则，“为什么这一步 token 暴涨”“为什么模型重复犯错”“为什么这一轮 prompt 很重”都只能停留在感觉层面。

### 11.1 可观测性不是附加品

在 Hawker 中，可观测性不是后加日志，而是内嵌在执行循环中的基础设施。

它覆盖：

- trace_id / run_id / step
- agent step span
- tool span
- LLM 请求 span
- 运行目录产物

这使得上下文引擎具备被工程化分析的能力。

### 11.2 审计视角下的 prompt 可回放

`llm_io.json` 的意义不仅是“保存一次请求”，而是把模型每一步真实看到的信息拆开保存：

- 最终发送的 messages
- Workspace 的内容
- Recent Messages
- 被折叠和省略的部分
- 传输层原始返回

这样在分析一个任务时，可以回答：

- 是哪个区块拉高了 token
- 是否 DOM Workspace 长期过重
- 是否 recent messages 保留过多
- 模型最终代码与 Observation 是否匹配

### 11.3 可观测性反过来驱动上下文优化

上下文引擎本身也受益于可观测性。

例如，DOM 生命周期折叠之所以能被准确修正，就是因为可以通过输入 token 曲线和 step 记录看到：

- full DOM 出现在某一步后
- 后续步骤 token 长期维持高位
- 修正 TTL 后 token 在下一轮回落

因此，Observability 并不是对上下文引擎的补充，而是它持续迭代的反馈回路。

---

## 12. Persistence Semantics

上下文引擎不仅组织 prompt，还定义哪些状态应该被正式保留，哪些只应短暂存在。

这就是持久化语义的核心问题。

### 12.1 正式数据与过程数据必须分离

在 Hawker 中，并不是所有运行中产生的内容都应该以同样方式保存。

应当区分：

- 正式采集结果
- 中间检查点
- Notebook 式执行记录
- LLM 输入输出记录
- 临时 DOM 证据

如果不做这层语义分离，系统很容易把“过程痕迹”误当成“正式结果”。

### 12.2 append_items 是唯一正式写入路径

Hawker 要求正式采集结果必须通过统一入口写入。

其意义在于：

- 结果去重有统一语义
- 数据数量能稳定进入 Runtime Snapshot
- 最终输出和中间检查点保持一致
- Workspace 可以确信哪些数据已经正式持久化

这本质上是在为上下文引擎建立“可信结果集”。

### 12.3 不同产物解决不同问题

当前 Hawker 的主要运行产物各自承担不同职责：

- `run.ipynb`
  - 面向 Notebook 式复盘
- `result.json`
  - 面向正式结果交付
- `llm_io.json`
  - 面向 prompt / response 审计
- checkpoint
  - 面向任务中断恢复

这套语义分层的价值在于：上下文引擎既能服务执行，又能服务结果交付和问题定位，而不会把所有内容混成一个统一但不可用的大文件。

---

## 13. 当前实现映射

当前实现已落地到以下模块：

- `hawker_agent/models/history.py`
  - `Notebook Workspace`
  - STM/LTM 拆分
  - `DOM Workspace`
  - DOM 生命周期折叠
- `hawker_agent/agent/compressor.py`
  - Observation 语义压缩
  - Namespace Snapshot 摘要
- `hawker_agent/browser/dom_utils.py`
  - 页面语义快照
  - `DOM Summary`
  - `DOM Diff`
- `hawker_agent/browser/actions.py`
  - `summary / diff / full` 返回模式
- `hawker_agent/tools/browser_tools.py`
  - `mode="auto"` 自动决策
  - 失败补诊断 DOM
- `hawker_agent/agent/runner.py`
  - 运行态指标同步
  - LLM I/O 记录
- `hawker_agent/storage/exporter.py`
  - `llm_io.json` 导出

---

## 14. 工程视角下的上下文流转路径

如果从工程实现角度看，Hawker 的上下文引擎并不是一个单点模块，而是一条贯穿执行主循环的状态流。

可以将一次 step 的核心路径概括为：

```text
task
  -> history.from_task(...)
  -> history.build_prompt_package()
  -> llm.complete(messages)
  -> parse_response(text)
  -> execute(code, namespace)
  -> tool calls / browser actions / observations
  -> history.record_step(...)
  -> next prompt package
```

真正关键的不是这条链路“存在”，而是每个节点都在为状态流服务。

### 14.1 Prompt 生成路径

从 prompt 侧看，数据流是：

```text
历史原始消息
  -> 压缩 / 提纯
  -> Notebook Workspace
  -> Recent Messages
  -> Final Prompt Messages
```

在这个过程中：

- `history` 负责把历史消息变成工作区状态
- `compressor` 负责把 Observation 和变量视图压缩成高密度表达
- `DOM Workspace` 负责把页面结构证据纳入可衰减状态

因此，最终发给模型的 prompt 不是直接拼接历史，而是“经过状态整形后的结果”。

### 14.2 执行反馈路径

从执行侧看，数据流是：

```text
LLM text
  -> thought / code
  -> executor
  -> namespace commit or rollback
  -> observation
  -> state markers update
  -> history.record_step(...)
```

这里有三个关键点：

1. 代码执行并不会直接修改长期上下文，而是先进入事务式执行层。
2. 只有成功后的状态才会进入下一轮 Workspace。
3. Observation 不直接等于下一轮 prompt，而是会经过摘要化和结构化再回写。

这使得执行反馈天然适合状态流，而不是原样拼回聊天历史。

### 14.3 浏览器上下文路径

浏览器侧的数据流目前是：

```text
browser action
  -> dom snapshot
  -> summary / diff / full
  -> tool auto strategy
  -> DOM Workspace
  -> TTL advance / fold
```

关键含义在于：

- 浏览器动作不直接把整页 DOM 塞回模型
- DOM 会先被语义化
- 再根据模式和当前状态选择注入形式
- 最终受生命周期控制

这条路径保证了浏览器上下文是工作区证据，而不是长期正文。

### 14.4 审计路径

可观测性侧的数据流是：

```text
prompt package + litellm response + parsed output + execution result
  -> state.llm_records
  -> save_llm_io_json(...)
```

因此，Hawker 的审计能力不是事后从日志反推，而是在执行过程中同步构建。

这使得工程团队可以针对同一 step 同时检查：

- 发给模型的真实 prompt
- 被折叠和省略的部分
- 模型原始返回
- 执行输出和副作用

### 14.5 为什么这条工程链路重要

只有当这几条流转路径被同时建立起来，上下文引擎才真正成立。

如果缺少其中任何一环：

- 没有事务式执行，Workspace 会失真
- 没有 DOM 生命周期，token 会持续高位
- 没有 prompt package 审计，就无法定位上下文膨胀原因
- 没有工具边界，状态流会被外部噪音污染

所以从工程角度看，Hawker 的上下文引擎不是一个“prompt 技巧”，而是一条完整的运行时状态管线。

---

## 15. 技术收益

这套设计的收益不只是“省 token”，更重要的是提升长期任务中的稳定性。

### 15.1 Token 更可控

通过：

- 历史提纯
- Observation 语义压缩
- DOM 自动分级
- DOM 生命周期折叠

模型输入不再只增不减，而会在关键步骤后自然回落。

### 15.2 推理焦点更稳定

模型不再被迫阅读大量历史 Trial-and-Error，而是直接面向：

- 当前状态
- 当前限制
- 当前变量
- 当前页面结构摘要

### 15.3 更易分析与继续优化

有了 `llm_io.json` 后，可以把上下文问题从“感觉很重”变成“可量化分析”的工程问题，例如：

- 哪一步是 DOM 拉高了 token
- 哪个 Workspace 区块占比最高
- 哪种动作最容易触发 full DOM

---

## 16. 后续方向

当前版本已完成核心结构改造，但仍有进一步优化空间：

1. Workspace 分块 token 统计
   - 单独统计 `Runtime / Namespace / DOM Workspace / Recent Messages` 的 token 占比
2. DOM Workspace 更细粒度折叠
   - 对 `diff` 做更短 TTL 或更激进的折叠
3. 自动策略自适应
   - 根据历史任务效果动态调整 `auto` 决策
4. Focused DOM
   - 在必要时支持局部 DOM 查询，而非直接请求 full DOM

这些方向都应建立在当前“状态流 + 结构化工作区 + DOM 生命周期控制”的基础上推进，而不是回到线性聊天历史的旧模式。

---

## 17. 总结

Hawker 的上下文引擎，本质上是在做一件事：

把大模型从“阅读冗长历史的被动执行器”，转变为“面向工作区状态做决策的主动调度器”。

这套技术的核心不在某一个压缩函数，而在以下组合：

- 状态流替代聊天流
- STM / LTM 分离
- Namespace Snapshot
- DOM Summary / DOM Diff / DOM Full 分级
- DOM 生命周期折叠
- 自动 DOM 策略
- 完整 LLM I/O 可观测性

这就是 Hawker 当前上下文系统的核心技术资产。
