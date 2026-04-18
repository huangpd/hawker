# Agent Compiler 与 Fast/Slow Healing 研究方案

## 背景

Hawker 当前的执行链路是典型的交互式 Agent：

1. `runner` 组装 prompt
2. 主模型输出 `Thought + Code`
3. `executor` 执行代码
4. 产出 Observation
5. 再回到主模型做下一步决策

这条链路的优点是灵活，但在两个地方已经接近天花板：

- **长任务成本过高**：翻页、列表遍历、批量详情采集会让主模型变成昂贵的循环控制器
- **小错误修复代价过高**：一个局部 `SyntaxError` 或 `IndexError`，现在也会污染主模型上下文并触发整轮再推理

因此，下一阶段的优化目标不是继续微调 prompt，而是把 Hawker 从“逐步交互 Agent”推进到“探索 + 编译 + 自愈”的混合执行框架。

---

## 目标

本研究聚焦两个方向：

1. **Agent -> Compiler**
   - 让主模型只负责探索、识别范式、产出可复用策略
   - 一旦确认操作模式稳定，切换到“编译态”，由本地脚本无 Token 执行剩余流程

2. **Fast / Slow Healing**
   - 主模型只处理真实决策问题
   - 将低级代码报错修复旁路给廉价小模型，避免污染主上下文

这两个方向的共同目标是：

- 降低主模型调用次数
- 降低长任务 token 成本
- 提高长循环任务的吞吐
- 保持主上下文干净

---

## 一、范式跃迁：Agent -> Compiler

### 1.1 当前问题

目前 Hawker 的主链路本质上是：

`思考 -> 生成一小段代码 -> 执行 -> 观察 -> 再思考`

这意味着：

- 翻 100 页，可能要 100 次 LLM 决策
- 采 100 个详情页，可能也要几十到上百次主模型介入
- 大模型在长任务中大量消耗在“重复控制”而不是“高价值决策”

在商业化场景下，这种复杂度近似是：

- **LLM 成本 = O(页面数 / 交互数)**

这不合理。

### 1.2 目标形态

把任务拆成两种状态：

#### Explore（探索态）

由主模型负责：

- 理解页面结构
- 发现数据 API / DOM 规则 / 翻页规则
- 验证前 1 到 3 页是否稳定
- 明确字段 schema
- 判断任务是否可编译

#### Compile（编译态）

当系统判断模式已稳定时，不再让主模型继续控制循环，而是让其输出：

- 一段无需继续调用主模型的纯 Python / Playwright / Hawker runtime 脚本

然后由系统本地直接执行这段脚本，完成：

- 剩余 97 页翻页
- 剩余批量详情采集
- 统一保存、清洗、去重

此时主模型的角色不再是“逐页控制器”，而是“爬虫工程师”。

---

## 1.3 触发条件：何时从 Explore 切到 Compile

切换条件必须保守，不宜一开始就过早进入编译态。

建议分三层门控：

### 条件 A：模式稳定

最近 2 到 3 步满足以下任意模式之一：

- `提取数据 -> 点击下一页 -> 提取数据`
- `发现 API -> fetch(page=2) -> fetch(page=3)`
- `进入详情页 -> 提取字段 -> 返回列表 -> 重复`

也就是：**动作图已经明显重复**。

### 条件 B：数据结构稳定

最近连续两步采集出的数据：

- 字段名一致
- 样本结构一致
- 去重规则稳定

### 条件 C：页面/接口路径稳定

满足任一：

- DOM 容器和翻页按钮规则稳定
- 已发现可重放 API
- 详情页 URL 模式稳定

只有同时满足 A + B + C，才允许触发编译态。

---

## 1.4 编译产物应该长什么样

不要让模型自由输出大段散乱脚本。必须要求其输出结构化编译结果。

建议 schema：

```json
{
  "kind": "compiled_plan",
  "mode": "pagination_dom",
  "goal": "抓取剩余分页数据",
  "assumptions": [
    "当前已定位列表项容器",
    "当前翻页按钮规则稳定"
  ],
  "script_python": "...",
  "expected_outputs": {
    "items_append": true,
    "checkpoint": true
  },
  "safety": {
    "max_pages": 100,
    "stop_on_duplicate_rounds": 3
  }
}
```

最关键的是 `script_python`，它必须：

- 不依赖主模型后续推理
- 能直接在当前 `namespace` 环境中运行
- 继续使用 Hawker 现有工具与 helper

例如：

- `await fetch(...)`
- `await nav(...)`
- `await click_index(...)`
- `await js(...)`
- `await append_items(...)`
- `await save_checkpoint(...)`

而不是要求系统再解释一遍。

---

## 1.5 编译态执行器的落点

建议新增一层：

- `hawker_agent/agent/compiler.py`

职责：

1. 检测是否满足编译触发条件
2. 构造“编译请求 prompt”
3. 调主模型输出结构化 `compiled_plan`
4. 校验脚本是否安全、是否引用非法对象
5. 交给 `executor` 直接本地执行

### 与现有模块的关系

#### `runner.py`

新增逻辑：

- 在每步结束后检查 `step_meta + state + llm_records`
- 若满足模式稳定，调用 `maybe_compile(...)`
- 若编译成功，进入 `compiled_execution` 分支，而不是继续常规 step loop

#### `models/state.py`

新增字段建议：

```python
compile_mode: bool = False
compiled_script: str | None = None
compiled_plan_kind: str | None = None
compiled_from_step: int | None = None
compile_attempts: int = 0
```

#### `models/history.py`

应增加一类 workspace 摘要：

- `Compilation Workspace`

内容包括：

- 已验证的循环模式
- 可编译原因
- 编译后脚本摘要

这样主模型后续知道系统已经进入编译态。

---

## 1.6 编译态的第一期 MVP

不要一上来支持所有模式。第一期只做最有价值的两类：

### MVP-A：分页列表编译

适用：

- 列表页翻页稳定
- 采集字段稳定

典型任务：

- 新闻列表
- 商品列表
- 项目榜单

### MVP-B：API 分页编译

适用：

- 已抓到稳定数据 API
- 参数规律明确

典型任务：

- JSON 列表接口
- page / offset / cursor 型接口

这两类占掉大量高频任务，收益最大。

---

## 1.7 编译态的失败回退

编译不是一条单向路。

若编译脚本执行中出现：

- 页面结构突变
- API 失效
- 重复数据超阈值
- DOM 无法继续定位

系统应：

1. 终止编译态
2. 将失败原因压缩写入 `Long-Term Memory`
3. 回退到 Explore 状态
4. 让主模型重新接管

这要求编译态和探索态可以双向切换，而不是“一次编译到底”。

---

## 二、Fast / Slow Healing：廉价修复专员

### 2.1 当前问题

现在 `executor.execute(...)` 一旦报错：

- 会回滚 namespace
- 返回完整 traceback
- runner 再把这个错误加入 history
- 主模型看到报错后重新思考

问题是很多错误属于非常低级的本地代码修复问题，例如：

- 引号没闭合
- 变量名拼错
- `None` 判空缺失
- `IndexError`
- `KeyError`
- 简单的缩进或括号问题

让主模型反复为这类问题付费，性价比极低。

### 2.2 目标形态

引入两层思考：

#### Slow Brain

当前主模型，负责：

- 高层策略
- 工具选择
- 页面理解
- 编译决策

#### Fast Healer

廉价小模型，负责：

- 局部代码修补
- 不改任务策略
- 不重写整段逻辑
- 只修当前报错 cell

这类模型可以是：

- GPT-4o-mini
- Claude Haiku
- 更便宜的 code model

---

## 2.3 Healing Loop 的执行流程

建议在 `executor.py` 报错后，先不要直接把 traceback 丢回主 history，而是：

1. 捕获异常
2. 判断是否属于“可局部修复”的错误
3. 若是，则构造修复请求：
   - 原代码
   - traceback
   - 可用变量摘要
4. 调小模型尝试修复
5. 最多 3 次
6. 若修复后成功，则主模型完全不知道这次错误
7. 若修复失败，再把原始错误交回主模型

这就是：

- **Fast loop 先救火**
- **Slow loop 只处理真正复杂问题**

---

## 2.4 哪些错误适合走 Healing

推荐只接这几类：

- `SyntaxError`
- `NameError`
- `KeyError`
- `IndexError`
- `TypeError`
- `AttributeError`

但要排除：

- 安全限制错误
- 工具权限错误
- 明显策略错误
- 多步状态依赖错误

换句话说：

- Healing 只修 **局部代码**
- 不修 **全局决策**

---

## 2.5 Healing 模块落点

建议新增：

- `hawker_agent/agent/healer.py`

职责：

1. 判断异常是否适合旁路修复
2. 构造修复 prompt
3. 调用小模型
4. 返回修复后的代码

建议 API：

```python
async def try_heal_code(
    *,
    code: str,
    error: str,
    namespace_snapshot: dict[str, str],
    max_attempts: int = 3,
) -> str | None:
    ...
```

若返回 `None`，说明 healing 失败，回退主模型路径。

---

## 2.6 与现有 executor 的接法

当前 `executor.execute(...)` 是：

- 编译
- 执行
- 成功 commit
- 失败 rollback

建议在失败路径里插入：

1. rollback
2. 调 `maybe_heal(...)`
3. 若得到修复代码：
   - 再执行一次 `execute(...)`
   - 标记这是 healing retry
4. 若仍失败且次数未超限，继续 healer
5. 超限后再向上抛给主链路

重要的是：

- healer 重试不应污染主 history
- 但应保留到 `llm_io.json` 或专门日志中，方便审计

---

## 2.7 Healing 的 prompt 原则

必须强约束，不让小模型“改策略”。

只允许它：

- 修语法
- 修变量引用
- 补判空
- 修局部结构

明确禁止：

- 改任务目标
- 改字段定义
- 改数据 schema
- 改主要抓取路径

它应该像一个“代码修复专员”，不是第二个 agent。

---

## 三、两个方向如何协同

这两个方向不是独立的，它们可以组合：

### 组合路径

1. 主模型处于 Explore 状态
2. 局部代码报错时，由 Fast Healer 清理
3. 一旦任务模式稳定，触发 Compile
4. 编译脚本执行中若出现局部小错误，也优先走 Healing
5. 只有真正结构变化或策略失效时，才回到主模型

最终目标是形成：

- **主模型：少调用，但负责关键判断**
- **小模型：高频局部修复**
- **本地运行时：承担绝大多数重复执行**

---

## 四、一期实施建议

### Phase 1：Fast Healing MVP

先做这个，因为收益快、风险低。

实施项：

1. 新增 `agent/healer.py`
2. `executor` 增加最多 3 次局部修复循环
3. 只支持 5 到 6 类典型异常
4. 修复记录写入日志，但不进入主 history

预期收益：

- 减少无意义主模型重试
- 保持 prompt 干净

### Phase 2：Compile MVP

再做这个，聚焦两种模式：

1. 列表分页编译
2. API 分页编译

实施项：

1. 新增 `agent/compiler.py`
2. 在 `runner` 中增加模式检测与编译切换
3. 新增 `compiled_plan` schema
4. 编译失败可回退

预期收益：

- 大幅降低长列表任务 token 消耗
- 显著提高多页任务吞吐

---

## 五、关键风险

### 风险 1：过早编译

如果系统过早进入编译态，可能把不稳定规则固化，导致批量失败。

应对：

- 编译门槛宁严勿松
- 首期只支持稳定模式

### 风险 2：Healer 越权改策略

小模型如果重写逻辑，会带来隐藏行为漂移。

应对：

- 强 schema
- 强 prompt 约束
- 限制只改当前 cell

### 风险 3：编译脚本变成“第二条执行系统”

如果脚本能力过强，会绕开现有工具约束。

应对：

- 编译脚本仍只能运行在当前 `namespace` 和工具边界内
- 不开放额外高危能力

---

## 六、结论

Hawker 下一阶段最值得投入的方向，不是继续堆更多工具或更多记忆字段，而是把执行链路从单一的交互式 Agent，升级为：

- `Explore`：高价值探索与决策
- `Compile`：低成本本地批量执行
- `Heal`：廉价局部错误修复

这会让 Hawker 从“会操作浏览器的大模型”真正进化成“带编译器和自愈能力的爬虫运行时”。

如果后续进入实现阶段，推荐顺序是：

1. 先做 Fast Healing MVP
2. 再做分页 / API 两类 Compile MVP
3. 最后才做更复杂的详情页批量编译和多阶段编译
