# HawkerAgent 代码审查报告

> 仓库: https://github.com/huangpd/hawker.git  
> 审查范围: `hawker_agent/` 下全部模块 + 顶层 `run.py` / `case_run.py`  
> 审查日期: 2026-04-17

---

## 1. 总体印象

HawkerAgent 是一个以 LLM 驱动的异步浏览器 Agent 框架,整体设计**成熟度相当高**,比大多数"ReAct 类 Agent 玩具项目"要完整得多。值得肯定的设计要点:

- **清晰的分层架构**: `agent/` (核心循环) / `browser/` (Playwright+CDP) / `llm/` (litellm 封装) / `memory/` (SQLite 持久化) / `tools/` (注册表) / `storage/` (产物导出) / `observability/` (trace+日志),职责边界清楚。
- **三层命名空间 (`HawkerNamespace`) + 事务性代码执行** 的设计很扎实 —— 成功则 commit 到 session,失败则 rollback。<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/namespace.py" lines="37-127" /> <ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/executor.py" lines="107-250" />
- **顶层 `await` + coroutine 自愈执行**: 用 `PyCF_ALLOW_TOP_LEVEL_AWAIT` + `inspect.isawaitable` 兜底的写法巧妙,能容忍 LLM 漏写 `await` 的常见错误。<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/executor.py" lines="150-186" />
- **多层安全护栏**: 静态 import 黑名单、SSRF 防护、路径穿越校验、可配置的 Healer 变更比例上限。<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/executor.py" lines="24-58" /> <ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/tools/http_tools.py" lines="20-44" /> <ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/tools/data_tools.py" lines="81-91" />
- **分布式追踪**: 自研 `Span/trace` + Langfuse 集成 + logging 上下文注入,比只靠 print 调试好很多。
- **记忆系统 (`MemoryStore`)**: SQLite 持久化 + 站点/意图双维度召回 + 多样性裁剪 + "记忆引导 DOM 护栏",是比 mem0 类库更贴合浏览器场景的设计。
- **Evaluator + Healer 双子模型**: 用 `small_model_name` 做代码局部修复和最终交付门禁,成本敏感且架构清洁。
- **DOM 上下文的衰减模式** (summary/diff/full + TTL) 是非常有想法的 token 经济学优化。<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/models/history.py" lines="39-81" />

整体代码风格**干净**、注释**大量且中文化**(`Args/Returns` 格式一致)、命名**可读**。下面是需要改进的地方。

---

## 2. 严重问题 (建议优先修复)

### 2.1 `healing_records` 成功 case 双重记录(运行态数据一致性 Bug)

<ref_file file="/home/ubuntu/repos/hawker/hawker_agent/agent/healer.py" />  
即便 healing 成功后,`state.healing_records` 里留下的最后一条仍然是 `status="candidate"`,外部消费方(`save_llm_io_json`)无法区分"候选被接受"和"候选被丢弃"。建议在 L180-186 返回前再 append 一条 `status="accepted"` 或把上面 L145 的记录改为 `status="success"`。

### 2.2 `httpx.AsyncClient` 事件循环不安全

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/tools/http_tools.py" lines="17-52" />  
模块级全局 `_client` 单例,但 `httpx.AsyncClient` 绑定到**创建它的 event loop**。在长期运行且可能跨 `asyncio.run()` 调用(如测试 / CLI 重复执行)的场景下会报 `RuntimeError: Event loop is closed`。建议:
- 要么改为 per-run 创建并在 `run()` 结束时 `aclose()`;
- 要么通过 `contextvars` 存取 loop 关联的 client。

### 2.3 `_validate_url` SSRF 防护存在绕过

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/tools/http_tools.py" lines="26-44" />  
当前仅对"URL 里直接写的是 IP 字面量"生效。如果攻击者/模型传入 `http://evil.example.com/`,而该域名解析到 `169.254.169.254`(AWS metadata)或 `127.0.0.1`,会**直接绕过**。正确做法:
```python
import socket
for family, _, _, _, sockaddr in socket.getaddrinfo(hostname, None):
    ip = sockaddr[0]
    if _is_private_ip(ip):
        raise ValueError(...)
```
注意实际发起请求时还要**再次**解析+锁定 IP(防 DNS rebinding),或者使用 `httpx.Transport` 的自定义 resolver。

### 2.4 `_check_imports` 只做顶层 AST 扫描,存在逃逸

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/executor.py" lines="34-58" />  
以下写法全都能绕过黑名单:
```python
__import__("os").system("...")
eval("__import__('subprocess')")
getattr(__builtins__, "__import__")("os")
```
Python 沙箱**无法**通过纯黑名单做到真正安全。建议在 README 明确声明"本执行环境不是安全沙箱,仅做善意错误拦截",或者接入 `restrictedpython` / 进程级隔离(subprocess + seccomp / firejail)。这一点直接关系到把用户接入 HawkerAgent 做生产使用的风险边界,务必要有明确声明。

### 2.5 `eval()` 执行顶层代码 ≠ `exec()`,多语句赋值会静默丢失

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/executor.py" lines="151-186" />  
```python
compiled = compile(code, "<hawker-cell>", "exec", flags=...)
maybe_coro = eval(compiled, view)
```
`"exec"` 编译后用 `eval` 执行是合法的,但 `eval(compiled)` 返回值永远是 `None`(除了顶层 await 表达式 coroutine 的特殊情况);而 `view` 在注释里被描述为"同时作为 globals 和 locals"(L162-163)——这里只传了一个 dict 到 `eval`,实际 globals 和 locals 是同一个对象,这点是对的,但注释"模拟顶层执行环境"容易误导。实际上这段代码大部分时候能工作是因为 `ALLOW_TOP_LEVEL_AWAIT` + `exec` 模式在 CPython 里对最后一条表达式会产生隐式 coroutine(CPython 实现细节),**在 PyPy / 未来 CPython 不保证成立**。建议跟进官方文档更新,必要时改为 `types.CoroutineType` 显式构造或 `ast.PyCF_ALLOW_TOP_LEVEL_AWAIT` + `exec` + 检测全局变量 `_`/自定义标记。

### 2.6 `namespace.commit()` 变量提升可能无意持久化敏感数据

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/namespace.py" lines="73-116" />  
规则是"名字 > 1 字符 + 不以 `_` 开头 + 非模块"。如果 LLM 写了 `cookies = await get_cookies()` 或 `token = "xxx"`,这些会被**永久留在 session**,然后被 `llm_io.json` 完整落盘(`save_llm_io_json` 里面又会包括 prompt),产生**凭证写入磁盘**的风险。建议:
- 对长度超过阈值的 str 做 `***redacted***`;
- 明确在 Prompt 里规定敏感值不要赋给顶层变量;
- 或者提供显式的 `namespace.mark_sensitive(var_name)` API。

### 2.7 `copy.deepcopy(namespace.session)` 可能很昂贵

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/executor.py" lines="137-142" />  
当 `all_items` 累积到数千条后,**每一步**都深拷贝一次,会出现 O(step × items) 的复制开销。对长任务影响明显。建议:
- 用 "append-only + 事务游标" 替代深拷贝(记录 session dict 的 key 快照 + 新增 key 集合,rollback 时只 `del` 新 key 即可,大多数场景足够);
- 或在 commit/rollback 时只对"本步骤真正改过的"变量做处理。

### 2.8 `final_answer` 执行后仍继续走完本步

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/runner.py" lines="507-568" />  
当 `final_answer_requested` 被设置并通过 evaluator 放行后,`state.done = True`,但本步后续的 `record_step` / `llm_records.append` / `_inject_reflection_prompts` 仍然会执行,接着才在 L646 检查 `state.done`。这里没有 bug,但浪费计算且让日志出现"任务完成但仍注入反思 prompt"的奇怪痕迹。建议 L568 之后立刻 `break`/`return _finish("done", step)`。

---

## 3. 中等问题

### 3.1 `LLMClient._complete_internal` 超时和重试配置硬编码

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/llm/client.py" lines="196-207" />  
- `timeout=180`、`max_tokens=2500`、3 次 429 重试固定 5 秒退避——都没有通过 `Settings` 暴露。
- 对 Gemini 强制 `temperature=1.0` 是 litellm 的限制,逻辑应该改用"模型能力表"而不是 `"gemini" in model.lower()`(匹配会误伤 `claude-vertex-gemini-router` 等命名)。

### 3.2 `parse_response` 的 JS 命名块注入存在注入风险

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/parser.py" lines="54-60" />  
```python
escaped = content.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
code_parts = [f'{var_name} = "{escaped}"', ...]
```
手写 JS 字符串转义 + Python 字符串拼接 = 历史上最常见的 RCE 类 bug 模式。`var_name` 来自 LLM 生成的 markdown,没有做**合法 Python 标识符**校验,如果模型生成 ````js __import__('os').system ... ``` 作为 `var_name`,会在 Python 层直接执行(虽然之后被黑名单拦截,但设计上脆弱)。建议:
```python
if not var_name.isidentifier():
    continue
python_blocks.append(f"{var_name} = {json.dumps(content)}")
```
用 `json.dumps` 做字符串字面量生成,比手写 escape 可靠得多。

### 3.3 `get_settings()` 用 `lru_cache` 实现单例,但 `Path` 默认值在类定义时求值

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/config.py" lines="62-65" />  
```python
scrape_dir: Path = Path("hawker_file")
memory_db_path: Path = scrape_dir / Path("memory.db")
```
`memory_db_path` 默认值**在类定义时**用 `Path("hawker_file")` 计算,当用户通过环境变量改 `scrape_dir` 时,`memory_db_path` **不会自动跟随**。这是 pydantic-settings 的常见陷阱。修复:
```python
@model_validator(mode="after")
def _sync_paths(self):
    if self.memory_db_path == Path("hawker_file/memory.db"):  # default
        self.memory_db_path = self.scrape_dir / "memory.db"
    return self
```

### 3.4 `MemoryStore.search` SQL 评分表达式与 `_prune_matches` 逻辑分散

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/memory/store.py" lines="516-593" />  
评分逻辑(100+25+10+confidence*10+seen_count)硬编码在 SQL 里,后续调权非常痛苦。建议抽出 `compute_memory_score(entry, task_intent, site_keys) -> float` 纯 Python 函数,SQL 只负责召回候选集,Python 层打分+裁剪,一致且可测试。

### 3.5 `build_raw_code_memories` 抽取启发式太脆弱

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/memory/store.py" lines="287-369" />  
只按"步骤活动度"筛成功代码,但不识别代码是否"具备通用性"(例如 URL / session cookies / 一次性 token 被写死了)。这会导致记忆库被脏样本污染后劣化。建议:
- 在入库前正则屏蔽/模板化常见 secrets(Bearer、cookie、session id);
- 记录 `source_task` 用于**负样本学习**;
- 对 `detail` 做最小长度 / AST 可解析性校验。

### 3.6 `observability.py` 的 `_TRACE_PROCESSORS` 全局 list 非线程安全

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/observability.py" lines="23-86" />  
多个 Agent 并发运行时 `add_trace_processor` / `remove_trace_processor` 会踩同一个 list,而 `trace()` 上下文在 finally 里迭代它。建议改成 `contextvars.ContextVar[tuple[TraceProcessor, ...]]` 或加锁。

### 3.7 `ClearableList` 只代理了一部分方法

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/namespace.py" lines="161-184" />  
它**继承自 `list`**(把底层数据复制进去),又持有 `_data` 引用,调用 `.append()` 会走到 `list.append`(写进**父类那份副本**)而不是 `_data`。模型如果写 `all_items.append(x)` 表面上能跑但**完全无效**——很容易在真实任务里造成静默丢数据。必须重写 `append/extend/insert/__setitem__/pop/remove/__iadd__` 并同步 `_data` 和 `_store`。或者干脆不继承 list,只实现 `__getitem__/__iter__/__len__/__repr__/append/extend/clear`。

### 3.8 `execute()` 内部二层 `bind_log_context`

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/executor.py" lines="129-134" />  
```python
log_context = state.bind_log_context(step) if state else contextlib.nullcontext()
with log_context:
    with trace(f"execute_code_{step or 'anon'}") as span:
        step_context = state.bind_log_context(step) if state else contextlib.nullcontext()
        with step_context:
```
第二层 `bind_log_context` 是多余的(`trace` 内部已经维护了 step 名,且外层已经 bind),建议删掉。

### 3.9 无 TTL 清理 / 无大小上限的 SQLite 记忆库

`MemoryStore._init_db` 没有任何 VACUUM/TTL/上限机制(仅从 outline 推测,未贴出)。长期跑之后库会持续膨胀 + 脏样本比例上升。建议:
- 按 `updated_at` 或 `seen_count` 做软 LRU 裁剪;
- 增加 `MemoryStore.prune()` 和对应定时任务/CLI 命令。

### 3.10 `run.py` 顶层的 TASK 覆盖写法不规范

<ref_file file="/home/ubuntu/repos/hawker/run.py" />  
连续 12 个 `TASK="""..."""` 全部赋值到同一变量,只有最后一个生效。这是开发者本地草稿文件,不应该 commit 到主分支。建议:
- 抽到 `examples/tasks/*.md`;
- 让 `run.py` 接受 `--task-file <path>` 参数;
- 默认 `case_run.py` 应作为用户入口,`run.py` 留作内部调试。

---

## 4. 轻微问题 / 风格改进

| 位置 | 问题 |
|---|---|
| `agent/runner.py` L399-402 | 函数内 `from ... import ...` 四行,应提到文件顶部(存在潜在循环导入的话注明 `# deferred: avoid circular import`)。 |
| `agent/runner.py` L679 中文断言/日志 | "任务在达到最大步数 (%d) 后终止" 推荐保留但追加 trace_id 方便排查。 |
| `browser/netlog.py` | JS 注入字符串里硬编码了很多 hostname 黑名单,应改为可配置 set。 |
| `browser/actions.py` L261-288 等 | `await session.raw.navigate_to(url); await asyncio.sleep(1)` 的"睡一秒"是魔法数字,建议改成 `wait_until="networkidle"` 或条件等待。 |
| `browser/cdp.py` | `run_js` 在 "Illegal return statement" 时自动包裹 IIFE,好设计;但没有捕获 `exceptionDetails` 栈,错误信息对模型不够丰富。 |
| `models/history.py` | 多处 `self._messages[1:]` / `self._messages[1:-4]` 硬编码,建议封装 `iter_recent(n)` / `iter_middle()`。 |
| `models/history.py` L178 | `inject_dom` 仅保存 pending_dom;如果同步多次调用后者覆盖前者,会丢失中间状态。建议改成队列或显式 merge。 |
| `storage/exporter.py` `_to_jsonable` | 用 `repr(value)` 做最终 fallback 会把 `<function>` 塞进 llm_io.json,后续重放时噪音大。建议 `"<non-serializable:type>"`。 |
| `storage/logger.py` L65 | `logger.info("✅ 运行目录就绪...)` 含 emoji,若下游日志消费是 ASCII-only 管道会乱码。 |
| `tools/registry.py` L42/L57 | `kwargs = {k: str(v)[:200] ...}` 中 value 被截断但没标记 `...truncated`,会被 Langfuse 上显示成看起来完整的数据。 |
| `tools/http_tools.py` L216 | `data = json_lib.loads(text)` 与参数名 `data` 同名,shadowing;建议改名 `parsed`。 |
| `config.py` L64 | `memory_db_path: Path = scrape_dir / Path("memory.db")` —— 如上 3.3,是在类体内对 `Path` 表达式求值,容易误导。 |
| 整体 | 日志/注释中英文混杂(大量中文) + emoji(`✨🚀✅`),对国际化协作不友好;生产建议保留中文注释但把**日志字符串**英文化。 |

---

## 5. 架构层面的建议

### 5.1 Runner 单文件 684 行,职责过多

<ref_file file="/home/ubuntu/repos/hawker/hawker_agent/agent/runner.py" />  
`run()` 把"初始化 / 记忆加载 / 浏览器 / 工具注册 / 主循环 / 终止处理"全部放在一个异步函数里,导致局部作用域管理 `state`, `history`, `namespace`, `br`, `reg`, `stats_proc`, `llm`, `memory_store`, `cells`, `no_progress_steps` 这 10+ 个变量。建议拆成一个 `AgentSession` 类:
```python
class AgentSession:
    def __init__(self, task, cfg): ...
    async def __aenter__(self): ...  # init tools, memory, browser
    async def __aexit__(self): ...   # finish, flush_langfuse, cleanup
    async def step(self, i): ...     # 单步
    async def run(self, max_steps): # 循环
```
这样也方便**外部调用者接入自定义主循环**(例如人机协同 Agent / 并发多 Agent Orchestrator)。

### 5.2 Tool 注册与执行耦合

`ToolRegistry.register()` 会**立即包装**成 trace wrapper 并写入 `_tools`。如果下游想复用一组工具到不同 trace 系统(如 OpenTelemetry),需要把"包装"和"注册"解耦。可以让 `register` 只存原函数,`as_namespace_dict` 按需生成带 trace 的版本。

### 5.3 `history.record_step` 通过 `self._notebook_mode_enabled = True` 隐式切模式

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/models/history.py" lines="322-323" />  
第一次 `record_step` 被调用时才打开 Notebook 模式,是个**隐式副作用**,让 `build_prompt_package` 的行为"第一步和其他步不一样"。建议把 `notebook_mode_enabled` 在 `from_task` 时显式决策,并文档化。

### 5.4 `ensure` / `parse_http_response` 等低层函数同时注册为工具和预注入到 namespace

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/tools/data_tools.py" lines="109-112" /> <ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/namespace.py" lines="261-266" />  
同一个函数既在 `ToolRegistry` 里 `expose_in_prompt=False`,又在 `build_namespace` 中用 `sys_` 前缀注入。两条路径容易产生"改了一个忘改另一个"的 drift,建议统一由 `ToolRegistry` 负责,不要在 `build_namespace` 里再手动注入。

### 5.5 评估器 / Healer 缺少熔断

`final_evaluator_enabled` / `healer_enabled` 是静态配置,没有运行时熔断:当 small model 连续失败 N 次时应该自动降级为直通,避免 evaluator 成为 agent 陷在"不断被拒绝→重新提交"死循环里的祸首(目前看 L508-553 确实可能发生)。

### 5.6 缺少 Agent 并发支持

所有单例(`_client`,`_TRACE_PROCESSORS`,`_LOG_CONTEXT` 这个还好是 contextvar)与工具 wrapper 都假定"一进程一 Agent"。若未来计划多 Agent 并发(见 README 的 Roadmap),需要:
- `LLMClient` 不要做 Langfuse `flush()` 全局副作用;
- `run_dir` / `log_dir` 必须严格 scoped;
- `MemoryStore` 需要确认 SQLite 并发写入策略(WAL + 适当 retry)。

### 5.7 测试
`tests/` 下文件目录很全(`test_executor_async.py`, `test_full_async_integration.py`, `test_memory.py`, `test_dom_context.py` 等),这一点已经**超过同类开源项目平均水平**。但从命名看覆盖还偏"功能单元",建议补充:
- **LLM mock 下的 runner 端到端**: 用确定性的 FakeLLM 回放一组固定 trace,断言 `state.items` / `stop_reason` / `run_dir` 产物一致;
- **sandbox 逃逸测试**: 用已知的绕过 pattern 作为 regression suite;
- **Memory 回放测试**: 保证一次跑留下的记忆可以被下一次跑召回到 top1。

---

## 6. 安全审计小结

| 类别 | 状态 | 备注 |
|---|---|---|
| 代码执行沙箱 | ⚠ 薄弱 | 仅黑名单 + 静态 AST,非真沙箱。建议在 README 明示。 |
| SSRF | ⚠ 部分 | 仅 IP 字面量生效,域名绕过存在。需加 DNS 解析后校验。 |
| 路径穿越 | ✅ 良好 | `_safe_join` 实现正确。 |
| 命令注入 | ✅ 无可执行入口 | 已禁用 `os`/`subprocess`/`shutil`。 |
| 凭证泄露 | ⚠ 风险 | session namespace 会持久化 cookies/tokens 并写入 llm_io.json。 |
| 日志脱敏 | ❌ 缺失 | headers/cookies/prompt 里的 Bearer token 都会原样落盘。 |
| 依赖供应链 | - | `pyproject.toml` 锁了主要依赖,建议加 `pip-audit` / `safety` CI。 |
| Prompt 注入 | ⚠ 未处理 | 浏览器抓到的 DOM 文本直接进 system prompt,恶意页面可诱导模型执行任意代码——当前架构下无现成防御,但至少应在 Healer/Evaluator 层避免把 DOM 内容当"可信指令"。 |

---

## 7. 性能/成本优化机会

1. **每步 deepcopy 的 session** (见 2.7) —— 最大优化空间。
2. `count_tokens` 走 litellm,**每步重算整段 history** 的 token 数,对大 prompt 开销不小。可缓存 "message id → token count"。
3. `emit_tool_observation` 把 tool 输出拼成字符串塞 trace,配合 Langfuse 远程上报可能成为主循环瓶颈;需要异步化或限速。
4. `MemoryStore.search` 用 `LIMIT limit*4` 再 Python 裁剪,在百万条记忆时会退化。可以对 `(site_key, task_intent, updated_at)` 建复合索引(从 outline 看只有基础表,未确认)。
5. `netlog.py` 客户端保存最近 100 条请求,**每条 body 5KB**,频繁翻页的页面会 CPU 打满(JS string push/shift)。可以配置为"只保存 xhr/fetch + content-type 包含 json/text"。

---

## 8. 推荐的改进优先级

| P | 建议 |
|---|---|
| **P0** | 3.3 `ClearableList.append` 静默丢数据 + 2.7 deepcopy 性能 + 2.3 SSRF 绕过 + 2.6 凭证落盘 |
| **P1** | 2.1 healing_records 记录、2.2 httpx client loop、2.4 沙箱风险声明、2.8 final_answer 后多余执行、3.2 parser.py JS 注入硬编码 |
| **P2** | 3.3 `memory_db_path` 默认值、3.4 评分抽函数、3.7 Evaluator 熔断、5.1 `AgentSession` 重构、4 的所有风格项 |
| **P3** | 文档化沙箱边界 / 加入 pip-audit CI / 写端到端 FakeLLM 测试 |

---

## 9. 结论

HawkerAgent 的工程水准**已经超过绝大多数开源 agent 项目**,尤其在 "可观测性 / 事务性执行 / 记忆系统 / 交付门禁" 这几个点上做了认真思考。主要风险集中在:

1. **沙箱边界未明示**(会让使用者误以为它是真沙箱),
2. **少量正确性 bug**(`ClearableList.append`、healing_records、final_answer 后继续走步),
3. **部分性能隐患**(deepcopy、SQLite 记忆无 TTL、httpx 全局 client),
4. **安全深化空间**(SSRF 绕过、敏感变量持久化到 llm_io.json)。

修完上面 P0/P1,项目完全具备"面向团队内部 / 可信任务"的生产能力。如果要对外服务,还需要把沙箱替换为进程级隔离 + Prompt Injection 防线 + 日志脱敏 pipeline。

如需我对任何一条建议做更深入的分析(例如 3.3 的最小可运行补丁、或 2.7 的事务性游标实现),请直接告诉我。
