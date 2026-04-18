# HawkerAgent 代码审查 — 增量 Review (commit 019c100)

> 基线: `623fc12` (上一次审查)  
> 当前: `4234bc7` (main) / 核心变更: `019c100 Harden HTTP tools and redact exported records`  
> 范围: `observability.py`, `storage/exporter.py`, `tools/http_tools.py`, `tests/test_tools.py`

---

## 1. 已修复的问题 (good)

| 上次编号 | 状态 | 说明 |
|---|---|---|
| **2.2** httpx 事件循环绑定 | ✅ 已修 | 改为 `dict[int, AsyncClient]` 按 loop 缓存 (`_clients_by_loop`),并新增 `close_http_clients()` 清理函数。<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/tools/http_tools.py" lines="82-99" /> |
| **2.3** SSRF DNS 绕过 | ✅ 已修 | `_resolve_host_ips` + DNS 解析后的 IP 二次校验;测试用 `fake_resolve_host_ips` 覆盖了 `169.254.169.254` 的 DNS 场景。<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/tools/http_tools.py" lines="47-79" /> |
| **2.6** 凭证落盘 | ⚠ 部分修 | 新增 `_SENSITIVE_KEYWORDS` + `_is_sensitive_key`,在 `_to_jsonable` 中按 key 脱敏为 `***redacted***`。但见 §2.1 的重要副作用。<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/storage/exporter.py" lines="16-60" /> |
| **4** `_to_jsonable` 的 `repr(value)` fallback | ✅ 已修 | 改为 `<non-serializable:{type}>`。<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/storage/exporter.py" lines="78" /> |
| **3.8** `trace()` 重复 `bind_log_context` | ⚠ 部分修 | `observability.py` 里把 `step=name` 改成 `step=existing_ctx.step or name`,外层 step 不再被内层覆盖。executor 里那层 `bind_log_context` 还在,仍可再精简。<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/observability.py" lines="144-151" /> |
| `http_json` 变量 shadowing (`data`) | ✅ 已修 | 改名 `parsed`。<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/tools/http_tools.py" lines="263-269" /> |
| 测试 | ✅ 补齐 | 新增 `test_save_llm_io_json_redacts_sensitive_fields_and_non_serializable_values` 和 `test_validate_url_rejects_private_ip_resolved_from_dns`,质量不错。 |

整体方向**正确**,修复完整度合格。

---

## 2. 新引入的问题 (需要在合并前处理)

### 2.1 🔴 **P0 回归**: 脱敏关键字用**子串匹配**,误伤大量合法字段

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/storage/exporter.py" lines="16-34" />

```python
_SENSITIVE_KEYWORDS = ("authorization", "token", "secret", "password",
                       "passwd", "cookie", "session", "api_key", "apikey",
                       "access_key", "refresh_key")

def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower()
    return any(keyword in normalized for keyword in _SENSITIVE_KEYWORDS)
```

`keyword in normalized` 是子串判断,这会把下面这些**非敏感**字段全部误脱敏:

| llm_records 里实际字段 | 含的子串 | 被误脱敏结果 |
|---|---|---|
| `input_tokens` (runner.py L463, L619) | `token` | `"input_tokens": "***redacted***"` |
| `output_tokens` | `token` | 同上 |
| `cached_tokens` | `token` | 同上 |
| `total_tokens` | `token` | 同上 |
| `tokens` (healer.py L154) | `token` | 整个子 dict 被替换为字符串 |
| `session_vars` / `namespace.session` 里任何 key 包括 `session` 作为前缀 | `session` | 误脱敏 |
| `browser_session_id` 等 | `session` | 误脱敏 |

**直接后果**: `llm_io.json` 里所有 per-step token 消耗与成本字段都变成 `"***redacted***"`,**成本/预算/回放分析完全失效**,这是一项重要的可观测性数据。同时 healer 的 `"tokens": {...}` 子 dict 也是以整体被替换,连 input/output/cached 细分都看不到。

运行完后应该能在 `llm_io.json` 里直接 grep 验证:
```bash
python -c "import json; r=json.load(open('...llm_io.json'))['records'][0]; print(r['llm_response'])"
# 预期看见 input_tokens=123,实际会看到 "***redacted***"
```

**修复建议** (二选一,推荐 A):

A. 精确匹配白名单 / 正则:
```python
_SENSITIVE_PATTERNS = re.compile(
    r"^(?:authorization|cookie|set[-_]cookie|"
    r"(?:api|access|refresh|secret)[-_]?key|"
    r"x[-_]?api[-_]?key|password|passwd|"
    r"auth[-_]?token|bearer[-_]?token|"
    r"session[-_]?id|sessionid|csrf[-_]?token)$",
    re.IGNORECASE,
)
def _is_sensitive_key(key: str) -> bool:
    return bool(_SENSITIVE_PATTERNS.match(key.strip()))
```

B. 保留子串匹配但加白名单例外:
```python
_WHITELIST = {
    "input_tokens", "output_tokens", "cached_tokens", "total_tokens",
    "tokens", "token_stats", "session_vars",
}
def _is_sensitive_key(key: str) -> bool:
    key = key.strip().lower()
    if key in _WHITELIST:
        return False
    return any(k in key for k in _SENSITIVE_KEYWORDS)
```

同时建议补一条**负向回归测试**:
```python
def test_save_llm_io_json_does_not_redact_token_stats(tmp_path):
    path = save_llm_io_json(tmp_path, "t", records=[
        {"llm_response": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}}
    ])
    data = json_mod.loads(path.read_text())
    assert data["records"][0]["llm_response"]["input_tokens"] == 100
```

### 2.2 🟡 **P1**: 脱敏只看 key,不看 value —— 真正的 Bearer token 仍会落盘

最能掉落凭证的位置其实是 `llm_response.raw` / `llm_response.text` / `prompt_package` 里的**字符串正文**(LLM 经常会把完整的 `Authorization: Bearer xxxx` 回显在 thought/code 里)。当前脱敏对 `str` 类型无操作,风险仍在。

建议在 `_to_jsonable` 的 `str` 分支加一层正则扫描(至少覆盖 `Bearer\s+\S+`、`ghp_[A-Za-z0-9]{20,}`、`sk-[A-Za-z0-9]{20,}`、形如 `eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+` 的 JWT),匹配后替换为 `"***redacted:<type>***"`。

### 2.3 🟡 **P1**: DNS Rebinding 仍可绕过

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/tools/http_tools.py" lines="63-79" />

现在的顺序是:**校验时解析一次 DNS**,然后**httpx 发请求时再独立解析一次**。攻击者控制的权威 DNS 可在两次解析之间返回不同结果(TTL=0 + 第一次返回 public IP,第二次返回 `169.254.169.254`)。这就是经典的 DNS rebinding。

正确修复需要"解析一次,然后把解析到的**安全 IP 锁定写死到连接层**":

```python
# 方案 A: 用自定义 transport / resolver
import httpx

class _PinnedResolver:
    def __init__(self, ip: str): self.ip = ip
    def __call__(self, host, port, family, type_, proto):
        return [(family, type_, proto, "", (self.ip, port))]

# 或方案 B: 改用 URL 的 IP 直连 + Host header
parsed = urlparse(url)
safe_ip = next(iter(resolved_ips))  # 选一个已通过校验的
new_url = url.replace(parsed.hostname, safe_ip, 1)
headers["Host"] = parsed.hostname
# 注意 HTTPS SNI + 证书校验会因此失败,需要 httpx.AsyncClient(verify=...) 定制
```

完整防御比较繁琐,但如果只是想降低风险,**最小补丁**是设置 httpx 的 `transport` 用固定的 `IPResolver`,把第二步 DNS 绑死在第一步的结果上。

如果暂时不处理,也至少在 docstring / README 里明确声明"本 SSRF 防御不抗 DNS rebinding"。

### 2.4 🟡 **P2**: `close_http_clients()` 定义了但无处调用

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/tools/http_tools.py" lines="93-99" />

`grep -r close_http_clients` 结果只有定义行本身,**没有任何调用**。这意味着:
- 在单次 CLI 运行的场景不影响(进程退出一起收);
- 但在库/Jupyter/Pytest/多次 `asyncio.run()` 的场景下,`_clients_by_loop` 会继续缓存已关闭 loop 的 client,虽然 `client.is_closed` 分支会触发重建,但**旧的 client 永远留在 dict 里**,长期跑就是小内存泄漏。

建议:
1. 在 `runner.run()` 的 `finally` 里 `await close_http_clients()` (或至少关当前 loop 的那个);
2. `_get_client` 里检测到 `loop` 对应旧 client `is_closed` 时,从 dict 中 `pop` 掉再新建。

### 2.5 🟢 `trace()` step 继承语义微变

<ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/observability.py" lines="144-151" />

变更后 `step=existing_ctx.step or name`:
- 外层已绑定 step(如 `"step=3"`)时,内层 span 保留 `"3"`——日志更干净;
- 但 `get_log_context().step` 原本在 span 内返回 span 名 (`"llm_generation"`),现在返回 `"3"`。

需要确认 `LLMClient._complete_internal` 里 `current_step = get_log_context().step` 的用途(L191)—— 如果是拿它当"步骤编号"写入 span metadata,新行为更正确;如果期望它是"当前 span 名",那这次改动变了语义。建议同时加单测验证预期。

---

## 3. 仍未处理的上次 P0/P1 问题

以下是**上次 review 的 P0/P1 清单**中本次未动的项目,按优先级排列:

| P | 问题 | 位置 |
|---|---|---|
| **P0** | `ClearableList.append/extend/__setitem__` 静默丢数据 | <ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/namespace.py" lines="161-184" /> |
| **P0** | 每步 `deepcopy(namespace.session)` 随 `all_items` 线性膨胀 | <ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/executor.py" lines="137-142" /> |
| **P0** | 沙箱仅黑名单 + AST(`__import__("os")` 等可绕过),README 未声明非真沙箱 | <ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/executor.py" lines="24-58" /> |
| **P1** | `parser.py` JS 命名块用手写转义拼字符串,应该 `json.dumps` | <ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/parser.py" lines="54-60" /> |
| **P1** | `final_answer` 通过后本步仍继续走完 record_step / reflection | <ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/runner.py" lines="567-644" /> |
| **P1** | `healing_records` 成功 case 留的是 `status="candidate"`,外部无法区分 | <ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/agent/healer.py" lines="145-187" /> |
| **P2** | `memory_db_path` 默认值未跟随 `scrape_dir` | <ref_snippet file="/home/ubuntu/repos/hawker/hawker_agent/config.py" lines="63-64" /> |
| **P2** | Evaluator / Healer 没有连续失败熔断 | evaluator.py / healer.py |
| **P2** | MemoryStore 没有 TTL / 上限 / prune() | memory/store.py |
| **P2** | runner.py 单文件 684 行职责过多,建议 `AgentSession` 抽象 | runner.py |

这些都不是"必须这次修",但如果目标是"准备 v1 发布",建议至少把 **P0 三项 + P1 final_answer / parser.py** 处理完。

---

## 4. 建议的本次补丁 (最小集)

按"改动量/价值"排序,建议优先在本 PR 或跟进 PR 中处理:

1. **必改**: §2.1 脱敏改为精确匹配或加白名单(否则线上 llm_io.json token 数据全丢)。
2. **强烈建议**: §2.2 value 层 Bearer/JWT 正则扫描(补 1 段正则即可)。
3. **建议**: §2.4 在 `runner.run()` 的 `finally` 里 `await close_http_clients()`。
4. **建议**: README / `_validate_url` docstring 里加一行"本防御不抗 DNS rebinding",直到 §2.3 被真正实现。
5. **建议(跟进 PR)**: 处理本次未动的 P0 —— 尤其是 `ClearableList.append` 静默丢数据,这个在真实运行中是**可观测的 bug**,模型一旦写 `all_items.append(x)` 会以为成功,实际上 result.json 里那条数据不存在。

---

## 5. 结论

本次 commit 方向和质量**都很好**,精准命中了我上次标 P0 的两条(httpx loop、SSRF DNS)和 P0/P1 各一条(凭证脱敏、`_to_jsonable` fallback)。

主要遗憾是**脱敏子串匹配会误伤 `input_tokens` / `output_tokens` / `total_tokens` 等关键运营数据**,强烈建议合并前修复并加负向回归测试。

DNS rebinding 和 close_http_clients 未调用是中等问题,可合并后跟进;但 README 应先声明限制。

上次 review 的 P0 三项(ClearableList、deepcopy 开销、沙箱安全声明)不在本次 commit 范围内,建议排入下一批。

有任何需要我直接给出 patch(例如针对 §2.1 的最小修复)或者想让我深入某一项的,请直接告诉我。
