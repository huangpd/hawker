from __future__ import annotations

import pytest

from hawker_agent.memory.store import (
    MemoryEntry,
    MemoryMatch,
    MemoryStore,
    _utcnow_iso,
    build_raw_code_memories,
    extract_site_keys,
    infer_task_intent,
)
from hawker_agent.models.state import CodeAgentState


class TestMemoryHelpers:
    def test_utcnow_iso_uses_beijing_timezone(self) -> None:
        assert _utcnow_iso().endswith("+08:00")

    def test_extract_site_keys(self) -> None:
        task = "打开 https://mcp.aibase.com/zh/explore 并提取列表"
        assert extract_site_keys(task) == ["mcp.aibase.com"]

    def test_infer_task_intent(self) -> None:
        assert infer_task_intent("点击按下载量排序并抓包") == "sort"
        assert infer_task_intent("获取3页列表并提取 title url") == "extract"


class TestMemoryStore:
    def test_upsert_and_search(self, tmp_path) -> None:
        store = MemoryStore(tmp_path / "memory.db")
        entry = MemoryEntry(
            memory_type="site_lesson",
            site_key="mcp.aibase.com",
            task_intent="sort",
            page_kind="list_page",
            summary="不要直接猜排序参数，先点击 UI 再抓包",
            detail="source=Step 8",
            success=True,
            negative=False,
            confidence=0.85,
            source_run_id="run_1",
            source_step=8,
            source_url="https://mcp.aibase.com/zh/explore",
        )
        store.upsert_entries([entry])

        matches = store.search("打开 https://mcp.aibase.com/zh/explore 点击按下载量排序并抓包", limit=3)
        assert len(matches) == 1
        assert matches[0].entry.site_key == "mcp.aibase.com"
        assert "先点击 UI 再抓包" in matches[0].render()

    def test_memory_match_render_marks_recipe(self) -> None:
        entry = MemoryEntry(
            memory_type="extract_recipe",
            site_key="example.com",
            task_intent="extract",
            page_kind="list_page",
            summary="列表提取优先走 js()",
            detail="items = await js('...')",
            success=True,
            negative=False,
            confidence=0.86,
            source_run_id="run_4",
            source_step=2,
            source_url="https://example.com/list",
        )
        rendered = MemoryMatch(entry=entry, score=140.0, reason="site=example.com").render()
        assert rendered.startswith("- 可执行配方:")


class TestRawCodeMemory:
    def test_build_raw_code_memories_filters_low_value_steps(self) -> None:
        state = CodeAgentState()
        state.run_id = "run_5"
        state.last_dom_snapshot = {"url": "https://github.com/trending"}
        state.llm_records.extend(
            [
                {
                    "step": 4,
                    "parsed_output": {
                        "thought": "提取列表",
                        "code": "raw = await js('...')\nobserve(f'提取 {len(raw)} 条')",
                    },
                    "execution": {
                        "progress_made": True,
                        "observation": "提取 12 条",
                    },
                },
                {
                    "step": 5,
                    "parsed_output": {
                        "thought": "保存结果",
                        "code": "await save_checkpoint('checkpoint.json')\nawait final_answer('完成')",
                    },
                    "execution": {
                        "progress_made": True,
                        "observation": "保存成功",
                    },
                },
            ]
        )
        entries = build_raw_code_memories(
            "打开 https://github.com/trending 提取第一页项目 url start fork today_start",
            state,
        )
        assert len(entries) == 1
        assert entries[0].memory_type == "raw_extract_code"
        assert entries[0].source_step == 4

    def test_build_raw_code_memories_keeps_success_code_and_evidence(self) -> None:
        state = CodeAgentState()
        state.run_id = "run_6"
        state.last_dom_snapshot = {"url": "https://github.com/trending"}
        state.llm_records.append(
            {
                "step": 4,
                "parsed_output": {
                    "thought": "直接提取列表页字段",
                    "code": "raw = await js(\"(function(){ var rows=document.querySelectorAll('article.Box-row'); return Array.from(rows).map(r=>({url:r.querySelector('h2 a').href})); })()\")",
                },
                "execution": {
                    "progress_made": True,
                    "observation": "[js] 返回 12 条数据 | 签名: {url: str}",
                },
            }
        )

        entries = build_raw_code_memories(
            "打开 https://github.com/trending 提取第一页项目 url start fork today_start",
            state,
        )
        assert len(entries) == 1
        assert entries[0].summary == "github.com 成功提取代码 (extract)"
        assert "article.Box-row" in entries[0].detail
        assert "上次成功证据" in entries[0].detail

    def test_build_raw_code_memories_backfills_dependency_context(self) -> None:
        state = CodeAgentState()
        state.run_id = "run_7"
        state.last_dom_snapshot = {"url": "https://mcp.aibase.com/zh/explore"}
        state.llm_records.extend(
            [
                {
                    "step": 6,
                    "parsed_output": {
                        "thought": "准备 API 参数",
                        "code": "api_url = 'https://mcpapi.aibase.cn/api/mcp/querypage'\nbase_body = {'pageNo': 1, 'pageSize': 40}",
                    },
                    "execution": {
                        "progress_made": True,
                        "observation": "参数准备完成",
                    },
                },
                {
                    "step": 7,
                    "parsed_output": {
                        "thought": "循环抓取",
                        "code": (
                            "all_items = []\n"
                            "for page in range(1, 3):\n"
                            "    base_body['pageNo'] = page\n"
                            "    resp = await http_json(api_url, method='POST', body=json.dumps(base_body))\n"
                            "    observe(str(resp.get('code')))"
                        ),
                    },
                    "execution": {
                        "progress_made": True,
                        "observation": "[http_request] 200",
                    },
                },
            ]
        )

        entries = build_raw_code_memories(
            "打开 https://mcp.aibase.com/zh/explore 按下载量排序后抓取前两页",
            state,
        )

        assert len(entries) == 1
        detail = entries[0].detail
        assert "依赖前置变量: api_url, base_body" in detail
        assert "依赖变量示例" in detail
        assert "api_url =" in detail
        assert "base_body =" in detail
