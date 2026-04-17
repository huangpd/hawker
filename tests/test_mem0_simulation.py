from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from hawker_agent.config import get_settings


LLM_IO_FIXTURE_PATH = Path("/Users/hpd/code/hawker/hawker_file/9dfa4d24d440/llm_io.json")


def _load_llm_io(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task = str(payload.get("task") or "").strip()
    records = payload.get("records") or []
    if not isinstance(records, list):
        raise TypeError("llm_io.json 的 records 字段不是 list")
    return task, records


def _record_to_messages(task: str, record: dict[str, Any]) -> list[dict[str, str]]:
    step = int(record.get("step") or 0)
    parsed = record.get("parsed_output") or {}
    execution = record.get("execution") or {}
    thought = str(parsed.get("thought") or "").strip()
    code = str(parsed.get("code") or "").strip()
    observation = str(execution.get("observation") or "").strip()

    user_content = (
        f"Task:\n{task}\n\n"
        f"Step={step}\n"
        f"Thought:\n{thought or '[empty]'}\n\n"
        f"Code:\n{code or '[empty]'}"
    )
    assistant_content = f"Observation:\n{observation or '[empty]'}"
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def _build_mem0_config(collection_path: Path) -> dict[str, Any]:
    cfg = get_settings()
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": cfg.model_name,
                "api_key": cfg.openai_api_key,
                "openai_base_url": cfg.openai_base_url,
                "temperature": 0.0,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "api_key": cfg.openai_api_key,
                "openai_base_url": cfg.openai_base_url,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": f"mem0_test_{uuid.uuid4().hex[:8]}",
                "path": str(collection_path),
            },
        },
        "enable_graph": False,
    }


def test_mem0_import_add_search_quality(tmp_path: Path) -> None:
    try:
        from mem0 import Memory
    except Exception:
        pytest.skip("mem0 未安装，先执行: uv pip install mem0ai")

    if not LLM_IO_FIXTURE_PATH.exists():
        pytest.skip(f"缺少测试输入文件: {LLM_IO_FIXTURE_PATH}")

    cfg = get_settings()
    if not cfg.openai_api_key or not cfg.model_name:
        pytest.skip("缺少 OPENAI_API_KEY 或 MODEL_NAME")

    task, records = _load_llm_io(LLM_IO_FIXTURE_PATH)
    assert task
    assert records

    memory = Memory.from_config(_build_mem0_config(tmp_path / "mem0_qdrant_local"))
    user_id = f"mem0_user_{uuid.uuid4().hex[:8]}"

    add_events = 0
    for record in records:
        messages = _record_to_messages(task, record)
        response = memory.add(messages, user_id=user_id, metadata={"source": "llm_io", "run": "9dfa4d24d440"})
        assert isinstance(response, dict)
        assert "results" in response
        assert isinstance(response["results"], list)
        add_events += len(response["results"])

    assert add_events > 0, "mem0 未产生任何新增/更新记忆事件"

    query = "mcp aibase 搜索工具 python 按下载量 排序 querypage 分页 提取 名称 简介 url"
    search_response = memory.search(query, user_id=user_id, limit=8)
    assert isinstance(search_response, dict)
    assert "results" in search_response
    results = search_response["results"]
    assert isinstance(results, list)
    assert results, "mem0 检索结果为空"

    for item in results[:5]:
        assert "memory" in item and str(item["memory"]).strip()
        assert "score" in item
        assert "id" in item

    joined = " ".join(str(item.get("memory", "")) for item in results[:5]).lower()
    keyword_hits = sum(kw in joined for kw in ("搜索工具", "python", "下载量", "querypage", "mcp"))
    assert keyword_hits >= 2, f"检索内容质量偏弱，关键词命中不足: {joined[:300]}"
