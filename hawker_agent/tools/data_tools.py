from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hawker_agent.tools.registry import ToolRegistry


def get_type_signature(d: dict, max_keys: int = 10) -> str:
    """生成 dict 的类型签名字符串，用于 http_json 摘要。"""
    parts = []
    for k, v in list(d.items())[:max_keys]:
        if isinstance(v, list):
            inner = type(v[0]).__name__ if v else "Any"
            parts.append(f"{k}: list[{inner}]")
        elif isinstance(v, dict):
            sub_keys = ",".join(list(v.keys())[:5])
            parts.append(f"{k}: dict{{{sub_keys}}}")
        else:
            parts.append(f"{k}: {type(v).__name__}")
    return "{" + ", ".join(parts) + "}"


def parse_http_response(raw: str) -> tuple[int, str]:
    """解析 http_request() 返回值为 (status_code, body)。"""
    raw = raw.strip()
    if raw.startswith("[错误]"):
        raise RuntimeError(raw)
    match = re.match(r"^\[(\d{3})\]\s*\n?", raw)
    if not match:
        raise ValueError(f"无法解析 http_request 返回值: {raw[:120]}")
    return int(match.group(1)), raw[match.end() :]


def clean_items(items: list) -> list[dict]:
    """过滤非 dict 和 _truncated 标记元素。"""
    if not isinstance(items, list):
        raise TypeError(f"clean_items() 需要 list，收到 {type(items).__name__}")
    return [item for item in items if isinstance(item, dict) and not item.get("_truncated")]


def ensure(condition: object, message: str) -> None:
    """断言，不满足时抛 RuntimeError。"""
    if not condition:
        raise RuntimeError(message)


def normalize_items(items: object) -> list[dict]:
    """
    将 str/dict/list 统一转为 list[dict]，并调用 clean_items 过滤。
    迁移自 main.py _normalize_items。
    """
    if isinstance(items, str):
        items = json.loads(items)
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        raise TypeError(f"items 必须是 list/dict/JSON 字符串，收到 {type(items).__name__}")
    def _trim(value: object) -> object:
        if isinstance(value, list):
            return [_trim(item) for item in value]
        if not isinstance(value, dict):
            return value
        value = {key: _trim(subvalue) for key, subvalue in value.items()}
        if {"url", "filename", "path", "size"}.issubset(value.keys()):
            value.pop("ok", None)
            value.pop("requested_filename", None)
            value.pop("method", None)
        return value
    return clean_items([_normalize_entity_identity(_trim(item)) for item in items])

_EXPLICIT_KEY_FIELDS = ("entity_key", "ref", "id", "key", "uid")


def _normalize_entity_identity(item: object) -> object:
    if not isinstance(item, dict):
        return item
    row = dict(item)
    _normalize_download_shape(row)

    entity_key = _pick_existing_entity_key(row)
    if entity_key:
        row["entity_key"] = entity_key

    return row


def _normalize_download_shape(item: dict[str, Any]) -> None:
    """Fold legacy download fields into the nested `download` structure."""
    download = item.get("download")
    if not isinstance(download, dict):
        download = {}
        item["download"] = download

    downloaded_file = item.pop("downloaded_file", None)
    if isinstance(downloaded_file, str) and downloaded_file.strip() and not download.get("file"):
        download["file"] = downloaded_file.strip()

    download_status = item.pop("download_status", None)
    if isinstance(download_status, str) and download_status.strip():
        if download_status.strip() in {"success", "missing_on_disk", "unknown"}:
            download.setdefault("status", download_status.strip())

    size = item.pop("size", None)
    if isinstance(size, int) and size > 0 and not download.get("size"):
        download["size"] = size

    if not download:
        item.pop("download", None)


def _pick_existing_entity_key(item: dict[str, Any]) -> str | None:
    for key in _EXPLICIT_KEY_FIELDS:
        value = item.get(key)
        if _is_informative_identity(value):
            return f"{key}:{str(value).strip()}" if key != "entity_key" else str(value).strip()
    return None


def _is_informative_identity(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def summarize_json(data: object) -> str:
    """生成 http_json 返回值的摘要字符串。"""
    if isinstance(data, list):
        if not data:
            return "[http_json] 返回空列表 []"
        sample = data[0]
        sample_str = json.dumps(sample, ensure_ascii=False)
        if len(sample_str) > 150:
            sample_str = sample_str[:150] + "..."
        sig = f" | 签名: {get_type_signature(sample)}" if isinstance(sample, dict) else ""
        return f"[http_json] {len(data)} 条{sig} | 样本: {sample_str}"
    if isinstance(data, dict):
        return f"[http_json] dict | {get_type_signature(data)}"
    return f"[http_json] {type(data).__name__}"


def analyze_json_structure(
    data: object,
    *,
    max_depth: int = 4,
    max_list_items: int = 3,
    max_keys: int = 40,
    max_paths: int = 200,
    sample_chars: int = 120,
) -> dict[str, Any]:
    """分析任意 JSON-like 对象的结构，供模型决定字段路径。

    只返回结构、路径、key 和少量标量样本，不返回完整正文；适合先看大 JSON
    的真实 shape，再决定用哪个路径提取数据。
    """
    paths: list[dict[str, Any]] = []

    def sample_scalar(value: object) -> object:
        if isinstance(value, str):
            return value[:sample_chars] + ("..." if len(value) > sample_chars else "")
        if isinstance(value, int | float | bool) or value is None:
            return value
        return type(value).__name__

    def add_path(entry: dict[str, Any]) -> None:
        if len(paths) < max_paths:
            paths.append(entry)

    def walk(value: object, path: str, depth: int) -> None:
        if depth > max_depth or len(paths) >= max_paths:
            return
        if isinstance(value, dict):
            keys = list(value.keys())
            add_path(
                {
                    "path": path,
                    "type": "dict",
                    "keys": keys[:max_keys],
                    "key_count": len(keys),
                    "truncated_keys": len(keys) > max_keys,
                }
            )
            for key in keys[:max_keys]:
                walk(value[key], f"{path}.{key}" if path != "$" else f"$.{key}", depth + 1)
            return
        if isinstance(value, list):
            item_types = sorted({type(item).__name__ for item in value[:max_list_items]})
            item_keys: set[str] = set()
            for item in value[:max_list_items]:
                if isinstance(item, dict):
                    item_keys.update(item.keys())
            add_path(
                {
                    "path": path,
                    "type": "list",
                    "length": len(value),
                    "sample_item_types": item_types,
                    "sample_item_keys": sorted(item_keys)[:max_keys],
                    "truncated_item_keys": len(item_keys) > max_keys,
                }
            )
            for item in value[:max_list_items]:
                walk(item, f"{path}[]", depth + 1)
            return
        add_path({"path": path, "type": type(value).__name__, "sample": sample_scalar(value)})

    walk(data, "$", 0)
    if isinstance(data, dict):
        root_type = "dict"
        root_keys = list(data.keys())[:max_keys]
    elif isinstance(data, list):
        root_type = "list"
        root_keys = []
    else:
        root_type = type(data).__name__
        root_keys = []
    return {
        "root_type": root_type,
        "root_keys": root_keys,
        "path_count": len(paths),
        "truncated_paths": len(paths) >= max_paths,
        "paths": paths,
    }


def _safe_join(run_dir: str, filename: str) -> str:
    """Safely join run_dir and filename, preventing path traversal."""
    # Strip any directory components — only keep the basename
    safe_name = os.path.basename(filename)
    if not safe_name:
        safe_name = "untitled"
    resolved = os.path.normpath(os.path.join(run_dir, safe_name))
    # Final guard: ensure the resolved path is inside run_dir
    if not resolved.startswith(os.path.normpath(run_dir) + os.sep) and resolved != os.path.normpath(run_dir):
        raise ValueError(f"Path traversal blocked: {filename!r} resolves outside run_dir")
    return resolved


def save_file(data: str, filename: str, run_dir: str) -> str:
    """保存数据到 run_dir 下的文件。"""
    filepath = _safe_join(run_dir, filename)
    try:
        parsed = json.loads(data)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        count = len(parsed) if isinstance(parsed, list) else 1
        return f"[OK] 已保存 {count} 条记录到 {filepath}"
    except json.JSONDecodeError:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(data)
        return f"[OK] 已保存文本到 {filepath}"


def check_files_on_disk(run_dir: str | os.PathLike, items: list[dict]) -> dict[str, Any]:
    """核对 items 中声明的下载文件在磁盘或 OBS 上的真实状态。"""
    base = Path(run_dir)
    total = 0
    verified = 0
    obs_verified = 0
    missing = []
    zero_byte = []
    
    for item in items:
        for file_ref in _iter_item_file_refs(item):
            if file_ref.get("obs_key"):
                total += 1
                obs_verified += 1
                continue

            fpath = file_ref.get("file") or file_ref.get("filename") or file_ref.get("path")
            if not fpath or not isinstance(fpath, str):
                continue
            total += 1
            p = Path(fpath)
            if not p.is_absolute():
                p = base / fpath
            if not p.exists() or not p.is_file():
                missing.append(fpath)
            elif p.stat().st_size == 0:
                zero_byte.append(fpath)
            else:
                verified += 1
            
    return {
        "total_downloads": total,
        "verified_count": verified,
        "obs_verified_count": obs_verified,
        "missing_files": missing,
        "empty_files": zero_byte,
        "summary": f"Total: {total}, Verified: {verified}, OBS: {obs_verified}, Missing: {len(missing)}, Empty: {len(zero_byte)}"
    }


def _iter_item_file_refs(item: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    download = item.get("download")
    if isinstance(download, dict):
        refs.append(download)
    artifacts = item.get("artifacts")
    if isinstance(artifacts, dict):
        file_artifact = artifacts.get("file")
        if isinstance(file_artifact, dict):
            refs.append(file_artifact)
    return refs


def register_data_tools(registry: ToolRegistry) -> None:
    """将数据处理辅助工具注册到工具注册表。只注册大模型需要手动调用的工具。"""
    registry.register(ensure, category="同步工具", expose_in_prompt=False)
    registry.register(parse_http_response, category="同步工具", expose_in_prompt=False)
    registry.register(analyze_json_structure, category="同步工具")
