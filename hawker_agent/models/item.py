from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ItemStore:
    """
    包装 all_items 列表和去重逻辑。
    替换 _build_namespace 中的 append_items 闭包。
    """

    _items: list[dict] = field(default_factory=list)
    _seen_keys: set[str] = field(default_factory=set)

    def append(self, items: list[dict]) -> tuple[int, int]:
        """追加数据，自动去重。返回 (added_count, skipped_count)。"""
        added, skipped = 0, 0
        for item in items:
            if not isinstance(item, dict):
                skipped += 1
                continue
            key = self._make_key(item)
            if key in self._seen_keys:
                skipped += 1
                continue
            self._seen_keys.add(key)
            self._items.append(item)
            added += 1
        return added, skipped

    def _make_key(self, item: dict) -> str:
        """按优先级选取唯一标识字段，兜底使用 JSON 序列化。"""
        for key in ("id", "url", "link", "href", "path", "download_url", "name", "title"):
            val = item.get(key)
            if val:
                return f"{key}:{val}"
        return json.dumps(item, ensure_ascii=False, sort_keys=True)

    def to_list(self) -> list[dict]:
        """返回 items 的副本。"""
        return list(self._items)

    def get_raw_list(self) -> list[dict]:
        """返回 items 的原始引用，慎用。"""
        return self._items

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        """清空所有数据和去重记录。"""
        self._items.clear()
        self._seen_keys.clear()

    def __bool__(self) -> bool:
        return bool(self._items)
