from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


_EXPLICIT_ID_FIELDS = ("entity_key", "ref", "id", "key", "uid")
_STABLE_MERGE_FIELDS = set(_EXPLICIT_ID_FIELDS)

_PLACEHOLDER_STRINGS = {
    "",
    "unknown",
    "none",
    "null",
    "n/a",
    "na",
    "tbd",
    "pending",
    "missing",
    "missing_on_disk",
    "not_found",
    "unavailable",
}


@dataclass
class ItemStore:
    """数据存储与去重管理器。

    用于管理代理在执行过程中采集到的所有数据项，并根据预定义的标识符字段进行自动去重。

    Attributes:
        _items (list[dict]): 存储的所有数据项列表。
        _index_by_key (dict[str, int]): 当前实体键到列表索引的映射。
        _last_changed_index (int | None): 最近一次发生新增或合并的实体索引。
    """

    _items: list[dict] = field(default_factory=list)
    _index_by_key: dict[str, int] = field(default_factory=dict)
    _last_changed_index: int | None = None

    def append(self, items: list[dict]) -> tuple[int, int]:
        """向存储中写入数据项，并自动执行 upsert/merge。

        Args:
            items (list[dict]): 待追加的数据项列表。

        Returns:
            tuple[int, int]: 返回一个元组，包含 (发生新增/更新的数量, 无变化/跳过的数量)。
        """
        changed, unchanged = 0, 0
        for item in items:
            if not isinstance(item, dict):
                unchanged += 1
                continue
            index = self._find_existing_index(item)
            if index is not None:
                merged = self._merge_records(self._items[index], item)
                if merged == self._items[index]:
                    unchanged += 1
                    continue
                self._items[index] = merged
                self._index_item(index, merged)
                self._last_changed_index = index
                changed += 1
                continue
            self._items.append(item)
            self._index_item(len(self._items) - 1, item)
            self._last_changed_index = len(self._items) - 1
            changed += 1
        return changed, unchanged

    @classmethod
    def _make_key(cls, item: dict) -> str:
        """为数据项生成稳定的实体标识键。

        按优先级尝试使用显式 identity 字段；若缺失，则使用稳定序列化作为兜底。

        Args:
            item (dict): 数据项字典。

        Returns:
            str: 生成的唯一标识字符串。
        """
        entity_key = cls._canonical_entity_key(item)
        if entity_key:
            return entity_key
        return json.dumps(cls._stable_serialize(item), ensure_ascii=False, sort_keys=True)

    @classmethod
    def _make_aliases(cls, item: dict) -> set[str]:
        aliases: set[str] = set()
        entity_key = cls._canonical_entity_key(item)
        if entity_key:
            aliases.add(entity_key)
        if aliases:
            return aliases

        return {cls._make_key(item)}

    def _find_existing_index(self, item: dict) -> int | None:
        for alias in self._make_aliases(item):
            if alias in self._index_by_key:
                return self._index_by_key[alias]
        return None

    def _index_item(self, index: int, item: dict) -> None:
        for alias in self._make_aliases(item):
            self._index_by_key[alias] = index

    @classmethod
    def _canonical_entity_key(cls, item: dict[str, Any]) -> str | None:
        for key in _EXPLICIT_ID_FIELDS:
            val = item.get(key)
            if not cls._is_informative(val):
                continue
            normalized = cls._normalize_identity_value(val)
            return normalized if key == "entity_key" else f"{key}:{normalized}"
        return None

    @staticmethod
    def _normalize_identity_value(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        return str(value)

    @classmethod
    def _is_informative(cls, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() not in _PLACEHOLDER_STRINGS
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) > 0
        return True

    @classmethod
    def _stable_serialize(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): cls._stable_serialize(v) for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, list):
            return [cls._stable_serialize(v) for v in value]
        if isinstance(value, tuple):
            return [cls._stable_serialize(v) for v in value]
        if isinstance(value, set):
            return sorted(cls._stable_serialize(v) for v in value)
        return value

    @classmethod
    def _merge_records(cls, current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for key in current.keys() | incoming.keys():
            if key not in incoming:
                merged[key] = current[key]
                continue
            if key not in current:
                merged[key] = incoming[key]
                continue
            merged[key] = cls._merge_values(current[key], incoming[key], key)
        return merged

    @classmethod
    def _merge_values(cls, current: Any, incoming: Any, key: str | None = None) -> Any:
        if isinstance(current, dict) and isinstance(incoming, dict):
            return cls._merge_records(current, incoming)
        if isinstance(current, list) and isinstance(incoming, list):
            return cls._merge_lists(current, incoming)
        if not cls._is_informative(incoming):
            return current
        if not cls._is_informative(current):
            return incoming
        if key in _STABLE_MERGE_FIELDS:
            return current
        return incoming

    @classmethod
    def _merge_lists(cls, current: list[Any], incoming: list[Any]) -> list[Any]:
        if not incoming:
            return current
        if not current:
            return incoming
        if all(isinstance(item, dict) for item in current + incoming):
            merged: list[dict[str, Any]] = [dict(item) for item in current]
            index_by_key = {cls._make_key(item): idx for idx, item in enumerate(merged)}
            for item in incoming:
                key = cls._make_key(item)
                if key in index_by_key:
                    idx = index_by_key[key]
                    merged[idx] = cls._merge_records(merged[idx], item)
                else:
                    index_by_key[key] = len(merged)
                    merged.append(item)
            return merged

        merged = list(current)
        seen = {
            json.dumps(cls._stable_serialize(item), ensure_ascii=False, sort_keys=True)
            for item in current
        }
        for item in incoming:
            marker = json.dumps(cls._stable_serialize(item), ensure_ascii=False, sort_keys=True)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(item)
        return merged

    def to_list(self) -> list[dict]:
        """获取所有已存储数据项的副本。

        Returns:
            list[dict]: 包含所有数据项的列表副本。
        """
        return list(self._items)

    def get_raw_list(self) -> list[dict]:
        """获取内部数据项列表的原始引用。

        警告:
            对返回列表的修改将直接影响存储内部状态。

        Returns:
            list[dict]: 内部存储的原始列表引用。
        """
        return self._items

    def __len__(self) -> int:
        """返回已存储的数据项总数。

        Returns:
            int: 数据项数量。
        """
        return len(self._items)

    def clear(self) -> None:
        """清空所有存储的数据项及去重记录。"""
        self._items.clear()
        self._index_by_key.clear()
        self._last_changed_index = None

    def get_last_changed(self) -> dict | None:
        """返回最近一次新增或合并后的实体快照。"""
        if self._last_changed_index is None:
            return None
        if self._last_changed_index >= len(self._items):
            return None
        return dict(self._items[self._last_changed_index])

    def __bool__(self) -> bool:
        """判断存储是否为空。

        Returns:
            bool: 若包含数据项则返回 True，否则返回 False。
        """
        return bool(self._items)
