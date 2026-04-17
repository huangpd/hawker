from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ItemStore:
    """数据存储与去重管理器。

    用于管理代理在执行过程中采集到的所有数据项，并根据预定义的标识符字段进行自动去重。

    Attributes:
        _items (list[dict]): 存储的所有数据项列表。
        _seen_keys (set[str]): 存储已发现数据项的唯一标识符，用于快速去重。
    """

    _items: list[dict] = field(default_factory=list)
    _seen_keys: set[str] = field(default_factory=set)

    def append(self, items: list[dict]) -> tuple[int, int]:
        """向存储中追加数据项，并自动执行去重。

        Args:
            items (list[dict]): 待追加的数据项列表。

        Returns:
            tuple[int, int]: 返回一个元组，包含 (成功添加的数量, 被跳过/去重的数量)。
        """
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
        """为数据项生成唯一标识键。

        按优先级尝试使用常见的唯一标识字段（如 id, url 等），若无此类字段则使用 JSON 序列化作为兜底方案。

        Args:
            item (dict): 数据项字典。

        Returns:
            str: 生成的唯一标识字符串。
        """
        for key in ("id", "url", "link", "href", "path", "download_url", "name", "title"):
            val = item.get(key)
            if val:
                return f"{key}:{val}"
        return json.dumps(item, ensure_ascii=False, sort_keys=True)

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
        self._seen_keys.clear()

    def __bool__(self) -> bool:
        """判断存储是否为空。

        Returns:
            bool: 若包含数据项则返回 True，否则返回 False。
        """
        return bool(self._items)
