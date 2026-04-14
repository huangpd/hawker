import logging
import re
import hashlib
from typing import Any, List, Dict

from browser_use.browser.session import BrowserSession
from browser_use.dom.views import DOMInteractedElement

logger = logging.getLogger(__name__)

_INTERACTIVE_RE = re.compile(r"\[i_(\d+)\](.+)")
_REGION_TAG_RE = re.compile(r"<(main|nav|form|dialog|table|ul|ol|section|article)\b", re.IGNORECASE)

def is_dynamic_value(value: str) -> bool:
    """
    判断属性值是否像动态生成的（如随机哈希、包含过多数字等）。
    """
    if not value:
        return True
    if re.search(r'\d{5,}', value):
        return True
    if len(value) < 4 and any(c.isdigit() for c in value):
        return True
    return False

def generate_css_selector_for_node(node: DOMInteractedElement) -> str:
    """
    为单个节点生成最严谨的 CSS 选择器。
    """
    # 1. 尝试语义化属性 (data-testid, etc.)
    semantic_attrs = ['data-testid', 'data-qa', 'aria-label', 'name', 'placeholder']
    for attr in semantic_attrs:
        val = node.attributes.get(attr)
        if val and not is_dynamic_value(val):
            return f'{node.tag_name}[{attr}="{val}"]'

    # 2. 尝试唯一 ID (非动态)
    element_id = node.attributes.get('id')
    if element_id and not is_dynamic_value(element_id):
        return f"#{CSS_escape(element_id)}"

    # 3. 尝试类名 (过滤掉动态类)
    classes = node.attributes.get('class', '').split()
    stable_classes = [c for c in classes if not is_dynamic_value(c)]
    if stable_classes:
        class_selector = "".join([f".{CSS_escape(c)}" for c in stable_classes[:2]])
        return f"{node.tag_name}{class_selector}"

    return node.tag_name

def CSS_escape(value: str) -> str:
    """简单的 CSS 转义。"""
    if not value:
        return ""
    return re.sub(r'([!"#$%&\'()*+,./:;<=>?@\[\\\]^`{|}~])', r'\\\1', value)

async def get_selector_from_index(browser_session: BrowserSession, index: int) -> Dict[str, Any]:
    """
    严谨的选择器提取引擎。
    """
    node = await browser_session.get_element_by_index(index)
    if not node:
        raise ValueError(f"Element with index {index} not found in current snapshot.")

    # 现在的 DOMInteractedElement 在不同版本结构可能不同，我们采用最稳健的属性访问
    target_selector = generate_css_selector_for_node(node)
    
    # 简单的路径回溯
    shadow_path = []
    in_iframe = False
    
    # 构建描述性路径
    path_str = target_selector
    
    result = {
        "selector": target_selector,
        "shadow_path": shadow_path,
        "in_iframe": in_iframe,
        "full_path": path_str
    }
    
    logger.debug(f"Generated selector for index {index}: {result}")
    return result


def _normalize_preview(text: str, limit: int = 80) -> str:
    """
    将文本压缩为单行预览。

    参数:
        text (str): 原始文本。
        limit (int): 预览最大长度。

    返回:
        str: 压缩后的单行文本。
    """
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "..."


def _extract_interactives(dom_repr: str, limit: int = 8) -> list[str]:
    """
    从 DOM 表示中提取可交互元素预览。

    参数:
        dom_repr (str): DOM 文本表示。
        limit (int): 最多提取的元素数量。

    返回:
        list[str]: 可交互元素预览列表。
    """
    interactives: list[str] = []
    for line in dom_repr.splitlines():
        match = _INTERACTIVE_RE.search(line.strip())
        if not match:
            continue
        interactives.append(f"[i_{match.group(1)}] {_normalize_preview(match.group(2))}")
        if len(interactives) >= limit:
            break
    return interactives


def _extract_region_tags(dom_repr: str, limit: int = 6) -> list[str]:
    """
    从 DOM 表示中提取主要区域标签。

    参数:
        dom_repr (str): DOM 文本表示。
        limit (int): 最多提取的区域数量。

    返回:
        list[str]: 区域标签列表。
    """
    tags: list[str] = []
    for match in _REGION_TAG_RE.finditer(dom_repr):
        tag = match.group(1).lower()
        if tag not in tags:
            tags.append(tag)
        if len(tags) >= limit:
            break
    return tags


def build_dom_snapshot(
    *,
    title: str,
    url: str,
    dom_repr: str,
    pages_above: float = 0.0,
    pages_below: float = 0.0,
    pending_requests: int = 0,
    tabs: int = 1,
) -> dict[str, Any]:
    """
    从 DOM 文本表示中构建轻量语义快照。

    参数:
        title (str): 页面标题。
        url (str): 页面 URL。
        dom_repr (str): DOM 文本表示。
        pages_above (float): 当前视口上方页数。
        pages_below (float): 当前视口下方页数。
        pending_requests (int): 当前待处理网络请求数。
        tabs (int): 当前标签页数量。

    返回:
        dict[str, Any]: 用于 DOM 摘要和 diff 的页面快照。
    """
    interactives = _extract_interactives(dom_repr, limit=8)
    regions = _extract_region_tags(dom_repr, limit=6)
    fingerprint_source = "\n".join(
        [
            url,
            title,
            str(len(interactives)),
            "|".join(interactives[:5]),
            "|".join(regions),
        ]
    )
    return {
        "title": title,
        "url": url,
        "interactive_count": dom_repr.count("[i_"),
        "interactive_preview": interactives,
        "regions": regions,
        "pages_above": round(pages_above, 1),
        "pages_below": round(pages_below, 1),
        "pending_requests": pending_requests,
        "tabs": tabs,
        "fingerprint": hashlib.sha1(fingerprint_source.encode("utf-8")).hexdigest()[:12],
    }


def render_dom_summary(snapshot: dict[str, Any]) -> str:
    """
    将页面快照渲染为轻量摘要文本。

    参数:
        snapshot (dict[str, Any]): 页面快照。

    返回:
        str: 适合注入上下文的 DOM 摘要。
    """
    lines = [
        f"[DOM Summary] {snapshot.get('title') or '(无标题)'}",
        f"URL: {snapshot.get('url') or '(未知 URL)'}",
        f"交互元素: {snapshot.get('interactive_count', 0)}",
    ]
    regions = snapshot.get("regions") or []
    if regions:
        lines.append(f"区域: {', '.join(regions)}")
    if snapshot.get("pages_above") or snapshot.get("pages_below"):
        lines.append(
            f"滚动: 上方 {snapshot.get('pages_above', 0)} 页, 下方 {snapshot.get('pages_below', 0)} 页"
        )
    if snapshot.get("pending_requests"):
        lines.append(f"待处理请求: {snapshot['pending_requests']}")
    interactives = snapshot.get("interactive_preview") or []
    if interactives:
        lines.append("交互示例:")
        lines.extend(f"- {item}" for item in interactives[:6])
    return "\n".join(lines)


def render_dom_diff(previous: dict[str, Any] | None, current: dict[str, Any]) -> str:
    """
    计算前后两个页面快照的语义差异。

    参数:
        previous (dict[str, Any] | None): 上一个页面快照。
        current (dict[str, Any]): 当前页面快照。

    返回:
        str: 面向模型的 DOM 增量变化描述。
    """
    if not previous:
        return "[DOM Diff]\n无前序快照，返回当前摘要。\n" + render_dom_summary(current)

    lines = ["[DOM Diff]"]
    prev_url = previous.get("url")
    curr_url = current.get("url")
    if prev_url == curr_url:
        lines.append("- URL 未变化")
    else:
        lines.append(f"- URL 变化: {prev_url} -> {curr_url}")

    prev_count = int(previous.get("interactive_count", 0))
    curr_count = int(current.get("interactive_count", 0))
    if prev_count != curr_count:
        lines.append(f"- 交互元素数: {prev_count} -> {curr_count}")
    else:
        lines.append(f"- 交互元素数不变: {curr_count}")

    prev_regions = set(previous.get("regions") or [])
    curr_regions = set(current.get("regions") or [])
    added_regions = sorted(curr_regions - prev_regions)
    removed_regions = sorted(prev_regions - curr_regions)
    if added_regions:
        lines.append(f"- 新增区域: {', '.join(added_regions)}")
    if removed_regions:
        lines.append(f"- 消失区域: {', '.join(removed_regions)}")

    prev_interactives = set(previous.get("interactive_preview") or [])
    curr_interactives = list(current.get("interactive_preview") or [])
    added_interactives = [item for item in curr_interactives if item not in prev_interactives]
    if added_interactives:
        lines.append(f"- 新增交互示例: {', '.join(added_interactives[:4])}")

    if len(lines) == 3 and previous.get("fingerprint") == current.get("fingerprint"):
        lines.append("- 页面主体结构基本未变")
    elif previous.get("fingerprint") == current.get("fingerprint"):
        lines.append("- 结构指纹未变，主要是局部更新")

    return "\n".join(lines)
