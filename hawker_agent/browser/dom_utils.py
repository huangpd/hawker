import logging
import re
import hashlib
from typing import Any, Dict

from browser_use.browser.session import BrowserSession
from browser_use.dom.views import DOMInteractedElement

logger = logging.getLogger(__name__)

_INTERACTIVE_RE = re.compile(r"\[i_(\d+)\](.+)")
_REGION_TAG_RE = re.compile(r"<(main|nav|form|dialog|table|ul|ol|section|article)\b", re.IGNORECASE)

def is_dynamic_value(value: str) -> bool:
    """判断属性值是否为动态生成的。

    例如包含随机哈希、过长数字序列等。

    Args:
        value (str): 要判断的属性值。

    Returns:
        bool: 如果看起来是动态生成的则返回 True，否则返回 False。
    """
    if not value:
        return True
    if re.search(r'\d{5,}', value):
        return True
    if len(value) < 4 and any(c.isdigit() for c in value):
        return True
    return False

def generate_css_selector_for_node(node: DOMInteractedElement) -> str:
    """为指定节点生成严谨且尽可能唯一的 CSS 选择器。

    依次尝试语义化属性、唯一 ID 和稳定类名。

    Args:
        node (DOMInteractedElement): 交互元素节点对象。

    Returns:
        str: 生成的 CSS 选择器字符串。
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
    """对字符串进行 CSS 转义。

    Args:
        value (str): 要转义的字符串。

    Returns:
        str: 转义后的字符串。
    """
    if not value:
        return ""
    return re.sub(r'([!"#$%&\'()*+,./:;<=>?@\[\\\]^`{|}~])', r'\\\1', value)

async def get_selector_from_index(browser_session: BrowserSession, index: int) -> Dict[str, Any]:
    """根据 DOM 索引获取元素的严谨选择器。

    Args:
        browser_session (BrowserSession): 浏览器会话对象。
        index (int): 元素的 DOM 索引。

    Returns:
        Dict[str, Any]: 包含选择器、Shadow Path、是否在 iframe 中以及完整路径的字典。

    Raises:
        ValueError: 如果在当前快照中找不到指定索引的元素。
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
    """将文本规范化并压缩为单行预览。

    Args:
        text (str): 原始文本。
        limit (int, optional): 预览最大长度。默认为 80。

    Returns:
        str: 规范化后的单行预览文本。
    """
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "..."


def _extract_interactives(dom_repr: str, limit: int = 8) -> list[str]:
    """从 DOM 文本表示中提取可交互元素的预览。

    Args:
        dom_repr (str): DOM 的文本表示字符串。
        limit (int, optional): 最多提取的元素数量。默认为 8。

    Returns:
        list[str]: 可交互元素的预览列表。
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
    """从 DOM 文本表示中提取主要区域的 HTML 标签。

    Args:
        dom_repr (str): DOM 的文本表示字符串。
        limit (int, optional): 最多提取的区域标签数量。默认为 6。

    Returns:
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
    """从 DOM 文本表示中构建轻量级的语义快照。

    Args:
        title (str): 页面标题。
        url (str): 页面 URL。
        dom_repr (str): DOM 的文本表示字符串。
        pages_above (float, optional): 当前视口上方的页数。默认为 0.0。
        pages_below (float, optional): 当前视口下方的页数。默认为 0.0。
        pending_requests (int, optional): 当前待处理的网络请求数。默认为 0。
        tabs (int, optional): 当前打开的标签页数量。默认为 1。

    Returns:
        dict[str, Any]: 包含页面快照信息的字典，用于生成摘要和计算差异。
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
    """将页面快照渲染为适合 LLM 阅读的轻量级摘要文本。

    Args:
        snapshot (dict[str, Any]): 页面快照字典。

    Returns:
        str: 渲染后的 DOM 摘要文本。
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
    """计算并渲染前后两个页面快照之间的语义差异。

    Args:
        previous (dict[str, Any] | None): 上一个页面的快照字典。
        current (dict[str, Any]): 当前页面的快照字典。

    Returns:
        str: 描述页面增量变化的文本。
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
