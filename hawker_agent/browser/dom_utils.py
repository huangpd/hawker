import json
import logging
import re
from typing import Any, Optional, List, Dict

from browser_use.browser.session import BrowserSession
from browser_use.dom.views import DOMInteractedElement

logger = logging.getLogger(__name__)

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
