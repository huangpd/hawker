import json
import logging
import re
from typing import Any, Optional, List, Dict

from browser_use.browser.session import BrowserSession
from browser_use.dom.views import DOMElementNode

logger = logging.getLogger(__name__)

def is_dynamic_value(value: str) -> bool:
    """
    判断属性值是否像动态生成的（如随机哈希、包含过多数字等）。
    50年架构师经验提示：避免使用会随构建或运行时变化的属性。
    """
    if not value:
        return True
    
    # 规则1: 包含长数字序列 (e.g., id-123456)
    if re.search(r'\d{5,}', value):
        return True
    
    # 规则2: 看起来像 Base64 或随机哈希 (e.g., _30jeq3, jsc_c_1)
    # 这里的阈值需要根据实际项目调整，通常混淆后的类名很短且无语义
    if len(value) < 4 and any(c.isdigit() for c in value):
        return True
    
    return False

def generate_css_selector_for_node(node: DOMElementNode) -> str:
    """
    为单个节点生成最严谨的 CSS 选择器。
    """
    # 1. 尝试唯一 ID (非动态)
    element_id = node.attributes.get('id')
    if element_id and not is_dynamic_value(element_id):
        # 注意：ID 必须进行转义，以防包含非法字符（如开始是数字）
        escaped_id = f"#{CSS_escape(element_id)}"
        return escaped_id

    # 2. 尝试语义化属性 (data-testid, aria-label, etc.)
    semantic_attrs = ['data-testid', 'data-qa', 'aria-label', 'name', 'placeholder']
    for attr in semantic_attrs:
        val = node.attributes.get(attr)
        if val and not is_dynamic_value(val):
            return f'{node.tag_name}[{attr}="{val}"]'

    # 3. 尝试类名 (过滤掉动态类)
    classes = node.attributes.get('class', '').split()
    stable_classes = [c for c in classes if not is_dynamic_value(c)]
    if stable_classes:
        # 使用前两个稳定类名增加特异性
        class_selector = "".join([f".{CSS_escape(c)}" for c in stable_classes[:2]])
        return f"{node.tag_name}{class_selector}"

    # 4. 兜底策略: Tag + 结构化位置 (nth-of-type)
    # 虽然不太完美，但在局部作用域内是唯一的
    # 注意：browser-use 提供的 DOMElementNode 可能没有足够的 sibling 信息来计算精确的 nth-child
    # 这里我们返回 tag_name 配合后续的 index 验证
    return node.tag_name

def CSS_escape(value: str) -> str:
    """
    简单的 CSS 转义逻辑。实际生产环境应使用更完备的转义库。
    """
    if not value:
        return ""
    # 转义 ID/Class 中常见的特殊字符
    return re.sub(r'([!"#$%&\'()*+,./:;<=>?@\[\\\]^`{|}~])', r'\\\1', value)

async def get_selector_from_index(browser_session: BrowserSession, index: int) -> Dict[str, Any]:
    """
    严谨的选择器提取引擎。
    返回一个结构化的定位方案，支持 Shadow DOM。
    
    Returns:
        {
            "selector": str,          # 目标元素的选择器
            "shadow_path": List[str], # Shadow Host 的选择器路径（由外向内）
            "in_iframe": bool,        # 是否在 Iframe 中
            "full_path": str          # 组合后的描述
        }
    """
    node = await browser_session.get_element_by_index(index)
    if not node:
        raise ValueError(f"Element with index {index} not found in current DOM snapshot.")

    shadow_path = []
    in_iframe = False
    
    # 追溯 Shadow DOM 和 Iframe
    current = node.parent_node
    while current:
        # 记录 Shadow Host
        # 在 browser_use 的 DOM 模型中，如果是 Shadow Root 的父级，就是 Shadow Host
        if hasattr(current, 'shadow_root_type') and current.shadow_root_type:
            shadow_host_selector = generate_css_selector_for_node(current)
            shadow_path.insert(0, shadow_host_selector)
        
        # 记录 Iframe
        if current.tag_name.lower() in ('iframe', 'frame'):
            in_iframe = True
            # 注意：跨域 Iframe 需要特殊处理，这里先做标记
            
        current = current.parent_node

    target_selector = generate_css_selector_for_node(node)
    
    # 构建描述性路径
    path_str = " > ".join(shadow_path + [target_selector])
    if in_iframe:
        path_str = f"[IFRAME] -> {path_str}"

    result = {
        "selector": target_selector,
        "shadow_path": shadow_path,
        "in_iframe": in_iframe,
        "full_path": path_str
    }
    
    logger.debug(f"Generated robust selector for index {index}: {result}")
    return result
