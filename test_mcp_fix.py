import asyncio
import json
from hawker_agent.browser.session import BrowserSession
from hawker_agent.tools.browser_tools import register_browser_tools
from hawker_agent.tools.http_tools import register_http_tools
from hawker_agent.tools.registry import ToolRegistry
from hawker_agent.models.state import CodeAgentState
from hawker_agent.models.history import CodeAgentHistoryList

async def run_test():
    state = CodeAgentState()
    reg = ToolRegistry()
    history = CodeAgentHistoryList.from_task("test", "system")
    
    async with BrowserSession(headless=False) as br:
        # 1. 注册工具
        register_browser_tools(reg, br, history)
        register_http_tools(reg)
        tools = reg.as_namespace_dict()
        
        nav = tools['nav']
        get_cookies = tools['get_cookies']
        http_json = tools['http_json']
        
        # 2. 先访问首页，建立会话并操作筛选触发请求
        print("--- Step 1: Navigating and interacting ---")
        await nav("https://mcp.aibase.com/zh/explore")
        await asyncio.sleep(2)
        # 点击一个筛选器触发真实的 API 请求
        from hawker_agent.browser.actions import click_index as direct_click
        await direct_click(br, 68) # 搜索工具
        await asyncio.sleep(2)
        
        # 3. 检查网络日志中的 Header
        print("--- Step 2: Inspecting Network Log for Headers ---")
        get_network_log = tools['get_network_log']
        log_raw = await get_network_log(filter="querypage")
        
        if isinstance(log_raw, str):
            log = json.loads(log_raw)
        else:
            log = log_raw
        
        target_headers = {}
        if log:
            # 找到最后一个成功的请求
            req = log[-1]
            print(f"Found request: {req['url']}")
            # 这里的字段名可能是 headers
            target_headers = req.get('headers', {})
            print(f"Captured headers keys: {list(target_headers.keys())}")
        
        # 4. 准备请求
        api_url = "https://mcpapi.aibase.cn/api/mcp/querypage?langType=zh_cn"
        payload = {
            "mcpName": "",
            "className": "搜索工具",
            "topics": "",
            "langType": "zh_cn",
            "certStatus": "",
            "devLang": "python",
            "contentLang": "",
            "pos": "",
            "mcpType": 1,
            "pageNo": 1,
            "pageSize": 5,
            "sort": "download|desc"
        }
        
        # 5. 发起带完整 Header 的请求
        print("--- Step 3: Sending request with captured headers ---")
        try:
            # 移除一些可能冲突的系统 header
            for k in ['content-length', 'host', 'connection']:
                target_headers.pop(k.lower(), None)
                
            res = await http_json(api_url, method="POST", json=payload, headers=target_headers)
            print("Successfully bypassed 401 with headers!")
            print(f"Data count: {len(res['data']['list']) if res.get('data') else 0}")
        except Exception as e:
            print(f"Failed with headers: {e}")

if __name__ == "__main__":
    asyncio.run(run_test())
