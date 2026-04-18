# case1:

1. 打开网址 https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix 
2. 找到并点击 "Files and versions" 标签页
3. 下滑页面到最底部，加载更多 "Load more files"
4. 遍历所有子文件夹
5. 提取页面的文件名和下载URL
样本数据:
{"file_name":".gitattributes","download_url":"https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix/resolve/main/.gitattributes?download=true"}


---

# case2:

1. 打开 https://www.ahnews.com.cn/df/hss/pc/lay/node_525.html
2. 点击"下一页"，获取3页数据
3. 获取列表页URL和title
样本数据:
{"title":"专人守护、一树一策 黄山多措并举保护古松树","URL":"http://www.ahnews.com.cn/anhui/pc/con/2026-01/23/562_1662432.html"}


---

# case3:
1. 打开网址 https://mcp.aibase.com/zh/explore 
2. 找到下一页按钮，获取前10页数据 ，分类点击 "搜索工具",认证状态点击"不限"，编程语言是"python"，类型 "MCP Server"，点击按"按下载量"排序
3. 提取所有name、desc(简介)、url
样本数据:
{"name":"Klavis","desc":"Klavis AI是一个开源项目，提供在Slack、Discord和Web平台上简单易用的MCP（模型上下文协议）服务，包括报告生成、YouTube工具、文档转换等多种功能，支持非技术用户和开发者使用AI工作流。","url":"https://mcp.aibase.com/zh/server/1528363509283561529"}


---

# case4:

1. 访问维基百科网站：https://en.wikipedia.org
2. 分别搜索 OpenAI、APPLE 二家公司
3. 打开维基百科的相关文章
4. 从信息框和文章中提取以下内容：名称、成立日期、总部、现任首席执行官/领导者、所属行业、主要产品/服务（列出）、收入（如有则提供）、员工人数（如有则提供），以及 2-3 句概括
提取字段:
- name: "公司名称"
- founded: "成立时间"
- headquarters: "总部地址"
- ceo: "现任 CEO"
- industry: "所属行业"
- products: "主要产品/服务列表"
- revenue: "营收（如有）"
- employees: "员工数量（如有）"
- summary: "公司简介（2-3句话）"

---

# case5:

1. 打开 https://github.com/trending
2. 获取当前页面的项目URL、start、fork、today_start
提取字段: 
- URL: 项目链接 
- start: start数 
- fork： fork数
- today_start: today_start数

---

# case6:

1. 打开 https://arxiv.org/search/advanced
2. 搜素web agent 论文，查找2026年1月14到2026年3月20之间的
3. 如果有"Next",获取下一页
4. 返回该条件下论文的下载链接(pdf格式)
---

# case7:
从以下content引用中找每篇论文的下载链接并下载PDF到本地，同时返回每篇论文的摘要和研究领域、下载链接、编号[81]返回JSON。
1. content='''
[86] Lutfi Eren Erdogan, Hiroki Furuta, Sehoon Kim, Nicholas Lee, Suhong Moon, Gopala Anumanchipalli, Kurt Keutzer, and Amir Gholami. Plan-and-Act: Improving planning of agents
for long-horizon tasks. In Forty-second International Conference on Machine Learning, 2025.
URL https://openreview.net/forum?id=ybA4EcMmUZ.
[87] Jae-Woo Choi, Hyungmin Kim, Hyobin Ong, Youngwoo Yoon, Minsu Jang, Jaehong Kim, et al.
Reactree: Hierarchical task planning with dynamic tree expansion using llm agent nodes. 2025.
[88] Siddharth Nayak, Adelmo Morrison Orozco, Marina Ten Have, Vittal Thirumalai, Jackson
Zhang, Darren Chen, Aditya Kapoor, Eric Robinson, Karthik Gopalakrishnan, James Harrison, et al. LLaMAR: Long-horizon planning for multi-agent robots in partially observable
environments. arXiv preprint arXiv:2407.10031, 2024.
[89] Anthropic. Model Context Protocol (MCP), 2024. URL https://www.anthropic.com/news
/model-context-protocol.
[90] Yingxuan Yang, Huacan Chai, Yuanyi Song, Siyuan Qi, Muning Wen, Ning Li, Junwei Liao,
Haoyi Hu, Jianghao Lin, Gaowei Chang, Weiwen Liu, Ying Wen, Yong Yu, and Weinan Zhang.
A survey of AI agent protocols, 2025. URL https://arxiv.org/abs/2504.16736.
[91] Yu Wang and Xi Chen. Mirix: Multi-agent memory system for llm-based agents, 2025. URL
https://arxiv.org/abs/2507.07957.
[92] Kai Mei, Xi Zhu, Wujiang Xu, Wenyue Hua, Mingyu Jin, Zelong Li, Shuyuan Xu, Ruosong
Ye, Yingqiang Ge, and Yongfeng Zhang. Aios: Llm agent operating system. arXiv preprint
arXiv:2403.16971, 2024.
'''

---

# case8:
1. 打开google搜索
2. 检索web agent Thesis 获取前5条论文
3. 返回前5条论文标题，下载链接，摘要，研究领域

---

测试表格

| ID    | Name                      | Status | Steps | Duration | Tokens  | crawl_Data | total_data | percent |
|-------|---------------------------|--------|-------|----------|---------|------------|------------|---------|
| case1 | HuggingFace Dataset Files | PASS   | 21    | 656.0s   | 228,091 | 214        | 220        | 13%     |
| case2 | AHNews List Pages         | PASS   | 29    | 1063.8s  | 535,057 | 0          | 72         | 0%      |
| case3 | MCP AIBase Explore        | PASS   | 22    | 804.2s   | 256,679 | 120        | 343        | 34%     |
| case4 | Wikipedia Research        | PASS   | 14    | 260.0s   | 138,767 | 2          | 2          | 100%    |
| case5 | GitHub Trending           | PASS   | 15    | 100.6s   | 60,536  | 14         | 14         | 100%    |
| case6 | arXiv Paper Search        | PASS   | 24    | 217.9s   | 106,572 | 50         | 57         | 87%     |
| case7 | 论文引用附录                    | PASS   | 11    | 390.9s   | 126,720 | 6          | 7          | 85%     |
| case8 | google serach Thesis      | PASS   | 4     | 125.1s   | 35,651 | 5          | 5          | 100%    |
