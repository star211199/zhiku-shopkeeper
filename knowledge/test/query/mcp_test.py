import os

import httpx
from dotenv import load_dotenv

load_dotenv()


async def direct_bailian_search(query: str):
    """直接以 HTTP 方式调用百炼网络搜索接口（仅用于联调对比）。"""
    url = "https://dashscope.aliyuncs.com/api/v1/services/web-search"
    # API Key 从环境变量读取，禁止硬编码
    api_key = os.getenv("MCP_DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("缺少必需的环境变量: MCP_DASHSCOPE_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "count": 3
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        return resp.json()
