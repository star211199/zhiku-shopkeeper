import asyncio

import httpx

async def direct_bailian_search(self, query: str):
    url = "https://dashscope.aliyuncs.com/api/v1/services/web-search"  # 假设的端点，需查阅文档
    headers = {
        "Authorization": "Bearer your_dashscope_api_key_here",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "count": 3
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        return resp.json()
