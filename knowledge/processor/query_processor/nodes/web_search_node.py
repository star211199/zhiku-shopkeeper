import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from agents.mcp import MCPServerStreamableHttp
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState


class WebSearchNode(BaseNode):
    name = "web_search_node"
    def process(self, state: QueryGraphState) -> QueryGraphState:
        rewritten_query, item_names = self._validate(state)
        if not self.config.mcp_dashscope_base_url or not self.config.mcp_dashscope_api_key:
            self.logger.warning("未配置 MCP 地址或 API Key，跳过网络检索")
            return {"web_search_docs": []}  # type:ignore

        try:
            execute_tool_result = self._run_coro(self.mcp_web_search(rewritten_query))
            contents = getattr(execute_tool_result, "content", None) or []
            if not contents:
                return {"web_search_docs": []}  # type:ignore

            result_text = contents[0].text
            json_object = json.loads(result_text)
            pages = json_object.get("pages") or []
        except Exception as e:
            self.logger.warning(f"网络检索失败，将跳过网页结果: {e}")
            return {"web_search_docs": []}  # type:ignore

        web_search_docs = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            web_search_docs.append({
                "snippet": page.get("snippet", ""),
                "title": page.get("title", ""),
                "url": page.get("url", ""),
            })

        return {"web_search_docs": web_search_docs}  # type:ignore

    @staticmethod
    def _run_coro(coro):
        """在已有/没有事件循环时都能执行协程，避免 asyncio.run 冲突。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    async def mcp_web_search(self, rewritten_query: str):
        async with MCPServerStreamableHttp(
                name="search_mcp",
                params={
                    "url": self.config.mcp_dashscope_base_url,
                    "headers": {"Authorization": f"Bearer {self.config.mcp_dashscope_api_key}"},
                    "timeout": 300,
                    "terminate_on_close": True,
                },
                max_retry_attempts=2,
                cache_tools_list=True,
        ) as client:
            # 2. 调用mcp中的网络搜索的工具
            execute_tool_result = await client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": rewritten_query, "count": 3}
            )

            return execute_tool_result

    def _validate(self, state) -> tuple[str, list[str]]:
        # 获取rewritten_query,item_names
        rewritten_query = state.get("rewritten_query","")
        item_names = state.get("item_names",[])

        # 校验参数
        if not rewritten_query or not isinstance(rewritten_query,str):
            self.logger.error("rewritten_query不能为空以及类型必须是str")
            raise StateFieldError(node_name=self.name,field_name="rewritten_query",expected_type=str)

        if not item_names or not isinstance(item_names,list):
            self.logger.error("item_names不能为空以及类型必须是list")
            raise StateFieldError(node_name=self.name,field_name="item_names",expected_type=list)

        return rewritten_query,item_names

if __name__ == "__main__":
    node = WebSearchNode()
    state = {
        "rewritten_query": "华为擎云 G540 计算机的使用方法是什么?",
        "item_names": ["华为擎云 G540 计算机"]
    }
    final_state = node.process(state)
    print(final_state)