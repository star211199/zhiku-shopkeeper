import asyncio
import json
from agents.mcp import MCPServerStreamableHttp
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState


class WebSearchNode(BaseNode):
    name = "web_search_node"
    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 参数校验
        rewritten_query,item_names = self._validate(state)
        # 2. 调用mcp工具进行网络检索
        execute_tool_result = asyncio.run(self.mcp_web_search(rewritten_query))
        print(execute_tool_result)
        # 3. 对mcp检索的结果进行格式化
        # 3.1 获取查询结果中的text，这其实就是一个json字符串
        result_text = execute_tool_result.content[0].text
        # 3.2 对json字符串进行反序列化
        json_object = json.loads(result_text)
        # 3.3 取出pages
        pages = json_object["pages"]

        # 声明一个存储web_search结果的容器
        web_search_docs = []
        # 3.4 遍历pages
        for page in pages:
            # 3.4.1 获取snippet、title、url
            snippet = page["snippet"]
            title = page["title"]
            url = page["url"]
            web_search_docs.append({
                "snippet": snippet,
                "title": title,
                "url": url
            })

        # 4. 将web_search_docs存入state中
        state["web_search_docs"] = web_search_docs

        return state

    async def mcp_web_search(self, rewritten_query: str):
        # 1. 创建MCP客户端
        async with MCPServerStreamableHttp(
                name="search_mcp",
                params={
                    "url": self.config.mcp_dashscope_base_url,  # MCP 服务端点
                    "headers": {"Authorization": "Bearer your_dashscope_api_key_here"},  # 认证头
                    "timeout": 300,  # 请求超时时间（秒）
                    "terminate_on_close": True,  # 关闭时终止连接
                },
                max_retry_attempts=2,  # 最大重试次数
                cache_tools_list=True,  # 缓存工具列表，避免重复请求
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