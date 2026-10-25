from typing import Tuple, List
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.exceptions import StateFieldError, EmbeddingError, LLMError, MilvusError
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import HYDE_USER_PROMPT_TEMPLATE,HYDE_SYSTEM_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import _item_names_filter, create_hybrid_search_requests, execute_hybrid_search_query

"""
hyde_search_node的实现：
1、先将商品名和问题交给大模型，生成假设性答案
2、将问题和假设性答案拼接，使用milvus向量数据库进行检索
3、将检索到的结果返回
"""
class HydeSearchNode(BaseNode):
    """先让大模型根据问题生成答案，然后拼接问题和答案拿到向量数据库中进行检索"""
    name = "hyde_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 校验参数
        rewritten_query, item_names = self._validate(state)
        # 2. 调用大模型生成假设性答案
        hy_document = self._generate_hy_document(rewritten_query,item_names)

        # 3. 获取嵌入模型对象，milvus客户端对象
        try:
            embedding_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"获取嵌入模型对象失败,{e}")
            raise EmbeddingError(message=f"获取嵌入模型对象失败,{e}")

        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"获取milvus客户端对象失败,{e}")
            raise MilvusError(message=f"获取milvus客户端对象失败,{e}")

        # 4. 将客户问题与假设性答案进行拼接
        embedding_input = f"{rewritten_query}\n{hy_document}"
        # 5. 将第四步拼接的内容进行向量化
        embedding_result = generate_bge_m3_hybrid_vectors(embedding_client,[embedding_input])

        # 6. 构建过滤条件
        expr,expr_params = _item_names_filter(item_names)

        # 7. 创建混合检索请求
        hybrid_search_requests = create_hybrid_search_requests(
            dense_vector=embedding_result.get("dense")[0],
            sparse_vector=embedding_result.get("sparse")[0],
            expr=expr,
            expr_params=expr_params,
            limit=self.config.hyde_search_limit
        )
        # 8. 执行混合检索请求
        hyde_search_result = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_search_requests,
            limit=self.config.hyde_search_limit,
            output_fields=["item_name", "title", "content"]
        )
        # print(hyde_search_result)
        # 9. 将hyde检索的结果存入state,返回
        # state["hyde_embedding_chunks"] = hyde_search_result[0]

        return {"hyde_embedding_chunks":hyde_search_result[0]} # type:ignore

    def _validate(self, state) -> Tuple[str, List[str]]:
        # 1. 获取rewritten_query,item_names
        rewritten_query = state.get("rewritten_query","")
        item_names = state.get("item_names",[])
        # 2. 校验rewritten_query,item_names
        if not rewritten_query or not isinstance(rewritten_query,str):
            self.logger.error(f"rewritten_query不能为空以及类型必须是str")
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            self.logger.error("item_names不能为空以及类型必须是list")
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        return rewritten_query,item_names

    def _generate_hy_document(self, rewritten_query, item_names):
        """将商品名和问题交给大模型时，让大模型生成解释性答案"""
        # 1. 获取llm客户端
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except Exception as e:
            self.logger.error(f"创建大模型客户端失败,{e}")
            raise LLMError(node_name=self.name, message=f"创建大模型客户端失败,{e}")

        # 2. 构建SystemMessage和HumanMessage
        hyde_system_prompt = HYDE_SYSTEM_PROMPT_TEMPLATE.format(item_names=item_names)
        hyde_user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(
            item_names=item_names,
            rewritten_query=rewritten_query
        )
        system_message = SystemMessage(content=hyde_system_prompt)
        human_message = HumanMessage(content=hyde_user_prompt)

        # 3. 调用大模型
        try:
            llm_result = llm_client.invoke([system_message, human_message])
        except Exception as e:
            self.logger.error(f"调用大模型客户端失败，{e}")
            raise LLMError(node_name=self.name,message=f"调用大模型客户端失败，{e}")

        return llm_result.content

if __name__ == "__main__":
    node = HydeSearchNode()
    state = {
        "rewritten_query": "华为擎云 G540 计算机的使用方法是什么?",
        "item_names": ["华为擎云 G540 计算机"]
    }
    final_state = node.process(state)
    print(final_state)
