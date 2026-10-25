import json
import re
from logging import Logger
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.config import QueryConfig
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import ITEM_NAME_SYSTEM_EXTRACT_TEMPLATE, ITEM_NAME_USER_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages


class _ItemNameExtractor:
    """将原始的问题交给大模型，让大模型提取出商品名，然后再去进行向量的检索"""

    def __init__(self, logger: Logger, name: str):
        self._logger = logger
        self._name = name

    def extract_item_name(self, original_query: str, formatted_history_str: str):
        """调用大模型获取商品名"""
        # 大模型调用后返回的结构
        llm_result = {
            "item_names": [], # 根据问题和历史对话信息提取商品名
            "rewritten_query": original_query # 重写用户的原始问题，包含商品名
        }
        # 1. 创建LLM对象
        try:
            llm_client = AIClients.get_llm_client(response_format=True)
        except Exception as e:
            self._logger.error(f"创建大模型对象失败，{e}")
            # 如果调用大模型失败，直接返回空结果
            return llm_result

        # 2. 组装prompt消息
        system_message = SystemMessage(content=ITEM_NAME_SYSTEM_EXTRACT_TEMPLATE)
        user_prompt = ITEM_NAME_USER_EXTRACT_TEMPLATE.format(
            history_text=formatted_history_str,
            query=original_query
        )
        human_message = HumanMessage(content=user_prompt)
        # 3. 调用大模型拿到结果
        try:
            llm_result = llm_client.invoke([system_message, human_message])
        except Exception as e:
            self._logger.error(f"调用大模型对象失败，{e}")
            return llm_result

        # 4. 清洗结果
        llm_content = llm_result.content
        if not llm_content:
            return llm_result

        # 5. 返回清洗后的结果
        return self._clean_and_parse(llm_content)

    def _clean_and_parse(self, llm_content: str) -> Dict[str, Any]:
        # 1. 去掉代码块
        # 1.1 去掉前面的 ```
        content = re.sub(r"^```(?:json)?\s*", "", llm_content)
        # 1.2 去掉后面的```
        content = re.sub(r"\s*```$", "", content)

        # 2. 将llm_content进行反序列化
        llm_content_obj: Dict[str, Any] = json.loads(content)
        # 2.1 获取item_names
        original_item_names = llm_content_obj.get("item_names")
        # 2.2 判断original_item_names的类型
        if not isinstance(original_item_names, list):
            item_names = []
        else:
            # 将original_item_names中的每一个字符串去空格之后，收集到item_names中
            item_names = [item_name.strip() for item_name in original_item_names if
                          isinstance(item_name, str) and item_name.strip()]

        # 2.3 获取rewritten_query
        original_rewritten_query = llm_content_obj.get("rewritten_query")
        # 2.4 判断original_rewritten_query的类型
        if not isinstance(original_rewritten_query, str):
            rewritten_query = ""
        else:
            rewritten_query = original_rewritten_query.strip()

        # print(f"item_names:{item_names},rewritten_query:{rewritten_query}")
        # 3. 返回Dict
        return {"item_names": item_names, "rewritten_query": rewritten_query}


class _ItemNameAligner:
    def __init__(self, logger: Logger, config: QueryConfig):
        self._logger = logger
        self._config = config

    def search_and_align(self, item_names: List[str]):
        # 1. 混合检索向量数据库
        search_result = self._search_vector(item_names)
        # print(search_result)
        # 2. 判断检索到的结构，如果为空，则表示confirmed和options都没有
        if not search_result:
            return [], []
        # 2. 根据混合向量检索到的结果做对齐【confirmed/options】
        confirmed, options = self._align(search_result)
        # print(f"confirmed:{confirmed}")
        # print(f"options:{options}")

        # 3. 分数差异化过滤
        # 3.1如果confirmed的长度大于1,则进行分数差异化过滤
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)

        return confirmed, options

    def _search_vector(self, item_names: List[str]):
        """
        混合向量检索商品名
        [{
            "extracted_name":"大模型提取的item_name",
            "matches":[{"item_name":"从向量数据库匹配到的item_name","score":分数}]
        }]
        :param item_names:
        :return:
        """
        # 1. 创建嵌入模型对象
        # 初始化检索的结果
        final_search_result = []
        try:
            embedding_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self._logger.error(f"获取嵌入模型失败,{e}")
            # 如果获取嵌入模型失败，返回空结果
            return final_search_result

        # 2. 对item_names进行想量化
        try:
            embedding_result = generate_bge_m3_hybrid_vectors(embedding_client, item_names)
            dense_vector_list = embedding_result.get("dense")
            sparse_vector_list = embedding_result.get("sparse")
        except Exception as e:
            self._logger.error(f"向量嵌入失败，{e}")
            return final_search_result
        # 创建milvus_client对象
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self._logger.error(f"获取milvus客户端失败，{e}")
            return final_search_result

        # 创建混合向量检索的请求
        # 3.1 对item_names进行遍历
        for index, item_name in enumerate(item_names):
            # 创建混合检索请求
            search_requests = create_hybrid_search_requests(
                dense_vector=dense_vector_list[index],
                sparse_vector=sparse_vector_list[index],
                limit=self._config.embedding_search_limit
            )
            # 执行混合检索
            milvus_search_results = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self._config.item_name_collection,
                search_requests=search_requests,
                limit=self._config.embedding_search_limit,
                output_fields=["item_name"]
            )
            # print(milvus_search_results)
            # 组装匹配结果
            matches = []
            for m in milvus_search_results[0]:
                matches.append({
                    "item_name": m["entity"]["item_name"],
                    "score": m["distance"]
                })

            final_search_result.append({
                "extracted_name": item_name,
                "matches": matches
            })

        return final_search_result

    def _align(self, search_result: List[Dict[str, Any]]):
        # 1. 定义两个容器：confirmed，options，分别存放确定好的商品名、以及待确认的商品名
        confirmed = []
        options = []
        # 2. 遍历检索结果
        for result in search_result:
            # 2.1 获取extracted_name 和 matches
            extracted_name = result.get("extracted_name")
            matches = result.get("matches") or []
            # 2.2 对matches按照score从大到小进行排序
            matches = sorted(matches, key=lambda m: m["score"], reverse=True)
            # 2.3 收集score>item_name_high_confidence的匹配结果，也就是高可置信的结果
            high_result = [m.get("item_name") for m in matches if
                           m.get("score") >= self._config.item_name_high_confidence]
            # 2.4 如果有高可置信的结果，对高置信度的商品名再次判断
            if high_result:
                # 2.4.1. 判断结果中的item_name与LLM提取的item_name是否一模一样，如果有，则将其添加到confirmed容器中（需要去重，要排除它已经在confirmed的情况）
                same_item_name = [h for h in high_result if h == extracted_name and h not in confirmed]
                if same_item_name: # 提取到的商品名和向量数据库中的商品名完全相同
                    confirmed.extend(same_item_name)
                elif len(high_result) == 1: # 没有相同的，但是高置信度的只有一个
                    # 2.4.1 如果没有一模一样的，但是只有一个高可置信的结果，则将这个结果的item_name添加到confirmed容器中（需要去重，要排除它已经在confirmed的情况）
                    high = high_result[0]
                    if high not in confirmed:
                        confirmed.append(high)
                else:
                    # 2.4.2 如果没有一模一样的，并且有多个高可置信的结果
                    # 2.4.2.1 判断最高分是不是比第二名高的分数超过阈值item_name_score_gap，如果超过，则将最高分的结果的item_name添加到confirmed容器中（需要去重）
                    max_score = matches[0]["score"]
                    if max_score - matches[1]["score"] > self._config.item_name_score_gap:
                        max_item_name = matches[0]["item_name"]
                        if max_item_name not in confirmed:
                            confirmed.append(max_item_name)
                    else:
                        # 2.4.2.2 2. 如果最高分比第二名高的分数没有超过阈值，则遍历高可置信结果集，有阈值设置了最大个数item_name_max_options，将这几个结果的item_name添加到options容器中（需要去重: 要排除它已经在confirmed、options中的情况）
                        for high in high_result[:self._config.item_name_max_options]:
                            if high not in confirmed and high not in options:
                                options.append(high)
            else:
                # 2.5 没有高可置信的结果
                # 2.5.1 收集中等可置信结果
                middle_result = [m.get("item_name") for m in matches if
                                 m.get("score") >= self._config.item_name_mid_confidence]
                # 2.5.2 如果有中等可置信的结果集，则将其item_name添加到options中，最多不能超过item_name_max_options个
                if middle_result:
                    for mid in middle_result[:self._config.item_name_max_options]:
                        if mid not in confirmed and mid not in options:
                            options.append(mid)

        return confirmed, options[:self._config.item_name_max_options]

    def _item_name_score_filter(self, confirmed: List[str], search_result: List[Dict[str, Any]]):
        # 1. 找到confirmed中所有item_name的最大得分
        # 1.1 建立商品名与分数的dict
        # 构建 商品名 → 最高分数 的映射
        item_name_score = {}
        for result in search_result:
            matches = result.get("matches", [])
            for m in matches:
                item_name = m.get("item_name")
                score = m.get("score")
                if not item_name or score is None:
                    continue
                item_name_score[item_name] = max(item_name_score.get(item_name, score), score)

        # 1.2 获取confirmed中所有item_name的最大得分
        max_score = 0
        for item_name in confirmed:
            max_score = max(max_score, item_name_score.get(item_name, 0))

        # 2. 判断每一个item_name的得分，相较于最大得分的差距是否超过阈值，超过阈值则丢弃
        final_confirmed = []
        for item_name in confirmed:
            if max_score - item_name_score.get(item_name, 0) <= self._config.item_name_score_gap:
                final_confirmed.append(item_name)

        return final_confirmed

"""
item_name_confrimed_node节点的实现：
1、获取用户对话原始问题和会话历史
2、调用大模型提取商品名
3、从向量数据库中搜索商品名，然后根据分数对齐
"""
class ItemNameConfirmedNode(BaseNode):
    name = "item_name_confirmed_node"

    def __init__(self):
        super().__init__()
        self._extractor = _ItemNameExtractor(self.logger, self.name)
        self._item_name_aligner = _ItemNameAligner(self.logger, self.config)

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 获取用户的原始问题
        original_query = state["original_query"]
        # 2. 获取历史的对话(mongodb)
        # 获取session_id
        session_id = state.get("session_id", "")
        # 获取历史会话结果
        history_messages = get_recent_messages(
            session_id=session_id
        )
        # 将历史会话信息存入state
        state["history"] = history_messages
        formatted_history_str = ""
        for msg in history_messages:
            role = msg.get("role")
            content = msg.get("text", "")
            formatted_history_str += f"{role}:{content}\n"

        # 3. 获取LLM结果，将问题交给大模型，提取商品名
        llm_result: Dict[str, Any] = self._extractor.extract_item_name(original_query, formatted_history_str)
        # 4. 根据item_names做判断，进行商品名的对齐
        extracted_item_names = llm_result.get("item_names")
        # 商品名对齐，将提取到的商品名分为高置信度的和中等置信度的
        confirmed, options = self._item_name_aligner.search_and_align(extracted_item_names)
        # 5. 决策
        rewritten_query = llm_result.get("rewritten_query", "")
        self._decide(confirmed, options, state, rewritten_query)
        return state

    def _decide(self, confirmed: List[str], options: List[str], state: QueryGraphState, rewritten_query: str):
        # 1. 判断confirmed中是否有数据
        if confirmed:
            state["item_names"] = confirmed
            state["rewritten_query"] = rewritten_query
        elif options:
            state["answer"] = (
                f"我不能确认你指的是哪款产品,您是在询问以下产品:{'、'.join(options)}吗"
            )
        else:
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"


if __name__ == "__main__":
    node = ItemNameConfirmedNode()
    state = {
        "original_query": "华为擎云 G540 计算机和万用表如何使用？"
    }
    result = node(state)
    print(result)
