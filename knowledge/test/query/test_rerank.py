from knowledge.utils.client.ai_clients import AIClients

reranker = AIClients.get_bge_m3_rerank_client()

# 计算相关性得分
pairs = [
    ["什么是万用表？", "万用表是一种测量电压、电流、电阻的仪器"],
    ["什么是万用表？", "今天天气很好"]
]
scores = reranker.compute_score(pairs)
# 输出： [7.895775318145752, -9.480398178100586]
print(scores)