from pymilvus import MilvusClient

# 1. 获取 Milvus 客户端
milvus_client = MilvusClient(uri="http://127.0.0.1:19530")
# 2. 稠密向量检索
# 实际业务当中，是将提取的商品名通过bge3模型转化成稠密向量
dense_vector = [0.123, -0.456, 0.789, ...]  # 1024维
results = milvus_client.search(
    collection_name="kb_chunks",  # 向量数据表
    data=[dense_vector],  # 查询的向量
    anns_field="dense_vector",  # 稠密向量的字段
    param={"metric_type": "COSINE"},  # 相似度度量
    limit=5,  # 返回的数据条数的限制
    output_fields=["content", "item_name"]  # 输出向量数据库查询到的数据的字段
)
# 3. 解析结果
for hit in results[0]:
    print(f"内容: {hit['entity']['content']}")
    print(f"分数: {hit['distance']}")


# 2. 稀疏向量检索
sparse_vector = {1234: 0.85, 5678: 0.72, 9012: 0.68}  # {token_id: weight}

results = milvus_client.search(
    collection_name="kb_chunks",
    data=[sparse_vector],           # 查询稀疏向量
    anns_field="sparse_vector",     # 稀疏向量字段名
    param={"metric_type": "IP"},    # 内积度量
    limit=5,
    output_fields=["content", "item_name"]
)

