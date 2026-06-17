from pymilvus.model.hybrid import BGEM3EmbeddingFunction

bge_m3 = BGEM3EmbeddingFunction(
    model_name=r"D:\AtGuiGu_Java+AI\12-java智能体项目之掌柜问数\2.资料\docker\embedding\bge-large-zh-v1.5",
    # 如果安装的torch是gpu版本，这里就是"cuda：θ"，如果安装的torch是cpu版本，这里就是"cpu"
    device_name="cpu",
    # 如果安装的torch是gpu版本，这里就是True，如果安装的torch是cpu版本，这里就是False
    use_fp16=True
)

# 2.嵌入向量
document_list = ["我是中国人"]
result = bge_m3.encode_documents(document_list)
print(result)
print("=" * 60)
# 解析出稠密向量和稀疏向量
# 3.1 稠密向量
dense_vector = result.get("dense")[0].tolist()

# 3.2 稀疏向量
sparse_csr = result.get("sparse")

print(len(dense_vector))
print(dense_vector)
print("=" * 60)
print(sparse_csr)
print("=" * 60)

# indices:表示稀疏向量的token_id,也就是不为0的维度
# data: 表示每个token_id的权重
# indptr: 每个稀疏向量的起始和终止（不包含的）下标
"""
最终想要得到的结果：
{
    "token_id": "权重"
}
"""
# 获取indptr
indptr = sparse_csr.indptr
# 获取向量的起始下标和结束下标
start_index = indptr[0]
end_index = indptr[1]
# 获取indices
indices = sparse_csr.indices
token_id_list = indices[start_index:end_index].tolist()
print(f"token_id_list: {token_id_list}")
# 获取data
data = sparse_csr.data
weight_list = data[start_index:end_index].tolist()
print(f"weight_list: {weight_list}")
# 稀疏向量
sparse_vector = dict(zip(token_id_list, weight_list))
print(sparse_vector)

