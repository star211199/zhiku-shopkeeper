from pymilvus.model.hybrid import BGEM3EmbeddingFunction

bge_m3 = BGEM3EmbeddingFunction(
    model_name=r"D:\AtGuiGu_Java+AI\12-java智能体项目之掌柜问数\2.资料\docker\embedding\bge-large-zh-v1.5",
    # 如果安装的torch是gpu版本，这里就是"cuda：θ"，如果安装的torch是cpu版本，这里就是"cpu"
    device_name="cpu",
    # 如果安装的torch是gpu版本，这里就是True，如果安装的torch是cpu版本，这里就是False
    use_fp16=True
)

document_list = ["我是中国人","我是美国人"]
result = bge_m3.encode_documents(document_list)
print(result)
print("="*60)
# 稠密向量
dense = result.get("dense",[])
print(f"dense: {dense}")
print(f"type(dense): {type(dense)}")
print("="*60)
dense_vector_list = []
for item in dense:
    dense_vector = item.tolist()
    dense_vector_list.append(dense_vector)
print(f"dense_vector_list: {dense_vector_list}")
print(len(dense_vector_list))
print("="*60)

# 稀疏向量
sparse_csr = result.get("sparse")
# 获取起始和终止下标
indptr = sparse_csr.indptr
sparse_vector_list = []
for i in range(len(document_list)):
    start_index = indptr[i]
    end_index = indptr[i+1]
    # 获取token_id
    token_list = sparse_csr.indices[start_index:end_index].tolist()
    # 获取data
    weight_list = sparse_csr.data[start_index:end_index].tolist()

    # 稀疏向量
    sparse_vector_list.append(dict(zip(token_list,weight_list)))
print(f"sparse_vector_list: {sparse_vector_list}")


