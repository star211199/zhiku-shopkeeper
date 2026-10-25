import json
from typing import List, Dict, Any, Tuple
from pymilvus import MilvusClient
from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.exceptions import StateFieldError, EmbeddingError, MilvusError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients


class _MilvusSchemaBuilder:
    """创建向量数据库schema"""
    @classmethod
    def build_schema(cls, milvus_client: MilvusClient, dim: int):
        # 创建Schema
        schema = milvus_client.create_schema()
        # 添加字段
        from pymilvus import DataType
        # 自增主键
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            auto_id=True,
            is_primary=True
        )
        # 标量
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="title",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="parent_title",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        # 向量
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )
        return schema


class _IndexParamsBuilder:
    """为向量字段创建索引"""
    @classmethod
    def build_index_params(cls, milvus_client: MilvusClient):
        # 创建索引
        index_params = milvus_client.prepare_index_params()
        # 添加索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )
        return index_params


class MilvusImportNode(BaseNode):
    name = "milvus_import_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验，校验chunks是否为空，是否有向量
        final_chunks, dim = self._validate_state(state)
        # print(final_chunks)
        # print(dim)

        # 2. 向量入库
        self._insert_data(final_chunks, dim)

        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], int]:
        # 1. 获取chunks
        chunks = state.get("chunks")
        # 2. 校验chunks是否为空、类型是否是list
        if not isinstance(chunks, list) or not chunks:
            self.logger.error("chunks 为空，或者类型不是list")
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)
        # 3. 遍历每一个chunk，校验chunk是否是dict，chunk是否有向量，如果没有向量则将这个chunk丢弃
        final_chunks = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            # 获取稠密向量和稀疏向量
            dense_vector = chunk.get("dense_vector", [])
            sparse_vector = chunk.get("sparse_vector")
            if not dense_vector or not sparse_vector:
                continue
            dim = len(dense_vector)
            final_chunks.append(chunk)

        # 4. 判断final_chunks是否为空
        if not final_chunks:
            self.logger.error("所有的chunk都没有向量，导入失败")
            raise EmbeddingError("所有chunk都没有向量,导入失败")
        return final_chunks, dim

    def _insert_data(self, final_chunks: List[Dict[str, Any]], dim: int):
        # 1. 创建MilvusClient
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"获取milvus客户端失败,{e}")
            raise MilvusError(f"获取milvus客户端失败,{e}")

        # 2. 从配置中获取存储chunk的Collection的name
        collection_name = self.config.chunks_collection
        # 3. 判断Collection是否存在,如果不存在则创建Collection
        if not milvus_client.has_collection(collection_name):
            # 3.1 创建Schema
            schema = _MilvusSchemaBuilder.build_schema(milvus_client, dim)
            # 3.2 创建索引
            index_params = _IndexParamsBuilder.build_index_params(milvus_client)
            # 3.3 创建collection
            milvus_client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
                timeout=30
            )
        # 4. 插入数据,获取自增长的id
        insert_results = milvus_client.insert(
            collection_name=collection_name,
            data=final_chunks  # 会自动匹配字段
        )
        ids = insert_results.get("ids")
        # 5. 将自增长的id回填到final_chunks中
        for i, chunk in enumerate(final_chunks):
            chunk["chunk_id"] = ids[i]


if __name__ == "__main__":
    node = MilvusImportNode()

    chunks_file = r"D:\AtGuiGu_Java+AI\code\python_workspace\zhiku-shopkeeper\knowledge\processor\import_processor\output_dir\万用表RS-12的使用\chunks_vector.json"
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    state = {
        "chunks": chunks
    }

    node(state)
