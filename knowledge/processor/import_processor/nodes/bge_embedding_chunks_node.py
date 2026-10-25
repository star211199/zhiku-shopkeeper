import json
from pathlib import Path
from typing import List, Dict, Any
from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.exceptions import StateFieldError, EmbeddingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors


class BgeEmbeddingChunksNode(BaseNode):
    name = "bge_embedding_chunks_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        chunks = self._validate(state)
        # 2. 每一个切片都嵌入成向量
        final_chunks = self._embedding_chunks(chunks)
        # 3. 将final_chunks更新到state中
        state["chunks"] = final_chunks
        # 4. 将final_chunks备份成json文件（方便后续节点的单独测试）
        self._backup_chunks(final_chunks,state)
        return state

    def _validate(self, state:ImportGraphState)->list:
        # 1. 获取chunks
        chunks = state.get("chunks")
        # 2.校验chunks
        if not isinstance(chunks, list) or not chunks:
            self.logger.error("chunks 为空或者不是list")
            raise StateFieldError(node_name=self.name, field_name="chunks",expected_type=list)

        # 3. 校验每一个chunk的类型
        for chunk in chunks:
            if not isinstance(chunk, dict):
                self.logger.error("chunk 不是dict")
                raise StateFieldError(node_name=self.name, field_name="chunks",expected_type=dict)

        # 返回chunks
        return chunks

    def _embedding_chunks(self, chunks:List[Dict[str,Any]])->list:
        # 1. 获取嵌入模型对象
        try:
            embedding_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"获取嵌入模型失败，{e}")
            raise EmbeddingError(node_name=self.name,message=f"获取嵌入模型失败，{e}")

        # 2. 按批次进行批量嵌入
        # 2.1 获取配置的batch_size
        batch_size = self.config.embedding_batch_size
        # 2.2 获取chunks的总长度
        total_length = len(chunks)
        # 2.3 声明final_chunks用来存储向量化后的chunks
        final_chunks = []
        # 2.4 遍历chunks，步长应该是embedding_batch_size，这样就可以按照批次进行向量化
        for i in range(0, total_length, batch_size):
            # 2.4.1 获取当前批次的chunk
            batch_chunks = chunks[i:i+batch_size]
            # 2.4.2 对当前批次的chunk的内容进行向量化
            # 2.4.2.1 将每一个chunk的item_name和content拼接成字符串
            document_list = [f"{chunk.get('item_name')}\n{chunk.get('content')}" for chunk in batch_chunks]
            # 2.2.4.2 对document_list进行向量化
            try:
                embedding_result = generate_bge_m3_hybrid_vectors(embedding_client, document_list)
            except Exception as e:
                self.logger.error(f"嵌入向量失败,{e}")
                raise EmbeddingError(node_name=self.name,message=f"嵌入向量失败,{e}")

            # 2.2.4.3 从embedding_result中获取稠密向量、获取稀疏向量
            dense_vectors = embedding_result.get("dense")
            sparse_vectors = embedding_result.get("sparse")

            # 2.4.3 遍历batch_chunks，向chunk字典中增加dense_vector和sparse_vector这两字段
            for idx, chunk in enumerate(batch_chunks):
                # 重新组装chunks，包含稀疏向量和稠密向量
                chunk["dense_vector"] = dense_vectors[idx]
                chunk["sparse_vector"] = sparse_vectors[idx]
                final_chunks.append(chunk)

        return final_chunks

    def _backup_chunks(self, final_chunks:list[dict[str,Any]], state:ImportGraphState)->None:
        # 1. 指定备份文件的路径
        # 1.1 获取文件输出的目录
        md_path = state.get("md_path","")
        md_path_obj = Path(md_path)
        file_dir = state.get("file_dir","")
        file_dir_obj = Path(file_dir)
        backup_dir = file_dir_obj / md_path_obj.stem
        # 1.2 判断该目录是否真实存在，如果不存在则创建目录
        backup_dir.mkdir(parents=True, exist_ok=True)
        # 1.3 指定备份文件的路径
        backup_file_path = backup_dir / "chunks_vector.json"
        try:
            with open(backup_file_path,"w",encoding="utf-8") as f:
                json.dump(final_chunks,f,ensure_ascii=False,indent=4)
        except Exception as e:
            self.logger.warning(f"{md_path_obj.stem}文件备份成chunks_vector.json失败,但是不影响主流程")


if __name__ == "__main__":
    node = BgeEmbeddingChunksNode()
    # 1. 读取chunks_item_name.json的内容
    json_path = r"D:\AtGuiGu_Java+AI\code\python_workspace\zhiku-shopkeeper\knowledge\processor\import_processor\output_dir\万用表RS-12的使用\chunks_item_name.json"

    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    state = {
        "chunks": chunks,
        "md_path": r"D:\AtGuiGu_Java+AI\code\python_workspace\zhiku-shopkeeper\knowledge\processor\import_processor\output_dir\万用表RS-12的使用\万用表RS-12的使用.md",
        "file_dir": r"D:\AtGuiGu_Java+AI\code\python_workspace\zhiku-shopkeeper\knowledge\processor\import_processor\output_dir",
    }
    node(state)