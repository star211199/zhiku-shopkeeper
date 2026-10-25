import json
import re
from logging import INFO
from pathlib import Path
from typing import Tuple, List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge.processor.import_processor.base import BaseNode, T, setup_logging
from knowledge.processor.import_processor.exceptions import StateFieldError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer

"""
document_split_done节点功能的实现：
1、进行参数校验：返回markdown文件的内容，文件标题，获取最小长度和最大长度
2、对内容按照标题进行划分
3、对内容进行二次划分或者合并，满足最小长度和最大长度的范围
4、将切分和合并后的内容组装成后续可直接使用的chunks
5、备份chunks到json文件当中
6、将切分后的内容存储到state中
"""


class DocumentSplitNode(BaseNode):
    """文件切分节点"""
    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        new_md_content, file_title, max_content_length, min_content_length = self._validate_state(state)
        # print(f"new_md_content={new_md_content}")
        # print(f"file_title={file_title}")
        # print(f"max_content_length={max_content_length}")
        # print(f"min_content_length={min_content_length}")
        # 2. 按标题划分
        sections_by_head = self._split_by_head(new_md_content, file_title)
        # for index, section in enumerate(sections_by_head):
        #     print(f"section_{index}={section}")

        # 3. 二次切分或合并
        final_sections = self._split_and_merge(sections_by_head, max_content_length, min_content_length)

        # 4. 将切分和合并后的内容组装成后续可直接使用的chunks
        chunks = self._assemble_chunks(final_sections)
        # 5. 备份chunks到json文件当中
        self._backup_chunks(chunks, state)
        # 6. 将切分后的内容存储到state中
        state["chunks"] = chunks
        return state

    def _split_and_merge(self, sections: List[Dict[str, Any]], max_content_length: int, min_content_length: int) -> \
    List[Dict[str, Any]]:
        """对文档进行二次划分或者合并"""
        # 1. current_sections用于收集二次切分后的section
        current_sections = []
        # 2. 遍历每一个section，对每一个section进行二次切分
        for section in sections:
            split_section = self._split_long_section(section, max_content_length)
            current_sections.extend(split_section)

        # for section in current_sections:
        #     print(section)
        #     print("=" * 60)

        # 针对较小的section进行合并
        final_sections = self._merge_short_section(current_sections, min_content_length)

        # 返回二次切分和合并后的结果
        return final_sections

    def _split_long_section(self, section: Dict[str, Any], max_content_length: int) -> List[Dict[str, Any]]:
        """对长的sections进行二次切分,使用textspliter且切分器进切分"""
        # 判断section的大小是否超过了max_content_length
        # 1.1 计算section的长度
        # 获取section的title
        title = section.get("title", "")
        # 防御性编程：避免title太长
        if len(title) > 80:
            title = title[:80]

        title_prefix = title + "\n\n"
        # 获取body
        body = section.get("body", "")

        # 针对body中的表格进行处理
        if "<table>" in body:
            body = MarkdownTableLinearizer.process(body)
            section["body"] = body

        # 计算长度
        total_length = len(body) + len(title_prefix)

        # 判断是否超出最大的长度
        if total_length <= max_content_length:
            # 没有进行切分，按照原来传入的返回
            return [section]

        # 超出了长度，需要切分
        text_spliter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "？", "！", "；", ".", "?", "!", ';', " ", ""],
            chunk_size=max_content_length,
            keep_separator=True,
            chunk_overlap=0  # 没有设置重叠，如果设置重叠的还，语义容易重复，检索的时候容易匹配到多个相关的片段
        )

        # 使用text_splitter对body进行切分
        split_body_list = text_spliter.split_text(body)
        # 切分后的内容是否有多个字符串
        if len(split_body_list) == 1:
            # 说明没有对section进行切分
            return [section]

        result_sections = []
        for index, split_body in enumerate(split_body_list):
            split_section = {
                "body": split_body,
                "title": f"{title}_{index + 1}",
                "parent_title": section.get("parent_title"),
                "file_title": section.get("file_title")
            }
            result_sections.append(split_section)
        return result_sections

    def _merge_short_section(self, current_sections: List[Dict[str, Any]], min_content_length: int) -> List[Dict[str, Any]]:
        """合并短的sections，两个片段在同一个父标题下且长度小于最小的长度现在才合并"""
        # 1. 获取下标为0的section，记录为current_section
        current_section = current_sections[0]
        # 2. 定义final_sections用于记录收集合并后的section列表
        final_sections = []
        # 3. 从下标为1开始遍历到最后，遍历出来的section记录为next_section
        '''
        合并的条件，文本块足够小，并且两个文本块是同源的（在相同的父标题下）
        '''
        for i in range(1, len(current_sections)):
            next_section = current_sections[i]
            # 判断是否同源
            same_parent = current_section.get("parent_title") == next_section.get("parent_title")
            # 判断current_section的长度是否小于min_content_length
            current_section_body = current_section.get("body", "")
            is_short = len(current_section_body) < min_content_length
            # 如果两个条件都满足就合并
            if is_short and same_parent:
                # 将两个文本块的body进行合并
                merged_body = current_section_body + "\n\n" + next_section.get("body")
                # 将合并后的body赋值给current_section的body
                current_section["body"] = merged_body
                # 将合并后的section的标题title设置成他的parent_title
                current_section["title"] = current_section.get("parent_title")
            else:
                # 如果不需要合并直接添加到final_sections当中
                final_sections.append(current_section)
                # current_section指向next_section重复下去
                current_section = next_section
        # 遍历完以后，将最后的一个current_section添加带哦final_sections当中
        final_sections.append(current_section)
        return final_sections

    def _assemble_chunks(self, final_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组装sections,优化body的内容"""
        # 声明一个chunks列表，用于存储组装后的结果
        chunks = []
        # 遍历final_sections逐个进行组装
        for section in final_sections:
            title = section.get("title", "")
            body = section.get("body", "")
            chunk = {
                "content": f"{title}\n\n{body}",
                "title": title,
                "parent_title": section.get("parent_title"),
                "file_title": section.get("file_title")
            }
            chunks.append(chunk)
        return chunks

    def _backup_chunks(self, chunks: List[Dict[str, Any]], state: ImportGraphState):
        """将chunks存储到本地的json文件当中"""
        # 指定输出的目录
        md_path = state.get("md_path", "")
        md_path_obj = Path(md_path)
        file_dir = state.get("file_dir", "")
        file_dir_obj = Path(file_dir)
        backup_dir = file_dir_obj / md_path_obj.stem
        # 判断路径是否真实存在，如果不存在直接创建
        backup_dir.mkdir(parents=True, exist_ok=True)
        # 指定备份文件的路径
        backup_file_path = backup_dir / "chunks.json"

        # 2. 将chunks写入到备份文件当中
        try:
            with open(backup_file_path, "w", encoding="utf-8") as f:
                # json.dump()将Python对象写入到json文件当中
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"将chunks写入文件失败，失败的原因是：{e}")

    def _validate_state(self, state: ImportGraphState) -> Tuple[str, str, int, int]:
        """校验和清洗数据，替换换行符，校验配置信息"""
        # 1. 统一换行符：将markdown中的"\r"替换成"\n","\r\n"替换成"\n\n"
        md_content = state.get("md_content", "")
        new_md_content = md_content.replace("\r", "\n").replace("\r\n", "\n\n")
        # 2. 校验配置信息：一个切片的最大长度和切片的最小长度
        max_content_length = self.config.max_content_length
        min_content_length = self.config.min_content_length
        if not max_content_length or max_content_length <= 0 or not min_content_length or min_content_length <= 0 or max_content_length < min_content_length:
            self.logger.error(
                f"max_content_length={max_content_length} 或者 min_content_length={min_content_length}配置错误,请检查")
            raise StateFieldError(node_name=self.name, field_name="max_content_length or min_content_length",
                                  expected_type=int)

        # 3. 获取文档标题
        file_title = state.get("file_title", "")

        # 4. 将md_content, file_title, config.max_content_length, config.min_content_length给下一步使用
        return new_md_content, file_title, max_content_length, min_content_length

    def _split_by_head(self, md_content: str, file_title: str) -> List[Dict[str, Any]]:
        """通过标题来切分文档"""
        # 1. md_content按行划分
        md_lines = md_content.split("\n")
        # 匹配标题的正则表达式
        heading_re = re.compile(r"^(#{1,6})\s+(.+)$")
        code_fence_pattern = re.compile(r"^\s*```")
        in_code_fence = False
        # hierarchy存储各级标题
        hierarchy = [""] * 7
        # 存储当前标题块section的内容部分
        body_lines = []
        # current_title 表示当前的标题
        current_title = ""
        # current_level,当前标题的层级
        current_level = 0
        # 声明final_sections,用来存储所有的块
        final_sections = []

        def _flush():
            """将收集的内容封装成session格式
            {
                "body":这块的正文内容,
                "title":这块的标题,
                "parent_title":这块的父标题,
                "file_title":这块所在的文档标题
            }
            """
            # 收集body的内容
            body = "\n".join(body_lines)
            # 收集标题
            title = current_title
            # 判断是否存在current_title、是否存在body,如果不存在不收集信息
            if not body and not title:
                return

            # 如果没有title，就使用file_title给其进行兜底
            title = title if title else file_title

            # 收集parent_title,从当前标题往前遍历hierarchy数组
            parent_title = ""
            for i in range(current_level - 1, 0, -1):
                if hierarchy[i] != "":
                    parent_title = hierarchy[i]
                    break

            sections = {
                "body": body,
                "title": title,
                "parent_title": parent_title if parent_title else title,
                "file_title": file_title
            }

            # 将section添加到final_sections中
            final_sections.append(sections)

        for index, md_line in enumerate(md_lines):
            # 判断是否在代码块当中
            if code_fence_pattern.match(md_line):
                in_code_fence = not in_code_fence

            # 匹配标题
            match = heading_re.match(md_line)
            if match and not in_code_fence:
                # 收集上一个标题块的内容
                _flush()
                # 设置当前标题的值
                current_title = md_line
                # 清空body_lines
                body_lines = []
                # 获取当前标题的层级
                current_level = len(match.group(1))
                # 将当前的标题放入到hierarchy数组中的current_level
                hierarchy[current_level] = current_title
                # 清空hierarchy中current_level下表后的所有元素
                for i in range(current_level + 1, 7):
                    hierarchy[i] = ""
            else:
                # 如果不是标题行或者在代码块当中
                body_lines.append(md_line)
        # 遍历到最后如果没有遇到title了，也需要收集到前一级标题的内容
        _flush()

        return final_sections


if __name__ == "__main__":
    setup_logging(level=INFO)
    node = DocumentSplitNode()

    md_path = r"D:\AtGuiGu_Java+AI\code\python_workspace\zhiku-shopkeeper\knowledge\processor\import_processor\output_dir\万用表RS-12的使用\万用表RS-12的使用_new.md"

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    state = {
        "md_path": r"D:\AtGuiGu_Java+AI\code\python_workspace\zhiku-shopkeeper\knowledge\processor\import_processor\output_dir\万用表RS-12的使用\万用表RS-12的使用_new.md",
        "md_content": md_content,
        "file_title": "万用表RS-12的使用_new.md",
        "file_dir": r"D:\AtGuiGu_Java+AI\code\python_workspace\zhiku-shopkeeper\knowledge\processor\import_processor\output_dir"
    }

    node(state=state)
