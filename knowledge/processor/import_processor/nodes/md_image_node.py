import re
from logging import DEBUG, INFO, Logger
from pathlib import Path
from typing import Tuple, List, Dict
from openai import OpenAI
from knowledge.processor.import_processor.base import BaseNode, T, setup_logging
from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.exceptions import StateFieldError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients


class ImageContext:
    """封装图像的内容，给vlm模型总结"""
    head_title: str  # 图片所在的目录标题
    pre_context: str  # 图片的上文
    post_context: str  # 图片的下文

    def __init__(self, head_title: str, pre_context: str, post_context: str):
        self.head_title = head_title
        self.pre_context = pre_context
        self.post_context = post_context


class ImageInfo:
    """封装图像的信息"""
    image_name: str
    image_path: str
    image_context: ImageContext  # 图像的内容

    def __init__(self, image_name: str, image_path: str, image_context: ImageContext):
        self.image_name = image_name
        self.image_path = image_path
        self.image_context = image_context


class _MdFileHandler:
    """处理markdown文件的类"""

    def __init__(self, logger, node_name):
        self.logger = logger
        self.node_name = node_name

    """获取markdown文件的信息，文件内容，文件路径，图片路径"""

    def read_md(self, md_path: str) -> Tuple[str, Path, Path]:
        # 1.校验md_path是否为空
        if not md_path:
            self.logger.error("md_path路径不能为空")
            raise StateFieldError(node_name=self.node_name, field_name="md_path", expected_type=str)

        # 2. 校验md_path是否存在
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            self.logger.error("md_path文件路径不是真实存在的")
            raise StateFieldError(node_name=self.node_name, field_name="md_path", expected_type=str)

        # 3. 读取md文件内容到内存中
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # 4. 获取md文件图片的目录路径
        image_dir_obj = md_path_obj.parent / "images"

        # 5. 返回内容
        return md_content, md_path_obj, image_dir_obj

    def backup_md(self,md_content:str,md_path:str)->None:
        # 1. 构建备份文件的路径,在原文件路径下，但名字但是名字叫做  原文件名_new.原文件后缀
        md_path_obj = Path(md_path)
        backup_md_path_obj = md_path_obj.parent / (md_path_obj.stem + "_new" + md_path_obj.suffix)
        # 将md_content写入到备份的文件当中
        try:
            with open(backup_md_path_obj, 'w', encoding='utf-8') as f:
                f.write(md_content)
        except Exception as e:
            self.logger.warning(f"备份md文件失败: {e},但是不影响程序的运行")
        self.logger.info(f"文件备份成功，文件路径是{str(backup_md_path_obj)}")


class _ImageScanner:
    """图像信息获取，上下文信息"""

    def __init__(self, config: ImportConfig):
        self.config = config

    def scan_img_dir(self, image_dir_obj: Path, md_content: str) -> Tuple[List[ImageInfo], List[str]]:
        image_info_list = []
        md_lines = md_content.split("\n")
        for image_file in image_dir_obj.iterdir():
            # 1. 过滤子目录
            if not image_file.is_file():  # 不是文件就是目录
                continue
            # 2. 过滤非图片文件
            if image_file.suffix not in self.config.image_extensions:
                continue

            # 3. 获取上下文组装信息
            # 3.1 按行切割文档

            # 3.2 匹配md中图片的正则表达式
            pattern = re.compile(r"!\[.*?]\(.*?" + re.escape(image_file.name) + r".*?\)")

            # 判断是否在代码块单中
            in_code_fence = False  # 默认不在代码代码块当中
            # 预定义匹配代码块的正则表达式
            code_fence_pattern = re.compile(r"^\s*```")

            # 使用for循环遍历md_lines
            for index, line in enumerate(md_lines):
                # 判断是否在代码块当中
                if code_fence_pattern.match(line):
                    in_code_fence = not in_code_fence

                # 判断是否匹配到图片
                if in_code_fence or not pattern.match(line):
                    continue

                # 匹配标题的正则表达式
                heading_pattern = re.compile(r"^#{1,6}\s+(.+)$")

                # 获取图片上文部分
                head_title, head_content = self._find_heading_content(md_lines, heading_pattern, index)

                # 获取图片的下文部分
                post_content = self._find_post_content(md_lines, heading_pattern, index)

                # 截取图片上下文信息，防止上下文内容过长
                image_content_length = self.config.img_content_length
                # 截取上文信息
                pre_context = self._extract_image_content(head_content, image_content_length, "up")
                # 截取下文信息
                post_context = self._extract_image_content(post_content, image_content_length, "down")
                # 封装ImageContent对象
                image_content = ImageContext(head_title, pre_context, post_context)

                # 组合信息ImageInfo对象
                image_info = ImageInfo(image_name=image_file.name, image_path=str(image_file),
                                       image_context=image_content)

                image_info_list.append(image_info)

        return image_info_list, md_lines

    # 获取图片的上文信息
    def _find_heading_content(self, md_lines: List[str], heading_pattern, image_index: int) -> Tuple[str, List[str]]:
        # 定义找到的最近的标题索引
        head_line_index = -1
        head_title = ""

        # 考虑是否在代码块当中
        code_fence_pattern = re.compile(r"^\s*```")
        in_code_fence = False
        # 遍历最近的标题
        for i in range(image_index - 1, -1, -1):
            # 匹配标题
            line = md_lines[i]
            # 遇到代码块标记反转
            if code_fence_pattern.match(line):
                in_code_fence = not in_code_fence

            # 找到了标题
            if not in_code_fence and heading_pattern.match(line):
                head_line_index = i
                head_title = line
                break
        # 获取图片的上文
        head_content = md_lines[head_line_index + 1:image_index - 1]
        return head_title, head_content

    # 获取图片的下文信息
    def _find_post_content(self, md_lines: List[str], heading_pattern, image_index: int) -> List[str]:
        post_line_index = len(md_lines)

        # 考虑是否在代码块当中
        code_fence_pattern = re.compile(r"^\s*```")
        in_code_fence = False
        # 遍历查找下文的标题
        for i in range(image_index+1, post_line_index):
            line = md_lines[i]
            # 如果在代码块当中进行反转
            if code_fence_pattern.match(line):
                in_code_fence = not in_code_fence

            if not in_code_fence and heading_pattern.match(line):
                post_line_index = i
                break

        post_content = md_lines[image_index+1:post_line_index+1]
        return post_content


    # 按段落优雅地截取上下文的内容
    def _extract_image_content(self, content: List[str], img_content_length: int, direction: str) -> str:
        # 匹配图片路径的正则表达式
        image_pattern = re.compile(r"!\[.*?\]\(.*?\)")
        # 定义存放当前段落内容的容器
        current_graph = []
        # 定义存储最终上文或者下文段落的容器
        final_graphs = []

        # 现将内容按照段落来划分
        for line in content:
            # 如果遇到空行或者图片，就需要划分新的段落
            if not line.strip() or image_pattern.match(line):
                # 如果当前短路容器不为空
                if current_graph:
                    final_graphs.append("\n".join(current_graph))
                    current_graph = []
            else:
                current_graph.append(line)

        if current_graph:
            final_graphs.append("\n".join(current_graph))

        # 根据段落进行截取
        extract_graphs = []
        total_length = 0

        if direction == "up":
            final_graphs.reverse()

        for graph in final_graphs:
            if total_length > img_content_length:
                break
            extract_graphs.append(graph)
            total_length += len(graph) + len("\n\n")

        if direction == "up":
            extract_graphs.reverse()

        return "\n\n".join(extract_graphs)


class _VLMSummarizer:
    """将图片信息交给大模型分析"""
    def __init__(self,logger:Logger,config:ImportConfig):
        self.logger = logger
        self.config = config

    """总结单张图片的信息"""
    def _summarize_single(self, document_name: str, image_info: ImageInfo, vlm_client: OpenAI):
        # 1. 获取图片的上下文信息
        image_context = image_info.image_context
        # 2. 将图片的上下文信息组成字符串
        final_context = f"{image_context.head_title} \n {image_context.pre_context} \n {image_context.post_context}"
        # 3. 读取图片
        image_path = image_info.image_path
        with open(image_path, "rb") as f:
            import base64
            image_data = base64.b64encode(f.read()).decode("utf-8")
        # 4. 组装大模型提示词
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                            f"背景信息：\n"
                            f"  1. 所属文档标题：\"{document_name}\"\n"
                            f"  2. 图片上下文：{final_context}\n"
                            f"请结合图片内容和上述上下文信息，"
                            f"用中文简要总结这张图片的内容，"
                            f"生成一个精准的中文标题摘要（不要包含图片二字）。"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        },
                    },
                ]
            }
        ]
        # 2. 调用大模型解析输出，拿到每一张照片的描述信息
        try:
            vlm_result = vlm_client.chat.completions.create(
                model=self.config.vl_model,
                messages=messages
            )
        except Exception as e:
            self.logger.warning(f"调用VLM模型失败，图片'{image_info.image_name}'的描述信息将无法生成: {e}")
            return "暂无摘要信息"

        # 解析的大模型的输出，拿到每一张图的描述信息
        return vlm_result.choices[0].message.content

    """遍历总结所有的图片"""
    def summarize_all(self, document_name, image_info_list: List[ImageInfo]):
        # 粗糙南湖每张照片的摘要信息，key:图片的名字，value：图片的摘要信息
        image_summarizes = {}
        # 1. 创建vlm模型对象
        try:
            vlm_client = AIClients.get_vlm_client()
        except Exception as e:
            self.logger.warning(f"无法连接到VLM模型,图片描述信息无法生成：{e}")
            # 兜底方法：将图片描述信息设置为：暂无摘要信息
            for image_info in image_info_list:
                image_summarizes[image_info.image_name] = "暂无摘要信息"
            return image_summarizes

        # 遍历每一张照片生成摘要信息
        for image_info in image_info_list:
            # 调用vlm模型获取图片的摘要信息
            image_summarize = self._summarize_single(document_name, image_info, vlm_client)
            image_summarizes[image_info.image_name] = image_summarize
        return image_summarizes


class _ImageUploader:
    """将图片上传掉Minio服务器"""
    def __init__(self, logger: Logger, config: ImportConfig):
       self.logger = logger
       self.config = config

    def upload_and_replace(self, md_lines, document_name: str, image_info_list: List[ImageInfo],image_summarizes: Dict[str, str]) -> str:
        # 1. 上传图片
        remote_urls = self._upload_all(document_name, image_info_list)
        print(remote_urls)

        # 2. 替换md中的内容，返回替换后的md内容
        new_md_content = self._replace_md_content(md_lines, image_summarizes, remote_urls)

        return new_md_content

    def _upload_all(self, document_name: str, image_info_list: List[ImageInfo]) -> Dict[str, str]:
        """上传所有的图片，字典的形式存储，返回的是上传的地址列表"""
        # 存放每张图片在MinIO服务器上的访问路径,key:图片的名字，value:上传地址
        remote_urls = {}
        # 创建Minio对象
        from knowledge.utils.client.storage_clients import StorageClients
        try:
            minio_client = StorageClients.get_minio()
        except Exception as e:
            self.logger.warning(f"无法连接到MinIO服务器，图片将无法上传: {e}")
            # 兜底操作
            for image_info in image_info_list:
                remote_urls[image_info.image_name] = image_info.image_path
            return remote_urls

        # 2. 调用minio_client的方法进行文件的上传操作
        for image_info in image_info_list:
            # 构建object_name：将文件上传到minio的存储路径
            object_name = document_name + "/" + image_info.image_name
            buket_name = self.config.minio_bucket
            try:
                minio_client.fput_object(
                    bucket_name=buket_name,
                    object_name=object_name,
                    file_path=image_info.image_path
                )
            except Exception as e:
                # 兜底图片的原路径
                remote_urls[image_info.image_name] = image_info.image_path
                continue

            # 组装文件在Minio上的访问url
            remote_url = self.config.get_minio_base_url() + "/" +buket_name + "/" + object_name
            remote_urls[image_info.image_name] = remote_url
        return remote_urls


    def _replace_md_content(self, md_lines: List[str], image_summarizes: Dict[str, str],remote_urls: Dict[str, str]) -> str:
        """markdown中图片原始的描述信息替换成vlm总结的信息和地址替换成Minio中的地址"""
        # 匹配图片
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
        in_code_fence = False
        code_fence_pattern = re.compile(r"^\s*```")

        # 声明一个容器存储替换后的md_lines
        new_md_lines = []
        for line in md_lines:
            if code_fence_pattern.match(line):
                in_code_fence = not in_code_fence
            # 如果不在代码块当中并且是图片就需要替换
            match = pattern.match(line)
            if not in_code_fence and match:
                image_name = Path(match[2]).name
                new_md_lines.append(f"![{image_summarizes[image_name]}]({remote_urls[image_name]})")
            else:
                new_md_lines.append(line)

        return "\n".join(new_md_lines)



class MdImageNode(BaseNode):
    """markdown中的图片处理节点"""
    name = "md_image_node"

    def __init__(self):
        super().__init__()
        self.md_file_handler = _MdFileHandler(logger=self.logger, node_name=self.name)
        self.image_scanner = _ImageScanner(self.config)
        self.vlmSummarizer =  _VLMSummarizer(self.logger,self.config)
        self.image_uploader = _ImageUploader(logger=self.logger, config=self.config)

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 读取md文件的内容到内存当中
        md_path = state.get("md_path", "")
        md_content, md_path_obj, image_dir_obj = self.md_file_handler.read_md(md_path)
        # print(f"md_content:{md_content}")
        # print(f"md_path_obj:{md_path_obj}")
        # print(f"image_dir_obj:{image_dir_obj}")

        # 2. 遍历目录中的每一张图片，与上面的md_lines中的进行匹配
        image_info_list, md_lines = self.image_scanner.scan_img_dir(image_dir_obj,md_content)
        # for image_info in image_info_list:
        #     print(image_info)
        # print(md_lines)

        # 3. 根据图片的上下文，使用vlm模型识别图片、获取图片的描述信息

        image_summarizes = self.vlmSummarizer.summarize_all(md_path_obj.stem,image_info_list)
        for key, value in image_summarizes.items():
            print(f"{key}: {value}")

        # 4. 将图片上传到Minio服务器，将图片的描述信息和服务器地址回填到md文档中
        new_md_content = self.image_uploader.upload_and_replace(md_lines=md_lines, document_name=md_path_obj.stem,image_info_list=image_info_list,image_summarizes=image_summarizes)

        # 5. 将回填的内容，备份成新的md文件
        self.md_file_handler.backup_md(new_md_content,md_path)
        state["md_content"] = new_md_content

        return state


if __name__ == "__main__":
    setup_logging(level=INFO)
    node = MdImageNode()

    state = {
        "md_path": r"D:\AtGuiGu_Java+AI\code\python_workspace\zhiku-shopkeeper\knowledge\processor\import_processor\output_dir\万用表RS-12的使用\万用表RS-12的使用.md"
    }

    # 自动调用process方法
    node(state)
