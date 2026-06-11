from pathlib import Path
from knowledge.processor.import_processor import base
from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_processor.state import ImportGraphState


class EntryNode(BaseNode):
    name="entry_node"

    # 这是父类中定义的抽象方法，子类必须实现
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 获取state中的import_file_path/file_dir，并判断是否为空
        self.log_step(step_name="Step1", message="判断路径是否为空")
        import_file_path = state.get("import_file_path")
        file_dir = state.get("file_dir")
        # 判断值是否为空，如果为空直接抛出异常
        if not import_file_path:
            self.logger.error(f"import_file_path not found in state {state}")
            raise StateFieldError(node_name=self.name,field_name="import_file_path",expected_type=str)
        if not file_dir:
            self.logger.error(f"file_dir not found in state {state}")
            raise StateFieldError(node_name=self.name,field_name="file_dir",expected_type=str)

        # 2. 判断在实际的路径中是否真实存在import_file_path/file_dir
        self.log_step(step_name="Step2", message="判断路径是否真实存在")
        import_file_path_obj = Path(import_file_path)
        file_dir_obj = Path(file_dir)
        # 导入文件的路径不是真实存在的
        if not import_file_path_obj.exists():
            self.logger.error(f"import_file_path not exist")
            raise StateFieldError(node_name=self.name,field_name="import_file_path",expected_type=str)

        # 导入（导出）的文件目录不是真实存在的
        if not file_dir_obj.exists():
            self.logger.error(f"file_dir not exist")
            raise StateFieldError(node_name=self.name,field_name="file_dir",expected_type=str)

        # 3. 判断文件类型
        # 3.1 获取文件的后缀
        self.log_step(step_name="Step3", message="识别文件的后缀类型")
        suffix = import_file_path_obj.suffix
        # 3.2 判断文件的后缀
        # 3.2.1 如果是pdf文件,则设置is_pdf_enable为True,设置pdf_path的值为import_file_path
        if suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        # 3.2.2 如果是md文件,则设置is_md_enable为True, 设置md_path的值为import_file_path
        elif suffix == ".pdf":
            state["is_md_read_enabled"] = False
            state["pdf_path"] = import_file_path
        else:
            # 抛异常
            self.logger.error(f"unsupported suffix {suffix}")
            raise ValidationError(message="unsupported suffix",node_name=self.name)

        # 4. 获取文件标题，并将文件标题设置到state中
        self.log_step(step_name="Step4", message="获取文件标题并设置state中的状态")
        file_title = import_file_path_obj.stem
        state["file_title"] = file_title
        return state

# 测试节点
if __name__=="__main__":
    base.setup_logging(level="DEBUG")
    entry_node = EntryNode()

    init_state = {
        "import_file_path": r"../input_dir/万用表RS-12的使用.pdf",
        "file_dir": r"../output_dir",
    }

    state = entry_node(init_state)

    print(state)