import logging
import os
import shutil
import time
import uuid
from datetime import datetime
from fastapi import UploadFile
from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_processor.exceptions import FileProcessingError
from knowledge.processor.import_processor.main_graph import import_app
from knowledge.utils.client.base import logger
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import add_running_task, update_task_status, TASK_STATUS_PROCESSING, add_done_task, \
    add_node_duration, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED

logger = logging.getLogger(__name__)


class FileProcessService:
    # 执行工作流
    def run_main_graph(self, import_file_path: str, file_dir: str, task_id: str):
        logging.basicConfig(level=logging.INFO)
        # 定义state
        state = {
            "task_id": task_id,
            "import_file_path": import_file_path,
            "file_dir": file_dir
        }
        try:
            for event in import_app.stream(state):
                for node_name, state in event.items():
                    print(f"{node_name}正在执行...")
            # 代表整个任务执行完成
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(e)
            # 代表任务失败
            update_task_status(task_id, TASK_STATUS_FAILED)
            return

    def get_base_dir(self):
        local_base_dir = get_local_base_dir()
        return os.path.join(local_base_dir, datetime.now().strftime('%Y%m%d'))

    def process_upload_file(self, file: UploadFile):
        # 生成task_id
        task_id = uuid.uuid4().hex[:8]

        # 放到任务列表当中
        add_running_task(task_id, "upload_file")

        # 整个任务开始，应该设置整个任务的状态RUNNING
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        # 计算耗时
        start_time = time.time()
        # 2. 将文件上传保存到本地临时文件当中
        try:
            import_file_path, file_dir = self._save_file_to_local(file)
        except Exception as e:
            logger.error(e)
            update_task_status(task_id, TASK_STATUS_FAILED)
            raise

        # 3. 将文件保存到Minio（做个备份，不成功也没关系）
        self._save_file_to_remote(import_file_path, file.filename)

        end_time = time.time()
        add_done_task(task_id, "upload_file")
        add_node_duration(task_id, "upload_file", duration=end_time - start_time)

        # 返回import_file_path、以及 file_dir
        return import_file_path, file_dir, task_id

    # 将上传的文件保存到本地的目录当中
    def _save_file_to_local(self, upload_file: UploadFile):
        # 获取本地临时文件目录
        base_dir = self.get_base_dir()

        # 判断base_dir是否存在，如果不存在则创建
        os.makedirs(base_dir, exist_ok=True)

        # 构建import_file_path
        import_file_path = os.path.join(base_dir, upload_file.filename)

        try:
            with open(import_file_path, 'wb') as f:
                shutil.copyfileobj(upload_file.file, f)
        except Exception as e:
            logger.error(f"文件{upload_file.filename}保存到本地临时目录失败,{str(e)}")
            raise FileProcessingError(message=f"文件{upload_file.filename}保存到本地临时目录失败,{str(e)}")

        return import_file_path, base_dir

    # 将文件备份到远程的Minio服务器
    def _save_file_to_remote(self, import_file_path: str, filename: str):
        # 1.创建Minio客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            logger.warning(f"获取minio客户端失败,但不影响主流程,{str(e)}")
            return

        # 2. 保存文件
        # 通的名称
        bucket_name = os.getenv("MINIO_BUCKET_NAME")
        # 上传到Minio的路径
        object_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{filename}"
        try:
            # import_file_path:文件上传的路径
            minio_client.fput_object(bucket_name, object_name, import_file_path)
        except Exception as e:
            logger.warning(f"minio备份文件失败,但不影响主流程,{str(e)}")
            return
