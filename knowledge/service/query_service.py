import uuid
from mcp.server.fastmcp.utilities import logging
from knowledge.processor.query_processor.main_graph import query_app
from knowledge.processor.query_processor.state import create_default_state
from knowledge.utils.mongo_history_util import get_recent_messages, clear_history
from knowledge.utils.task_util import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED, \
    TASK_STATUS_COMPLETED, get_task_result

logger = logging.get_logger(__name__)
class QueryService:
    def get_query_result(self,task_id: str):
        return get_task_result(task_id=task_id,key="answer")

    def generate_session_id(self):
        return uuid.uuid4().hex

    def generate_task_id(self):
        return uuid.uuid4().hex[:12]

    def run_query_graph(self, query: str, session_id: str, task_id: str, is_stream: bool):
        # 执行整个检索流程
        # 1. 创建State
        query_graph_state = create_default_state(
            original_query=query,
            session_id=str(session_id),
            task_id=task_id,
            is_stream=is_stream,
        )
        try:
            # 2. 调用LangGraph执行
            # 设置整个检索流程的状态为正在进行
            update_task_status(task_id=task_id,status_name=TASK_STATUS_PROCESSING)
            query_app.invoke(query_graph_state)
            # 设置整个检索流程状态为已完成
            update_task_status(task_id=task_id,status_name=TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(f"运行查询流程出现异常:{str(e)}")
            # 设置整个检索流程的状态为失败
            update_task_status(task_id=task_id, status_name=TASK_STATUS_FAILED)
            raise e

    def get_history(self, session_id:str):
        history_list = get_recent_messages(session_id)

        # 按照ts从小到大排序
        sorted_history_list = sorted(history_list, key=lambda x: x['ts'],reverse=False)
        return [
            {
                "_id": str(history.get("_id", "")),
                "session_id": history.get("session_id") or "",
                "role": history.get("role") or "",
                "text": history.get("text") or "",
                "rewritten_query": history.get("rewritten_query") or "",
                "item_names": history.get("item_names") or [],
                "ts": history.get("ts"),
            }for history in sorted_history_list
        ]

    def clean_history(self, session_id:str):
        clear_history(session_id)



if __name__ == '__main__':
    print(uuid.uuid4().hex)