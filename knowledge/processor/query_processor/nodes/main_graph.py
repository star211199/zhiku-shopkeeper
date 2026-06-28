from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph, StateGraph

from knowledge.processor.query_processor.state import QueryGraphState

# 加载环境变量
load_dotenv(encoding="utf-8")

def create_query_graph()->CompiledStateGraph:
    """创建查询流程图。

        Returns:
            编译后的 StateGraph 实例。

        流程结构::

            item_name_confirm
                  │
                  ├── (有答案) ──────────────────────────> answer_output
                  │                                            │
                  └── (无答案)                                  │
                       │                                       │
                       v                                       │
                  multi_search                                 │
                       │                                       │
                 ┌─────┼──────────┐                            │
                 │     │          │                            │
                 v     v          v                            │
            embedding  hyde    web_mcp                         │
                 │     │          │                            │
                 └─────┼──────────┘                            │
                       │                                       │
                       v                                       │
                     join                                      │
                       │                                       │
                       v                                       │
                      rrf                                      │
                       │                                       │
                       v                                       │
                    rerank                                     │
                       │                                       │
                       v                                       │
                 answer_output <───────────────────────────────┘
                       │
                       v
                      END
        """
    # 1. 定义LangGraph工作流
    workflow = StateGraph(QueryGraphState)  # type:ignore
    # 2. 实例化节点
    nodes={
        "item_name_confirm_node": ItemNameConfirmedNode(),

    }
    # 3. 添加节点

    # 4. 定义入口节点

    # 5. 添加条件边

    # 6. 添加业务边

