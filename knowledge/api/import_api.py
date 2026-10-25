import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from knowledge.core.deps import get_file_process_service
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.core.paths import get_front_page_dir
from fastapi import BackgroundTasks
from knowledge.service.file_process_service import FileProcessService
from fastapi import Depends
from knowledge.utils.task_util import get_task_info


def register_router(app: FastAPI):
    """创建路由"""

    @app.get("/hello")
    def hello_world():
        return "hello world"

    @app.post("/upload", response_model=UploadResponse)
    def upload_file(file: UploadFile,
                    background_tasks: BackgroundTasks,
                    file_process_service: FileProcessService = Depends(get_file_process_service)):  # 相当于Java中的依赖注入，不用频繁的地创对象
        # 将上传的文件进行处理(保存)
        import_file_path, file_dir, task_id = file_process_service.process_upload_file(file)
        # print(import_file_path)
        # print(file_dir)
        # print(task_id)

        # 执行导入的主流程
        background_tasks.add_task(file_process_service.run_main_graph,import_file_path,file_dir,task_id)
        return UploadResponse(message="文件上传成功", task_id=task_id)

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    def get_task_status(task_id: str):
        # 获取当前任务的信息
        task_info = get_task_info(task_id)
        return TaskStatusResponse(**task_info)

def create_app():
    # 1. 创建FastAPI
    app = FastAPI(
        description="掌柜智库导入流程API",
        version="1.0"
    )

    # 2. 处理跨域请求问题，这个项目可以不用考虑，允许哪些请求可以访问这个后端
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 3. 挂载静态文件，也就是指定静态文件的访问路径
    # 如果前端的请求路径是 /front/import.html 那么后端就 "挂载的目录" 找import.html
    front_dir = get_front_page_dir()
    from starlette.staticfiles import StaticFiles
    app.mount("/front", StaticFiles(directory=front_dir))

    # 4. 注册路由
    register_router(app)

    return app


if __name__ == "__main__":
    # 运行FastAPI服务
    app = create_app()

    uvicorn.run(app, host="0.0.0.0", port=8000)
