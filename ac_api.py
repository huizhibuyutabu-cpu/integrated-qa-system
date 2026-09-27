import sys
import os

# ★ 把 rag_qa/core 加进 sys.path
_core_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_qa", "core")
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from starlette.responses import PlainTextResponse

from rag_qa.core.vector_store import VectorStore
from base import nacos_config, logger, config

# 创建 FastAPI 实例
app = FastAPI(
    title="Agent Center RAG Server",
    description="黑马程序员智能体中心ARG服务"
)


# 异常处理
def system_exception_handler(req: Request, exc: Exception):
    """
    全局异常处理函数，将异常转换为 500 响应
    """
    return PlainTextResponse(
        content=str(exc),
        status_code=500
    )


# 添加全局异常处理器
app.add_exception_handler(Exception, system_exception_handler)


# ========================= Nacos 注册与注销 =========================
def register_service():
    """
    注册智能体中心服务到 Nacos 服务发现
    """
    client = nacos_config.get_discovery_client()  # 获取 Nacos 注册客户端
    ip = nacos_config.get_discovery_ip()  # 服务 IP
    service_name = nacos_config.get_discovery_name()  # 服务名称
    port = int(config.AC_PORT)  # 服务端口
    group_name = nacos_config.get_discovery_group()  # 分组名称

    # 将实例注册到 Nacos
    result = client.add_naming_instance(
        service_name=service_name,
        ip=ip,
        port=port,
        group_name=group_name,
        heartbeat_interval=10  # 心跳间隔 10 秒
    )
    logger.info(f"✅ Registered {service_name} to Nacos: {result}")
    return result


def deregister_service():
    """
    注销智能体中心服务
    """
    ip = nacos_config.get_discovery_ip()
    service_name = nacos_config.get_discovery_name()
    port = int(config.AC_PORT)

    # 从 Nacos 注销实例
    result = nacos_config.get_discovery_client().remove_naming_instance(
        service_name, ip, port
    )
    logger.info(f"🧹 Deregistered {service_name} from Nacos")
    return result

# ========================= 启动事件 =========================
async def startup():
    # 注册服务
    register_service()

# ========================= 关闭事件 =========================
async def shutdown():
    # 注销服务
    deregister_service()

# ========================= 事件注册 =========================
# 启动事件：初始化数据库、Agents、注册服务
app.add_event_handler("startup", startup)
# 关闭事件：关闭资源、注销服务
app.add_event_handler("shutdown", shutdown)

# 创建向量数据库实例
vector_store = VectorStore()


# ========================= 请求体模型 =========================
class QueryRequest(BaseModel):
    query_text: str  # 查询的字符串
    k: int = 5  # 检索数量
    m: int = 2  # 候选数量


@app.post("/query")
def query(req: QueryRequest):
    results = vector_store.hybrid_search_with_rerank(req.query_text, k=req.k, source_filter='tj')
    results = results[:req.m]  # ★ 新增：按请求里的 m 截断
    data = [{"content": result.page_content} for result in results]
    return {"data": data}  # 返回成功状态


def start():
    # 创建 uvicorn 异步服务器实例
    uvicorn.run(app,  # FastAPI 应用
                host=config.AC_HOST,  # 绑定主机
                port=config.AC_PORT,  # 绑定端口
                log_level="info",  # 日志等级
                access_log=False  # 关闭访问日志
                )


if __name__ == "__main__":
    # 项目的主入口，当以脚本方式运行时，启动所有 App 服务
    start()