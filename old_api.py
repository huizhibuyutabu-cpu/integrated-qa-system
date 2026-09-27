import sys
import os
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

# ---------- 修复导入路径（与 old_main.py 保持一致） ----------
current_dir = os.path.dirname(__file__)
core_dir = os.path.join(current_dir, 'rag_qa', 'core')
project_root = current_dir
base_dir = os.path.join(project_root, 'base')
for path in [core_dir, project_root, base_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# 从原有脚本导入问答系统核心类
from old_main import IntegratedQASystem

# ---------- FastAPI 相关导入 ----------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- 全局系统实例 ----------
qa_system: Optional[IntegratedQASystem] = None

# ===== 新增：问候语配置（直接放在 API 层） =====
GREETING_KEYWORDS = [
    "你好", "您好", "嗨", "hello", "hi", "hey",
    "你是谁", "你是什么", "介绍一下", "你能做什么", "功能", "介绍自己"
]
GREETING_RESPONSE_TEMPLATE = (
    "您好！我是智能知识问答助手，专注于为您解答关于自然语言处理、AI、机器学习等专业问题。\n"
    "我的知识库包含了丰富的课程资料、学术论文和常见问题解答。\n"
    "您可以直接提问，例如：'什么是Transformer？'、'如何微调BERT？'，我会基于知识库给出专业回答。\n"
    "如需人工帮助，请联系客服：{}"
)

# ---------- Lifespan 事件处理器 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_system
    logging.info("🚀 正在初始化问答系统...")
    try:
        qa_system = IntegratedQASystem()
        logging.info("✅ 问答系统初始化成功")
    except Exception as e:
        logging.error(f"❌ 系统初始化失败: {e}")
        raise RuntimeError("无法初始化问答系统")
    yield
    if qa_system:
        try:
            qa_system.mysql_client.close()
            logging.info("🛑 MySQL 连接已关闭")
        except Exception as e:
            logging.warning(f"关闭 MySQL 连接时出错: {e}")

# ---------- 创建 FastAPI 应用 ----------
app = FastAPI(
    title="集成问答系统 API",
    description="基于 MySQL + BM25 + RAG 的智能问答接口",
    version="1.0.0",
    lifespan=lifespan
)

# ---------- 跨域配置 ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 请求/响应模型 ----------
class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    processing_time: float

# ---------- 根路径 ----------
@app.get("/")
def read_root():
    return {
        "message": "欢迎使用集成问答系统 API",
        "endpoints": {
            "POST /query": "提交问题，获取答案"
        }
    }

# ---------- 健康检查 ----------
@app.get("/health")
def health_check():
    if qa_system is None:
        raise HTTPException(status_code=503, detail="系统未就绪")
    return {"status": "ok"}

# ---------- 核心查询接口（已添加问候语拦截） ----------
@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    # 首先检查系统是否就绪
    if qa_system is None:
        raise HTTPException(status_code=503, detail="系统尚未初始化")

    start_time = time.time()

    # ===== 新增：问候语拦截（在调用业务逻辑之前） =====
    # 检查查询是否匹配问候语关键词
    is_greeting = any(kw in request.query for kw in GREETING_KEYWORDS)
    if is_greeting:
        # 构造欢迎回复（使用配置中的客服电话）
        try:
            phone = qa_system.config.CUSTOMER_SERVICE_PHONE
        except AttributeError:
            phone = "400-000-0000"  # 回退默认
        answer = GREETING_RESPONSE_TEMPLATE.format(phone)
        elapsed = time.time() - start_time
        logging.info(f"检测到问候语，返回知识库介绍（耗时 {elapsed:.2f}s）")
        return QueryResponse(answer=answer, processing_time=elapsed)

    # 非问候语，正常调用业务逻辑
    try:
        answer = qa_system.query(
            query=request.query,
            source_filter=request.source_filter
        )
        elapsed = time.time() - start_time
        return QueryResponse(answer=answer, processing_time=elapsed)
    except Exception as e:
        logging.error(f"查询处理异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.get("/sources")
def get_sources():
    if qa_system is None:
        raise HTTPException(status_code=503, detail="系统未就绪")
    return {"sources": qa_system.config.VALID_SOURCES}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("old_api:app", host="0.0.0.0", port=8000, reload=True)