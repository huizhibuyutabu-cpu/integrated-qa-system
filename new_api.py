# new_api.py
# FastAPI 接口，基于 new_main.py 提供 RESTful API，支持会话管理、历史记录、流式输出等全部功能。

import sys
import os
import time
import logging
import uuid
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- 导入路径修复 ----------
current_dir = os.path.dirname(__file__)
core_dir = os.path.join(current_dir, 'rag_qa', 'core')
project_root = current_dir
base_dir = os.path.join(project_root, 'base')
for path in [core_dir, project_root, base_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# 导入增强版问答系统
from new_main import IntegratedQASystem

# ---------- 全局实例 ----------
qa_system: Optional[IntegratedQASystem] = None

# ---------- 生命周期 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_system
    logging.info("🚀 正在初始化增强版问答系统...")
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

# ---------- FastAPI 应用 ----------
app = FastAPI(
    title="增强版集成问答系统 API",
    description="支持会话管理、历史记忆、流式输出、学科过滤、清除历史等",
    version="2.0.0",
    lifespan=lifespan
)

# ---------- CORS ----------
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
    session_id: Optional[str] = None
    source_filter: Optional[str] = None
    stream: bool = False

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    processing_time: float

class SessionResponse(BaseModel):
    session_id: str

class MessageResponse(BaseModel):
    message: str

# ---------- 辅助 ----------
def get_session_id(request_session: Optional[str]) -> str:
    if request_session:
        return request_session
    return str(uuid.uuid4())

# ---------- 根路径 ----------
@app.get("/")
def read_root():
    return {
        "message": "增强版集成问答系统 API",
        "endpoints": {
            "POST /query": "提交问题，支持 session_id, source_filter",
            "POST /query/stream": "流式获取答案（真正流式）",
            "POST /session": "生成新的会话 ID",
            "DELETE /history/{session_id}": "删除指定会话的历史记录",
            "DELETE /history/all": "清空所有历史记录",
            "GET /sources": "获取支持的学科列表",
            "GET /health": "健康检查"
        }
    }

# ---------- 健康检查 ----------
@app.get("/health")
def health_check():
    if qa_system is None:
        raise HTTPException(status_code=503, detail="系统未就绪")
    return {"status": "ok"}

# ---------- 获取学科列表 ----------
@app.get("/sources")
def get_sources():
    if qa_system is None:
        raise HTTPException(status_code=503, detail="系统未就绪")
    return {"sources": qa_system.config.VALID_SOURCES}

# ---------- 生成新会话 ----------
@app.post("/session", response_model=SessionResponse)
def create_session():
    return {"session_id": str(uuid.uuid4())}

# ---------- 非流式查询（改造后：先判断是否命中 MySQL） ----------
@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if qa_system is None:
        raise HTTPException(status_code=503, detail="系统尚未初始化")

    session_id = get_session_id(request.session_id)
    start_time = time.time()
    try:
        # 先执行 BM25 搜索判断
        answer, need_rag = qa_system.bm25_search.search(request.query, threshold=0.75)
        if answer:
            # 命中 MySQL，直接返回答案并保存历史
            qa_system._save_conversation(session_id, request.query, answer, source="mysql")
            elapsed = time.time() - start_time
            return QueryResponse(
                answer=answer,
                session_id=session_id,
                processing_time=elapsed
            )
        elif need_rag:
            # 需要 RAG，返回特殊标记
            elapsed = time.time() - start_time
            return QueryResponse(
                answer="__NEED_STREAM__",
                session_id=session_id,
                processing_time=elapsed
            )
        else:
            # 未找到答案
            elapsed = time.time() - start_time
            return QueryResponse(
                answer="未找到答案",
                session_id=session_id,
                processing_time=elapsed
            )
    except Exception as e:
        logging.error(f"查询处理异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

# ---------- 流式查询（真正流式） ----------
@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    真正流式返回答案，使用 text/event-stream 格式。
    每产生一个 token 即发送，生成结束后自动保存历史。
    """
    if qa_system is None:
        raise HTTPException(status_code=503, detail="系统尚未初始化")

    session_id = get_session_id(request.session_id)

    try:
        # 调用 query_with_history(stream=True) 获得生成器
        generator = qa_system.query_with_history(
            query=request.query,
            session_id=session_id,
            source_filter=request.source_filter,
            stream=True
        )
    except Exception as e:
        logging.error(f"流式查询初始化失败: {e}")
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")

    # 定义异步生成器，将同步生成器转换为 SSE 格式
    async def event_generator():
        # 先发送 session_id 事件
        yield f"event: session\ndata: {session_id}\n\n"
        try:
            for chunk in generator:
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"
        finally:
            yield f"event: done\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ---------- 清空全部历史 ----------
@app.delete("/history/all")
def clear_all_history():
    if qa_system is None:
        raise HTTPException(status_code=503, detail="系统未就绪")
    success = qa_system.clear_all_history()
    if success:
        return {"message": "所有历史已清空"}
    else:
        raise HTTPException(status_code=500, detail="清空失败")

# ---------- 删除指定会话历史 ----------
@app.delete("/history/{session_id}")
def delete_session_history(session_id: str):
    if qa_system is None:
        raise HTTPException(status_code=503, detail="系统未就绪")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    success = qa_system.delete_conversation_history(session_id)
    if success:
        return {"message": f"会话 {session_id} 的历史已删除"}
    else:
        raise HTTPException(status_code=500, detail="删除失败")

# ---------- 获取指定会话的历史记录 ----------
@app.get("/history/{session_id}")
def get_session_history(session_id: str):
    """
    返回指定会话的所有历史消息（按时间正序）
    """
    if qa_system is None:
        raise HTTPException(status_code=503, detail="系统未就绪")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    sql = """
        SELECT user_query, answer, created_at
        FROM conversations
        WHERE session_id = %s
        ORDER BY created_at ASC
    """
    try:
        qa_system.mysql_client.connection.ping(reconnect=True)
        qa_system.mysql_client.cursor.execute(sql, (session_id,))
        rows = qa_system.mysql_client.cursor.fetchall()
        history = [
            {
                "user_query": row[0],
                "answer": row[1],
                "created_at": row[2].isoformat() if row[2] else None
            }
            for row in rows
        ]
        return {"history": history}
    except Exception as e:
        logging.error(f"获取会话历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取历史失败: {str(e)}")

# ---------- 启动 ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("new_api:app", host="0.0.0.0", port=8000, reload=True)