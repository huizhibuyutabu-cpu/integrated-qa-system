# new_main.py
# 功能：集成问答系统（增强版），整合 MySQL 精确匹配、BM25 检索、RAG 知识库检索、
#       会话管理、对话历史存储、流式输出（真正流式，且自动保存历史）、日志强制刷新等。
# 作者：EduRAG 项目组
# 版本：2.0 (mysql数据库里面通过redisee缓存，答案是非流式，剩下的问题答案是流式输出的)

import sys
import os
import re
import uuid
import time
import logging

# 获取当前脚本所在目录（即项目根目录 integrated_qa_system）
current_dir = os.path.dirname(__file__)

# 添加 rag_qa/core，使 document_processor、RAGSystem 等能被找到
core_dir = os.path.join(current_dir, 'rag_qa', 'core')
# 添加项目根目录，使 base、rag_qa 等顶层包可被识别
project_root = current_dir
# 添加 base 目录，确保 from base import ... 能正常导入
base_dir = os.path.join(project_root, 'base')

# 将以上路径插入 sys.path 最前面（优先搜索）
for path in [core_dir, project_root, base_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# 导入 MySQL 系统组件，用于数据库操作和搜索
from mysql_qa import MySQLClient, RedisClient, BM25Search
# 导入 RAG 系统组件（使用增强版，位于项目根目录）
from rag_qa import VectorStore
from new_rag_system import RAGSystem  # 增强版 RAG 系统（仅流式接口）
# 导入配置和日志工具，用于系统配置和日志记录
from base import logger, Config
# 导入 OpenAI 客户端，用于调用 DashScope API
from openai import OpenAI

# ===== 会话 ID 缓存文件 =====
SESSION_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".edurag_session")

# ===== 强制刷新日志函数 =====
def flush_all_loggers():
    """强制刷新所有日志处理器，确保日志立即输出到控制台"""
    for handler in logging.root.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    sys.stdout.flush()


class IntegratedQASystem:
    """集成问答系统主类，整合 MySQL 精确匹配、BM25 检索、RAG 检索及会话管理"""
    def __init__(self):
        self.logger = logger
        self.config = Config()
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

        # 初始化 OpenAI 客户端（用于 RAGSystem 内部调用）
        try:
            self.client = OpenAI(
                api_key=self.config.DASHSCOPE_API_KEY,
                base_url=self.config.DASHSCOPE_BASE_URL
            )
        except Exception as e:
            self.logger.error(f"OpenAI 客户端初始化失败: {e}")
            raise

        # 定义 LLM 调用函数（非流式，用于检索策略中的 LLM 调用）
        def llm_call(prompt):
            try:
                completion = self.client.chat.completions.create(
                    model=self.config.LLM_MODEL,
                    messages=[{"role": "system", "content": "你是一个有用的助手。"},
                              {"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return completion.choices[0].message.content if completion.choices else "直接检索"
            except Exception as e:
                self.logger.error(f"LLM 调用失败: {e}")
                return "直接检索"

        # 定义 LLM 流式调用函数（生成器）
        def llm_call_stream(prompt):
            try:
                stream = self.client.chat.completions.create(
                    model=self.config.LLM_MODEL,
                    messages=[{"role": "system", "content": "你是一个有用的助手。"},
                              {"role": "user", "content": prompt}],
                    temperature=0.1,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                self.logger.error(f"流式 LLM 调用失败: {e}")
                yield f"错误：流式 LLM 调用失败 - {e}"

        # 初始化向量存储，用于 RAG 系统的知识库管理
        self.vector_store = VectorStore()
        # 初始化 RAG 系统（增强版），传入非流式和流式 LLM 函数
        self.rag_system = RAGSystem(self.vector_store, llm_call, llm_stream=llm_call_stream)

        # 创建对话历史表（如果不存在）
        self._init_conversation_table()
        flush_all_loggers()

    def _init_conversation_table(self):
        """创建 conversations 表用于存储对话历史，若已存在则不重复创建"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS conversations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(64) NOT NULL,
            user_query TEXT NOT NULL,
            answer TEXT NOT NULL,
            source VARCHAR(20) DEFAULT 'mysql',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_session_id (session_id),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        try:
            self.mysql_client.connection.ping(reconnect=True)
            self.mysql_client.cursor.execute(create_sql)
            self.mysql_client.connection.commit()
            self.logger.info("对话历史表已就绪")
            flush_all_loggers()
        except Exception as e:
            self.logger.error(f"创建对话历史表失败: {e}")
            flush_all_loggers()

    def _get_conversation_history(self, session_id, limit=5):
        """获取指定 session_id 的最近 limit 轮对话历史（按时间正序）"""
        if not session_id:
            return []
        sql = """
            SELECT user_query, answer
            FROM conversations
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """
        try:
            self.mysql_client.connection.ping(reconnect=True)
            self.mysql_client.cursor.execute(sql, (session_id, limit))
            rows = self.mysql_client.cursor.fetchall()
            history = [(row[0], row[1]) for row in reversed(rows)]
            return history
        except Exception as e:
            self.logger.error(f"获取对话历史失败: {e}")
            return []

    def _save_conversation(self, session_id, user_query, answer, source):
        """保存一轮对话到数据库"""
        if not session_id:
            return
        sql = """
            INSERT INTO conversations (session_id, user_query, answer, source)
            VALUES (%s, %s, %s, %s)
        """
        try:
            self.mysql_client.connection.ping(reconnect=True)
            self.mysql_client.cursor.execute(sql, (session_id, user_query, answer, source))
            self.mysql_client.connection.commit()
        except Exception as e:
            self.logger.error(f"保存对话历史失败: {e}")

    def delete_conversation_history(self, session_id):
        """删除指定会话 ID 的所有历史记录"""
        if not session_id:
            return False
        sql = "DELETE FROM conversations WHERE session_id = %s"
        try:
            self.mysql_client.connection.ping(reconnect=True)
            self.mysql_client.cursor.execute(sql, (session_id,))
            self.mysql_client.connection.commit()
            affected = self.mysql_client.cursor.rowcount
            self.logger.info(f"已删除会话 {session_id} 的 {affected} 条历史记录")
            return True
        except Exception as e:
            self.logger.error(f"删除会话历史失败: {e}")
            return False

    def clear_all_history(self):
        """清空所有对话历史（谨慎使用）"""
        sql = "DELETE FROM conversations"
        try:
            self.mysql_client.connection.ping(reconnect=True)
            self.mysql_client.cursor.execute(sql)
            self.mysql_client.connection.commit()
            affected = self.mysql_client.cursor.rowcount
            self.logger.warning(f"已清空所有历史记录，共 {affected} 条")
            return True
        except Exception as e:
            self.logger.error(f"清空历史失败: {e}")
            return False

    def query_with_history(self, query, session_id=None, source_filter=None, stream=False):
        """
        增强查询方法，支持会话管理和历史记忆
        Args:
            query (str): 用户问题
            session_id (str, optional): 会话 ID，若 None 则自动生成
            source_filter (str, optional): 学科过滤
            stream (bool): 是否启用流式输出
        Returns:
            str or generator: 若 stream=False 返回完整字符串；若 stream=True 返回生成器
        """
        # 1. 会话 ID 处理
        if session_id is None:
            session_id = str(uuid.uuid4())
            self.logger.info(f"生成新会话 ID: {session_id}")

        # 2. 获取历史对话（最近5轮）
        history = self._get_conversation_history(session_id, limit=5)
        self.logger.info(f"获取到 {len(history)} 轮历史对话")

        # 3. 执行 BM25 搜索（阈值 0.75）
        self.logger.info(f"处理查询: '{query}'")
        answer, need_rag = self.bm25_search.search(query, threshold=0.75)
        sys.stdout.flush()

        # ===== 关键修改：MySQL 命中时直接返回字符串（忽略 stream 参数） =====
        if answer:
            self.logger.info(f"MySQL 答案: {answer}")
            self._save_conversation(session_id, query, answer, source="mysql")
            flush_all_loggers()
            return answer  # 总是返回字符串，不流式

        elif need_rag:
            self.logger.info("无可靠 MySQL 答案，回退到 RAG")
            rag_generator = self.rag_system.generate_answer_stream(query, source_filter)

            if stream:
                # ===== 流式分支：包装生成器，自动收集完整答案并保存历史 =====
                def wrapped_generator():
                    full_answer = ""
                    for chunk in rag_generator:
                        full_answer += chunk
                        yield chunk
                    # 生成结束后保存历史
                    self._save_conversation(session_id, query, full_answer, source="rag")
                    flush_all_loggers()
                return wrapped_generator()
            else:
                # 非流式：直接收集所有块，保存历史并返回完整字符串
                full_answer = ""
                for chunk in rag_generator:
                    full_answer += chunk
                self._save_conversation(session_id, query, full_answer, source="rag")
                flush_all_loggers()
                return full_answer
        else:
            self.logger.info("未找到答案")
            flush_all_loggers()
            self._save_conversation(session_id, query, "未找到答案", source="unknown")
            return "未找到答案"


def main():
    """主函数：交互式命令行界面（保留原有功能）"""
    qa_system = IntegratedQASystem()
    flush_all_loggers()

    try:
        print("\n" + "=" * 60)
        print("欢迎使用集成问答系统（增强版）！")
        print(f"支持的来源: {qa_system.config.VALID_SOURCES}")
        print("输入查询进行问答，输入 'exit' 退出。")
        print("输入 'clear' 清除会话历史，'clear all' 清空全部历史。")
        print("=" * 60)

        # 会话 ID 管理（与之前相同）
        last_session = None
        if os.path.exists(SESSION_CACHE_FILE):
            with open(SESSION_CACHE_FILE, "r") as f:
                last_session = f.read().strip()
                if last_session:
                    print(f"📌 上次使用的会话 ID: {last_session}")

        prompt = "请输入会话 ID（直接 Enter 将创建新会话，输入已有 ID 可延续历史）: "
        session_input = input(prompt).strip()
        if session_input:
            session_id = session_input
            with open(SESSION_CACHE_FILE, "w") as f:
                f.write(session_id)
            print(f"✅ 使用会话 ID: {session_id}")
        else:
            session_id = str(uuid.uuid4())
            with open(SESSION_CACHE_FILE, "w") as f:
                f.write(session_id)
            print(f"🆕 创建新会话 ID: {session_id}")

        current_session_id = session_id

        while True:
            flush_all_loggers()
            time.sleep(0.01)

            print("\n" + "-" * 50)
            user_input = input("输入查询（或命令）: ").strip()
            query = user_input

            # 处理清除命令
            if user_input.lower() in ["clear", "delete"]:
                print("\n--- 清除会话历史 ---")
                target = input("请输入要清除的会话 ID（输入 'all' 清空全部历史）: ").strip()
                if not target:
                    print("❌ 未输入任何内容，操作取消。")
                    continue
                if target.lower() == "all":
                    confirm = input("⚠️ 确认清空所有历史记录？(y/n): ").strip().lower()
                    if confirm == 'y':
                        if qa_system.clear_all_history():
                            print("✅ 所有历史已清空。")
                        else:
                            print("❌ 清空失败，请查看日志。")
                    else:
                        print("操作已取消。")
                else:
                    confirm = input(f"确认删除会话 '{target}' 的所有历史？(y/n): ").strip().lower()
                    if confirm == 'y':
                        if qa_system.delete_conversation_history(target):
                            print(f"✅ 会话 {target} 的历史已删除。")
                        else:
                            print("❌ 删除失败，请查看日志。")
                    else:
                        print("操作已取消。")
                continue

            # 拦截日志内容等（与原版相同）
            if "EduRAG" in query or re.search(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}', query):
                print("⚠️ 检测到日志内容，请重新输入您的问题。")
                continue

            if not query:
                print("❌ 查询内容不能为空，请重新输入。")
                continue

            if query.lower() == "exit":
                logger.info("退出系统")
                print("再见！")
                break

            # 学科过滤
            source_filter = input(
                f"输入来源过滤 ({'/'.join(qa_system.config.VALID_SOURCES)}) (按 Enter 跳过): "
            ).strip()
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                logger.warning(f"无效来源 '{source_filter}'，忽略过滤")
                print(f"无效来源 '{source_filter}'，继续无过滤。")
                source_filter = None

            # 流式选项
            stream_choice = input("是否启用流式输出？(y/n, 默认 n): ").strip().lower()
            stream = stream_choice == 'y'

            # 执行查询
            start_time = time.time()
            print("\n正在生成答案...")

            result = qa_system.query_with_history(
                query=query,
                session_id=current_session_id,
                source_filter=source_filter,
                stream=stream
            )

            flush_all_loggers()
            sys.stdout.flush()

            if stream and hasattr(result, '__iter__') and not isinstance(result, str):
                # 流式输出
                full_answer = ""
                first_chunk = None
                try:
                    first_chunk = next(result)
                except StopIteration:
                    pass

                print("答案: ", end="", flush=True)
                if first_chunk:
                    print(first_chunk, end="", flush=True)
                    full_answer += first_chunk

                for chunk in result:
                    print(chunk, end="", flush=True)
                    full_answer += chunk
                print()
                elapsed = time.time() - start_time
                print(f"✅ 生成耗时: {elapsed:.2f} 秒")
            else:
                print(f"\n答案: {result}")
                elapsed = time.time() - start_time
                print(f"✅ 查询耗时: {elapsed:.2f} 秒")

            sys.stdout.flush()

    except Exception as e:
        logger.error(f"系统错误: {e}")
        print(f"发生错误: {e}")
    finally:
        qa_system.mysql_client.close()


if __name__ == "__main__":
    main()