import sys
import os
import re
# 获取当前脚本所在目录（即项目根目录 integrated_qa_system）
current_dir = os.path.dirname(__file__)

# 添加 rag_qa/core，使 document_processor 能被找到
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
# 导入 RAG 系统组件，用于知识库检索和答案生成
from rag_qa import VectorStore, RAGSystem
# 导入配置和日志工具，用于系统配置和日志记录
from base import logger, Config
# 导入 OpenAI 客户端，用于调用 DashScope API
from openai import OpenAI
# 导入时间库，用于记录处理时间
import time
# ===== 新增：导入 logging 模块，用于操作日志处理器 =====
# 作用：通过 logging.root.handlers 可以获取所有已注册的日志处理器，
#       从而能够强制刷新日志缓冲区，确保日志及时输出到控制台。
import logging
# ===== 新增：强制刷新所有日志处理器的函数 =====
# 作用：解决日志延迟输出导致与 input() 提示混行的问题。
#       遍历所有日志处理器，调用 flush() 将缓冲区内容强制写入目标流，
#       最后再刷新 sys.stdout 确保数据真正落屏。
def flush_all_loggers():
    """强制刷新所有日志处理器，确保日志立即输出到控制台"""
    for handler in logging.root.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    sys.stdout.flush()

class IntegratedQASystem:
    def __init__(self):
        # 初始化日志工具，用于记录系统运行信息
        self.logger = logger
        # 初始化配置对象，加载系统参数
        self.config = Config()
        # 初始化 MySQL 客户端，用于数据库操作
        self.mysql_client = MySQLClient()
        # 初始化 Redis 客户端，用于缓存管理
        self.redis_client = RedisClient()
        # 初始化 BM25 搜索模块，结合 MySQL 和 Redis
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
        try:
            # 初始化 OpenAI 客户端，连接 DashScope API
            self.client = OpenAI(api_key=self.config.DASHSCOPE_API_KEY, base_url=self.config.DASHSCOPE_BASE_URL)
        except Exception as e:
            # 记录 OpenAI 初始化失败的错误日志
            self.logger.error(f"OpenAI 客户端初始化失败: {e}")
            # 抛出异常，终止初始化
            raise
        # 初始化向量存储，用于 RAG 系统的知识库管理
        self.vector_store = VectorStore()
        # 初始化 RAG 系统，传入向量存储和 DashScope API 调用函数
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)



    def call_dashscope(self, prompt):
        # 定义调用 DashScope API 的方法，生成自然语言答案
        try:
            # 创建聊天完成请求，调用 DashScope API
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,  # 使用配置中的语言模型
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},  # 系统提示
                    {"role": "user", "content": prompt},  # 用户输入的提示
                ]
            )
            # 检查响应是否有效，返回答案内容
            return completion.choices[0].message.content if completion.choices else "错误：无效的 LLM 响应"
        except Exception as e:
            # 记录 API 调用失败的错误日志
            self.logger.error(f"LLM 调用失败: {e}")
            # 返回错误信息，便于调试
            return f"错误：LLM 调用失败 - {e}"

    def query(self, query, source_filter=None):
        # 定义查询方法，处理用户输入的查询
        start_time = time.time()  # 记录查询开始时间
        # 记录查询信息到日志
        self.logger.info(f"处理查询: '{query}'")
        # 执行 BM25 搜索，获取答案和是否需要 RAG 的标志
        answer, need_rag = self.bm25_search.search(query, threshold=0.85)
        if answer:
            # 如果找到可靠答案，记录答案到日志
            self.logger.info(f"MySQL 答案: {answer}")
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # ===== 新增：返回答案前强制刷新日志 =====
            # 作用：确保所有日志（包括 MySQL 答案和处理耗时）
            #       在 return 之前完整输出到控制台，避免延迟。
            flush_all_loggers()

            # 返回 MySQL 答案
            return answer
        elif need_rag:
            # 如果需要 RAG，记录回退信息到日志
            self.logger.info("无可靠 MySQL 答案，回退到 RAG")
            # 调用 RAG 系统生成答案，支持学科过滤
            answer = self.rag_system.generate_answer(query, source_filter=source_filter)
            # 记录 RAG 答案到日志
            self.logger.info(f"RAG 答案: {answer}")
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # ===== 新增：返回答案前强制刷新日志 =====
            # 作用：确保所有日志（包括 RAG 答案和处理耗时）
            #       在 return 之前完整输出到控制台，避免延迟。
            flush_all_loggers()
            # 返回 RAG 答案
            return answer
        else:
            # 如果无答案，记录信息到日志
            self.logger.info("未找到答案")
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # ===== 新增：返回答案前强制刷新日志 =====
            # 作用：确保所有日志（包括未找到答案和处理耗时）
            #       在 return 之前完整输出到控制台，避免延迟。
            flush_all_loggers()

            # 返回默认答案
            return "未找到答案"


def main():
    qa_system = IntegratedQASystem()
    try:
        print("\n" + "=" * 60)
        print("欢迎使用集成问答系统！")
        print(f"支持的来源: {qa_system.config.VALID_SOURCES}")
        print("输入查询进行问答，输入 'exit' 退出。")
        print("=" * 60)

        while True:
            # 确保所有日志已刷新
            flush_all_loggers()
            time.sleep(0.01)

            # 打印分隔线，明确区分日志区和输入区
            print("\n" + "-" * 50)
            # 获取用户输入
            query = input("输入查询: ").strip()

            # ===== 拦截误输入的日志内容（无论用户粘贴还是系统混入） =====
            # 如果查询中包含 "EduRAG" 或时间戳格式，说明是日志，直接拦截
            if "EduRAG" in query or re.search(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}', query):
                print("⚠️ 检测到日志内容，请重新输入您的问题。")
                continue

            # ===== 拦截空查询 =====
            if not query:
                print("❌ 查询内容不能为空，请重新输入。")
                continue

            # ===== 退出处理 =====
            if query.lower() == "exit":
                logger.info("退出系统")
                print("再见！")
                break

            # ===== 获取学科过滤 =====
            source_filter = input(
                f"输入来源过滤 ({'/'.join(qa_system.config.VALID_SOURCES)}) (按 Enter 跳过): ").strip()
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                logger.warning(f"无效来源 '{source_filter}'，忽略过滤")
                print(f"无效来源 '{source_filter}'，继续无过滤。")
                source_filter = None

            # ===== 执行查询并输出答案 =====
            answer = qa_system.query(query, source_filter)

            # 打印答案前，再次强制刷新所有日志并稍作等待，确保日志先于答案显示
            flush_all_loggers()
            time.sleep(0.05)  # 0.05 秒足够让日志渲染完成

            print(f"\n答案: {answer}")

            # 强制刷新，确保答案打印完再进入下一次循环
            sys.stdout.flush()

    except Exception as e:
        logger.error(f"系统错误: {e}")
        print(f"发生错误: {e}")
    finally:
        qa_system.mysql_client.close()
if __name__ == "__main__":
    # 如果脚本作为主程序运行，调用 main 函数
    main()
