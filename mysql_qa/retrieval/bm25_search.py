# 导入 BM25 算法
from rank_bm25 import BM25Okapi
# 导入数值计算库
import numpy as np

import sys, os
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
"""
__file__ 是当前文件的完整路径（如 E:/py_base/.../mysql_client.py）。
os.path.abspath() 获取绝对路径，
os.path.dirname() 获取所在目录。
最终 current_dir 就是当前脚本所在的文件夹路径。
"""
module_dir = os.path.dirname(current_dir)
sys.path.insert(0, module_dir)
project_root = os.path.dirname(module_dir)
sys.path.insert(0, project_root)
#将项目根目录添加到 Python 的模块搜索路径首位，以便后续导入 base 等自定义模块。

# --- 新增：将 base 目录加入 sys.path ---
base_dir = os.path.join(project_root, 'base')
sys.path.insert(0, base_dir)
"""
构造 base 目录的路径并插入搜索路径，使得可以直接 from base import Config, logger。
"""
# 导入文本预处理
from utils.preprocess import preprocess_text
#导入MySQLClient
from db.mysql_client import MySQLClient
#导入redis_client
from cache.redis_client import RedisClient
# 导入日志
from base import logger


class BM25Search:
    def __init__(self, redis_client, mysql_client):
        # 初始化日志
        self.logger = logger
        # 初始化 Redis 客户端
        self.redis_client = redis_client
        # 初始化 MySQL 客户端
        self.mysql_client = mysql_client
        # 初始化 BM25 模型
        self.bm25 = None
        # 初始化问题列表
        self.questions = None
        # 初始化原始问题
        self.original_questions = None
        # 加载数据
        self._load_data()

    def _load_data(self):
        #加载数据
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        #从 Redis 获取原始问题
        self.original_questions=self.redis_client.get_data(original_key)


        # 从 Redis 获取分词问题
        tokenized_questions =self.redis_client.get_data(tokenized_key)
        # 如果 Redis 中没有数据，从 MySQL 加载
        if not self.original_questions or not tokenized_questions:
            # 从 MySQL 获取问题
            self.original_questions=self.mysql_client.fetch_questions()
            if not self.original_questions:
                # 记录无问题警告
                self.logger.warning("未加载到问题")
                return

            # 分词问题
            tokenized_questions=[preprocess_text(q[0]) for q in self.original_questions]
            #存储原始问题到 Redis
            self.redis_client.set_data(original_key,  [(q[0]) for q in self.original_questions])
            # 存储分词问题到 Redis
            self.redis_client.set_data(tokenized_key, tokenized_questions)


        # 设置问题列表
        self.questions = tokenized_questions
        # 初始化 BM25 模型
        self.bm25 = BM25Okapi(self.questions)
        # 记录 BM25 初始化成功
        self.logger.info(f"BM25 模型初始化完成")
        #1
        #self.logger.info(f"BM25 模型初始化完成,文档数: {len(self.questions)}")

    def _softmax(self, scores):
        #计算 Softmax 分数
        exp_scores = np.exp(scores - np.max(scores))
        # 返回归一化分数
        return exp_scores / exp_scores.sum()

    def search(self, query, threshold=0.85):
        # 搜索查询
        if not query or not isinstance(query, str):
            self.logger.error("无效的查询")
            return None, False

        # 检查 Redis 缓存
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            return cached_answer, False

        # ===== 1. 精确匹配（完全一致）=====
        if self.original_questions:
            for idx, q in enumerate(self.original_questions):
                if q == query:
                    answer = self.mysql_client.fetch_answer(q)
                    if answer:
                        self.redis_client.set_data(f"answer:{query}", answer)
                        self.logger.info(f"精确匹配成功")
                        return answer, False

        # ===== 2. BM25 搜索 =====
        try:
            query_tokens = preprocess_text(query)
            scores = self.bm25.get_scores(query_tokens)
            softmax_scores = self._softmax(scores)
            best_idx = softmax_scores.argmax()
            best_score = softmax_scores[best_idx]

            # ===== 3. 核心词匹配检查（防止"课程"侥幸匹配）=====
            core_words = [w for w in query_tokens if len(w) > 1 and w not in ["的", "了", "吗", "呢", "啊", "呀"]]
            candidate_question = self.original_questions[best_idx]
            candidate_tokens = preprocess_text(candidate_question)

            # 如果核心词在候选问题中的匹配数量为 0，直接回退到 RAG
            if core_words and not any(w in candidate_tokens for w in core_words):
                self.logger.info(f"BM25 最高分文档与查询核心词无交集，回退到 RAG")
                return None, True

            # ===== 4. 原有的阈值判断 =====
            if best_score >= threshold:
                # 获取原始问题
                original_question = self.original_questions[best_idx]
                # 获取答案
                answer = self.mysql_client.fetch_answer(original_question)
                if answer:
                    # 缓存答案
                    self.redis_client.set_data(f"answer:{query}", answer)
                    # 记录搜索成功
                    self.logger.info(f"搜索成功，Softmax 相似度: {best_score:.3f}")
                    # 返回答案和 False
                    return answer, False

            # 记录无可靠答案
            self.logger.info(f"未找到可靠答案，最高 Softmax 相似度: {best_score:.3f}")
            # 返回 None 和 True
            return None, True

        except Exception as e:
            # 记录搜索失败
            self.logger.error(f"搜索失败: {e}")
            # 返回 None 和 True
            return None, True

if __name__ == '__main__':
    redis_client = RedisClient()
    mysql_client = MySQLClient()
    bm25_search = BM25Search(redis_client, mysql_client)
    query="用上下文管理器实现函数运行时间的计算"
    print(bm25_search.search(query))
