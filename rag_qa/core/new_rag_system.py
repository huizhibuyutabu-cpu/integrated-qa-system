# new_rag_system.py
# 功能：增强版 RAG 系统，仅保留流式生成接口，非流式可通过收集流式结果实现。
# 说明：移除非流式 generate_answer 方法，统一使用 generate_answer_stream。
# 作者：EduRAG 项目组

from prompts import RAGPrompts
#   导入 time 模块，用于计算时间
import time

import sys, os
# 当前文件所在目录：.../core
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录：.../integrated_qa_system
root_dir = os.path.dirname(os.path.dirname(current_dir))
# base 目录：.../integrated_qa_system/base
base_dir = os.path.join(root_dir, 'base')

# 将根目录和 base 目录都加入 sys.path
# 注意顺序：先加入 base 目录，让 'config' 能直接找到，再加入根目录，让 'base' 包可导入
# 如果 base 目录先加入，import config 会找到 base/config.py；
# 然后根目录加入后，import base 会找到根目录下的 base 包

if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)   # 或使用 append，但 insert(0) 优先级最高
from base import logger, Config
from query_classifier import QueryClassifier  #   导入查询分类器
from strategy_selector import StrategySelector  #   导入策略选择器
from vector_store import VectorStore    #导入向量检索

conf = Config()

#   定义 RAGSystem 类，封装 RAG 系统的核心逻辑
class RAGSystem:
    #   初始化方法，设置 RAG 系统的基本参数
    def __init__(self, vector_store, llm, llm_stream=None):
        #   设置向量数据库对象
        self.vector_store = vector_store
        #   设置大语言模型调用函数（非流式，用于检索策略中的 LLM 调用）
        self.llm = llm
        #   设置流式大语言模型调用函数（生成器），若未提供则使用非流式函数（但会忽略流式）
        self.llm_stream = llm_stream if llm_stream is not None else llm
        #   获取 RAG 提示模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        #   初始化查询分类器
        self.query_classifier = QueryClassifier(model_path=os.path.join(os.path.dirname(__file__), 'bert_query_classifier'))
        #   初始化策略选择器
        self.strategy_selector = StrategySelector()

        # ===================== 新增：强制 RAG 关键词列表 =====================
        # 作用：当查询包含这些关键词时，直接归类为“专业咨询”，绕过分类器，
        # 确保这些术语相关的查询一定走 RAG 流程，避免被误判为通用知识。
        self.rag_keywords = [
            "LLM", "大语言模型", "Transformer", "预训练", "BERT", "GPT", "微调",
            "N-gram", "困惑度", "BLEU", "ROUGE",
            "语言模型", "神经网络语言模型", "词向量", "N-gram模型", "统计语言模型",
            "N-gram语言模型", "基于规则和统计的语言模型",
            # 扩充更多术语（根据需要添加）
            "RNN", "LSTM", "GRU", "CNN", "FastText", "知识图谱", "LangChain", "RAG",
            "迁移学习", "预训练模型", "微调", "命名实体识别", "关系抽取", "注意力机制"
        ]

        # ===================== 新增：二次验证的相似度阈值 =====================
        # 作用：当分类器将查询判为“通用知识”时，我们仍会执行一次轻量级向量检索，
        # 如果最高相似度 >= 此阈值，则强制将其转为“专业咨询”，确保与文档内容相关的查询不会漏掉。
        self.vector_threshold = 0.7

    # ===================== 流式生成答案方法（唯一接口） =====================
    def generate_answer_stream(self, query, source_filter=None):
        """
        流式生成答案：检索逻辑与原来相同，最后逐步 yield 生成内容。
        注意：此方法内部的检索日志会正常输出，完成日志已设为 debug 级别，
        避免干扰用户界面，耗时信息由调用方（如 new_main.py）统一打印。
        """
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 学科过滤: {source_filter}")

        # ===== 第一优先级：强制 RAG 关键词检测 =====
        if any(kw in query for kw in self.rag_keywords):
            query_category = "专业咨询"
            logger.info(f"✅ 查询包含强制 RAG 关键词，强制分类为 '专业咨询'")
        else:
            query_category = self.query_classifier.predict_category(query)
            logger.info(f"查询分类结果：{query_category} (查询: '{query}')")

            # ===== 第二优先级：向量相似度二次验证 =====
            if query_category == "通用知识":
                try:
                    docs = self.vector_store.hybrid_search_with_rerank(
                        query, k=1, source_filter=source_filter
                    )
                    if docs:
                        score = getattr(docs[0], 'score', None)
                        if score is None and hasattr(docs[0], 'metadata'):
                            score = docs[0].metadata.get('score')
                        if score is not None and score >= self.vector_threshold:
                            logger.info(f"🔍 二次验证：向量相似度 {score:.4f} ≥ {self.vector_threshold}，强制转为 '专业咨询'")
                            query_category = "专业咨询"
                        else:
                            logger.info(f"🔍 二次验证：向量相似度 {score if score is not None else 0:.4f} < {self.vector_threshold}，保持 '通用知识'")
                except Exception as e:
                    logger.warning(f"二次向量验证失败: {e}，仍按原分类处理")

        if query_category == "通用知识":
            logger.info("查询为通用知识，直接调用 LLM（无检索）")
            prompt_input = self.rag_prompt.format(
                context="", question=query, phone=conf.CUSTOMER_SERVICE_PHONE
            )
            try:
                for chunk in self.llm_stream(prompt_input):
                    yield chunk
            except Exception as e:
                logger.error(f"流式直接调用 LLM 失败: {e}")
                yield f"抱歉，处理您的通用知识问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
            processing_time = time.time() - start_time
            logger.debug(f"通用知识流式查询处理完成 (耗时: {processing_time:.2f}s)")
            return

        #   否则，进行 RAG 检索并生成答案（专业咨询）
        logger.info("查询为专业咨询，执行完整 RAG 流程（检索 + 流式生成）")
        #   选择检索策略
        strategy = self.strategy_selector.select_strategy(query)

        #   检索相关文档
        context_docs = self.retrieve_and_merge(
            query, source_filter=source_filter, strategy=strategy
        )

        #   准备上下文
        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs])
            logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
        else:
            context = ""
            logger.warning("未检索到相关文档，上下文为空（将使用 LLM 自身知识）")

        #   构造 Prompt，调用流式大语言模型生成答案
        prompt_input = self.rag_prompt.format(
            context=context, question=query, phone=conf.CUSTOMER_SERVICE_PHONE
        )

        try:
            full_answer = ""
            for chunk in self.llm_stream(prompt_input):
                full_answer += chunk
                yield chunk
            processing_time = time.time() - start_time
            # 使用 debug 级别，避免在控制台显示完成日志（耗时由调用方打印）
            logger.debug(f"专业咨询 RAG 流式处理完成 (耗时: {processing_time:.2f}s, 答案长度: {len(full_answer)})")
        except Exception as e:
            logger.error(f"流式 LLM 生成最终答案失败: {e}")
            yield f"抱歉，处理您的专业咨询问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"

    # 以下 retrieve_and_merge 及后续方法保持不变（完全保留原代码）
    def retrieve_and_merge(self, query, source_filter=None, strategy=None):
        """
        根据选定的策略执行文档检索，并合并结果。
        Args:
            query (str): 查询文本。
            source_filter (str, optional): 学科过滤。
            strategy (str, optional): 指定的检索策略，若不提供则自动选择。
        Returns:
            list[Document]: 最终选作上下文的文档列表。
        """
        #   如果未指定检索策略，则使用策略选择器选择
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        #   根据检索策略选择不同的检索方式
        ranked_sub_chunks = []  # 初始化
        if strategy == "回溯问题检索":
            ranked_sub_chunks = self._retrieve_with_backtracking(query, source_filter)
        elif strategy == "子查询检索":
            ranked_sub_chunks = self._retrieve_with_subqueries(query, source_filter)
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._retrieve_with_hyde(query, source_filter)
        else:  # 默认或“直接检索”
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
                query, k=conf.RETRIEVAL_K, source_filter=source_filter
            )

        logger.info(f"策略 '{strategy}' 检索到 {len(ranked_sub_chunks)} 个候选文档")
        final_context_docs = ranked_sub_chunks[:conf.CANDIDATE_M]
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")
        return final_context_docs

    #   定义类私有方法，使用回溯问题进行检索
    def _retrieve_with_backtracking(self, query, source_filter):
        """
        回溯问题检索策略：将复杂问题简化为更基础的问题，再检索。
        """
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()
        try:
            simplified_query = self.llm(backtrack_prompt_template.format(query=query)).strip()
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query, k=conf.RETRIEVAL_K, source_filter=source_filter
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []

    #   定义类私有方法，使用子查询进行检索
    def _retrieve_with_subqueries(self, query, source_filter):
        """
        子查询检索策略：将复杂查询拆分为多个子查询，分别检索后合并去重。
        """
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        subquery_prompt_template = RAGPrompts.subquery_prompt()
        try:
            subqueries_text = self.llm(subquery_prompt_template.format(query=query)).strip()
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return []

            all_docs = []
            for sub_q in subqueries:
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q, k=conf.RETRIEVAL_K//2, source_filter=source_filter
                )
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")

            unique_docs_dict = {doc.page_content: doc for doc in all_docs}
            unique_docs = list(unique_docs_dict.values())
            logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            return unique_docs
        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []

    #   定义类私有方法，使用假设文档进行检索（HyDE）
    def _retrieve_with_hyde(self, query, source_filter):
        """
        HyDE (假设文档嵌入) 检索策略：先生成一个假设的答案，再用该答案进行检索。
        """
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        hyde_prompt_template = RAGPrompts.hyde_prompt()
        try:
            hypo_answer = self.llm(hyde_prompt_template.format(query=query)).strip()
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer, k=conf.RETRIEVAL_K, source_filter=source_filter
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []


if __name__ == '__main__':
    # 用于快速测试
    vector_store = VectorStore()
    # 注意：测试时需提供有效的 llm 函数
    def dummy_llm(prompt): return "模拟答案"
    def dummy_stream(prompt): yield "模拟流式答案"
    rag_system = RAGSystem(vector_store, dummy_llm, dummy_stream)
    # 测试流式
    for chunk in rag_system.generate_answer_stream(query="AI学科的课程大纲内容有什么", source_filter="ai"):
        print(chunk, end="")