# EduRAG · 企业知识库智能问答系统

面向 IT 教育场景的学科在线答疑系统：高频标准问题走 FAQ 快速通道（MySQL 精确匹配 + Redis 缓存 + BM25 模糊检索），长尾问题自动回退 RAG 检索增强生成（Milvus 混合检索 + BGE-Reranker 精排 + LLM 流式生成），并支持会话记忆、BERT 意图分类路由、RAGAS 自动评估与 Nacos 服务治理。

## 核心特性

- **三级级联管道**：`MySQL 精确匹配 → Redis 缓存 + BM25 模糊检索（jieba 分词 + Softmax 阈值 0.75 + 核心词交集校验）→ Milvus RAG 检索`。快速通道亚秒返回并完全绕开 LLM 调用，长尾问题自动回退 RAG，显著降低延迟与推理成本。
- **混合检索 + 重排序**：BGE-M3 稠密 / 稀疏双通道嵌入写入 Milvus（`IVF_FLAT + SPARSE_INVERTED_INDEX`，加权融合 1.0 / 0.7）；父子分块（1200 / 300 / 重叠 50）Small-to-Big 检索——子块命中回溯父块，BGE-Reranker-Large 交叉编码精排候选父块。
- **意图分类与策略路由**：基于 `bert-base-chinese` 微调的二分类器（约 1830 条样本，验证准确率 90%+）区分「通用知识 / 专业咨询」，叠加领域关键词强制 + Top-1 相似度 ≥ 0.7 二次校验防漏判；LLM 在「直接 / HyDE 假设文档 / 子查询拆分 / 回溯简化」四种检索策略间自动路由，5 个核心 Prompt 模板多轮迭代。
- **多格式摄入与会话记忆**：自研 OCR 文档加载器（PDF / DOCX / PPT / 图片，基于 RapidOCR）与中文专用切分器（规则递归 + BERT 文档分割模型）；`session_id + 最近 5 轮对话历史`持久化 MySQL 并注入生成上下文；知识库覆盖 6 大学科域、支持按源过滤。
- **服务化演进（v1 → v2）**：v1 为 CLI 交互 + FastAPI REST；v2 增加会话记忆与 SSE 流式输出（`/query/stream` 提供 `session / error / done` 命名事件 + 增量 token 数据流），LLM 统一接入 DashScope `qwen-plus`（OpenAI 兼容协议），支持同步与流式两种响应模式。
- **自动评估与工程加固**：RAGAS 评估流水线（忠实度 / 答案相关性 / 上下文精确率 / 召回率，30 条评估样本，LLM 作为裁判）；自研 RAGAS 兼容的 DashScope 嵌入适配器；支持检索策略 A/B 对比调优。
- **可选服务治理**：AC 入口（`ac_api.py`）可注册 Nacos（服务发现 + 提示词热更新），接入「Agent 中台 → RAG」多智能体链路。

## 架构

```
用户输入
   │
   ▼
┌─────────────────────────────────────────────┐
│  三级级联管道                                  │
│  1) MySQL 精确匹配（FAQ 标准问题）              │
│  2) Redis 缓存 + BM25 模糊检索（jieba+0.75 阈值）│
│  3) 意图分类路由 + RAG 检索增强                  │
└──────────────┬──────────────────────────────┘
               ▼
 BERT 意图分类（bert-base-chinese 微调，90%+）
               │
               ▼
 Milvus 混合检索（BGE-M3 稠密+稀疏，k=5）
               │
               ▼
 BGE-Reranker-Large 精排（m=2，Small-to-Big 回溯父块）
               │
               ▼
 LLM 生成（DashScope qwen-plus，四策略自动路由，SSE 流式）
               │
               ▼
 会话记忆（MySQL 持久化最近 5 轮）→ 回答
```

## 目录结构

```
├── base/                    # 公共模块：配置 / 日志 / Nacos 客户端
├── mysql_qa/                # MySQL FAQ + Redis 缓存 + BM25 模糊检索
├── rag_qa/                  # RAG 核心
│   ├── core/                # 向量库 / RAG 系统 / 意图分类器 / 文档处理
│   ├── edu_document_loaders/ # 自研 OCR 文档加载器（PDF/DOCX/PPT/图片）
│   ├── edu_text_spliter/     # 中文专用切分器（规则递归 + BERT 文档分割）
│   ├── rag_assesment/        # RAGAS 评估流水线
│   └── samples/              # 多格式摄入示例文件
├── old_main.py / old_api.py # v1：CLI 交互 + FastAPI REST + Web UI
├── new_main.py / new_api.py # v2：会话记忆 + SSE 流式 + Web UI
├── ac_api.py                # AC 服务入口（Nacos 注册发现 + 提示词热更新）
└── config.ini.example       # 配置模板（复制为 config.ini 使用）
```

## 快速开始

### 1. 依赖与环境

```bash
git clone <repo-url>
cd integrated_qa_system
pip install -r requirements.txt
cp config.ini.example config.ini   # Windows: copy config.ini.example config.ini
```

编辑 `config.ini`，填入你的 MySQL / Redis / Milvus 连接与 DashScope API Key。

### 2. 模型获取（无需手动下载权重文件）

嵌入 / 重排 / 意图分类 / 文档分割模型均通过 Hugging Face hub id 加载（`BAAI/bge-m3`、`BAAI/bge-reranker-large`、`bert-base-chinese`、`infinitedrafting/nlp_bert_document-segmentation_chinese-base`），首次运行自动下载（国内可设置 `export HF_ENDPOINT=https://hf-mirror.com`）。

意图二分类器支持本地微调：`python -m rag_qa.core.query_classifier` 会基于 `rag_qa/classify_data/model_generic_5000.json` 训练并保存到 `rag_qa/core/bert_query_classifier/`（未微调时退化为 `bert-base-chinese` 初始化）。

### 3. 运行

```bash
# v2 命令行交互（会话记忆 + 流式）
python new_main.py

# v2 REST API（/query /query/stream /session /history ...，端口 18077 见 config.ini [ac]）
python new_api.py

# v1 对比版本
python old_main.py        # CLI
python old_api.py         # REST

# AC 服务入口（需先启动 Nacos）
python ac_api.py

# RAGAS 评估
python rag_qa/rag_assesment/rag_as.py
```

Web UI 直接浏览器打开 `new_index.html`（v2）或 `old_index.html`（v1）。

## 关键指标

| 指标 | 数值 |
| --- | --- |
| BERT 意图分类验证准确率 | 90%+（约 1830 条训练样本） |
| BM25 模糊检索阈值 | Softmax 0.75 + 核心词交集校验 |
| 混合检索 | k = 5 → Reranker m = 2 |
| 分块策略 | 父块 1200 / 子块 300 / 重叠 50 |
| RAGAS 评估 | 30 条样本，忠实度 / 相关性 / 精确率 / 召回率 |

## 相关项目

本系统作为检索增强能力已集成至 [AgentCenter 智能体中台](https://github.com/your-github-username/agent-center)（多智能体 + MCP 工具 + RAG + 微服务治理完整链路），见该项目 README。

## 许可证

[MIT](LICENSE)

