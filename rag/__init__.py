# rag/__init__.py
"""分层记忆 RAG 系统（方案 C，已集成 wiki_rag）。

五层结构：
    L1  QACache            —— 精准/模糊问答缓存（qa_cache.py，向量统一 BGE-M3）    -- agent命中即写入
    L2  Commonsense VDB    —— Wikipedia 常识向量库（wiki_rag.WikiRetriever，FAISS+BGE-M3）  -- 离线更新
    L3  History Archive    —— 用户历史 QA 归档（每次成功回答后异步入库）             -- 每次成功回答异步写
    L4  Web Search         —— 实时联网检索（复用 searcher.web_search）            -- 实时，网页搜索
    L5  Knowledge Graph    —— Wikidata truthy 知识图谱（wiki_rag.KGRetriever，SQLite+多跳）  -- 离线更新

编排：
    Router      —— 根据 query 类型决定要激活的层（可并行）
    Retriever   —— 并发拉取各层结果并调用 Fusion
    Fusion      —— RRF 融合（默认）+ 可选 BGE/cascade rerank（环境变量开启）
    IncrementalWorker —— 后台搜索历史 (query, answer, sources) 事件写 L3

统一编码（D1）：L1~L5 全部使用 FlagEmbedding BGE-M3，向量空间一致。
数据路径（D3）：全部位于项目根 data/rag_data/（见 rag/configs/default.yaml）。

对外主入口：
    >>> from rag import LayeredRetriever
    >>> retriever = LayeredRetriever()
    >>> ctx = retriever.retrieve("量子计算是什么")
    >>> retriever.archive(query, answer, sources)   # 由 agent 在回答成功后调用
"""
from .retriever import LayeredRetriever
from .types import Passage, RetrievalResult, LayerName

__all__ = ["LayeredRetriever", "Passage", "RetrievalResult", "LayerName"]
