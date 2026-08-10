"""wiki_rag —— 中文维基百科本地 RAG 工具包。

对外暴露三个高层构件：

    - `WikiRetriever`           基于 FAISS 索引的本地稠密检索
    - `hybrid_retrieve`         本地 Wiki + 外部 Web 搜索的后融合
    - `build_default_reranker`  BGE 重排器的默认工厂
"""
