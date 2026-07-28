# rag/layers.py
"""L1 ~ L5 五层检索器。每层实现同一接口：

    class Layer:
        name: LayerName
        def search(self, query: str, top_k: int) -> list[Passage]: ...

分工（本次改造后）：
    L1 QACache      —— 复用 qa_cache.QACache（精确 + 向量模糊，向量已统一 BGE-M3）
    L2 Commonsense  —— wiki_rag.WikiRetriever（FAISS + BGE-M3），只用 Wikipedia dump
    L3 History      —— 用户 QA 归档（VectorStore，可增量写；向量走统一 Embedder=BGE-M3）
    L4 Web          —— 复用 searcher.web_search
    L5 KG           —— wiki_rag.KGRetriever（Wikidata truthy → SQLite + 热门实体向量重排 + 可选多跳）

设计要点
--------
* **统一向量空间（D1）**：L2 内部用 FlagEmbedding BGE-M3 编码 query，与离线构建索引
  时同一模型；L3 用 rag.embedder.Embedder（底层同样是 BGE-M3），保证跨层可比。
* **L2/L5 惰性加载**：WikiRetriever/KGRetriever 首次 search 时才构造（要读 GB 级
  FAISS / SQLite），避免 agent 启动即加载大文件；构造后进程内复用。
* **CN-DBpedia 已彻底移除**：L2 只保留 Wikipedia 一路。
* **结果适配**：WikiRetriever/KGRetriever 返回 List[Dict]，这里统一转成 Passage，
  以维持 LayeredRetriever / RRF / RetrievalResult 的既有契约。
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

from searcher import web_search

from . import config as rag_config
from .embedder import Embedder
from .types import LayerName, Passage
from .vector_store import VectorStore
from .wiki_rag.kg_retriever import KGRetriever
from .wiki_rag.retriever import WikiRetriever


# ---------- L1: 精准 / 模糊 QA 缓存 ----------
class L1QACacheLayer:
    name: LayerName = "L1_qa"

    def __init__(self, qa_cache):
        """qa_cache 需要是 qa_cache.QACache 实例（外部传入以复用配置）。"""
        self.qa_cache = qa_cache

    def lookup(self, query: str) -> Optional[str]:
        """直接返回预设答案；命中即可绕过 LLM。"""
        return self.qa_cache.get(query)

    def search(self, query: str, top_k: int = 1) -> list[Passage]:
        ans = self.lookup(query)
        if ans is None:
            return []
        return [Passage(
            text=ans, title="QA Cache Hit", layer=self.name, score=1.0,
        )]


# ---------- L2: 常识向量库（Wikipedia，wiki_rag.WikiRetriever） ----------
class L2CommonsenseLayer:
    """L2 常识层：封装 wiki_rag 的 :class:`WikiRetriever`。

    - 数据来源：仅 Wikipedia dump（05/06 步构建的 FAISS 索引 + chunks.jsonl）。
    - 编码器：FlagEmbedding BGE-M3（WikiRetriever 内部持有，与索引同源）。
    - 惰性加载：首次 search 时才 read_index（GB 级），构造后进程内复用。
    """
    name: LayerName = "L2_wiki"

    def __init__(self):
        self._retriever: Optional[WikiRetriever] = None   # 懒加载
        self._lock = threading.Lock()

    def _lazy_retriever(self) -> WikiRetriever:
        """首次调用时构造 WikiRetriever（双检锁，进程内单例）。"""
        if self._retriever is None:
            with self._lock:
                if self._retriever is None:
                    self._retriever = WikiRetriever()
        return self._retriever

    def search(self, query: str, top_k: int = 8) -> list[Passage]:
        hits = self._lazy_retriever().search(query, top_k=top_k)
        passages: list[Passage] = []
        for h in hits:
            passages.append(Passage(
                text=h.get("text", ""),
                title=h.get("title", ""),
                url=h.get("url", ""),
                score=float(h.get("score", 0.0)),
                layer=self.name,
                metadata={
                    "source": h.get("source", "wiki"),
                    "chunk_id": h.get("chunk_id"),
                },
            ))
        return passages

    def warmup(self) -> None:
        """预热：加载索引 + BGE-M3 权重（服务启动时可显式调用）。"""
        self._lazy_retriever().warmup()


# ---------- L3: 用户历史 QA 归档 ----------
class L3HistoryLayer:
    """支持增量写入的历史 QA 向量库（越用越强）。

    向量走统一 Embedder（BGE-M3），与 L2/L5 同一向量空间。
    """
    name: LayerName = "L3_history"

    def __init__(self, embedder: Embedder, index_dir: Optional[str] = None):
        self.embedder = embedder
        self.index_dir = index_dir or rag_config.L3_HISTORY_INDEX_DIR
        os.makedirs(self.index_dir, exist_ok=True)
        self._store = VectorStore(self.index_dir, rag_config.RAG_EMBED_DIM)
        self._lock = threading.Lock()

    def search(self, query: str, top_k: int = 5) -> list[Passage]:
        if len(self._store) == 0:
            return []
        q_vec = self.embedder.embed(query)
        if q_vec is None:
            return []
        out: list[Passage] = []
        for meta, score in self._store.search(q_vec, top_k=top_k):
            q = meta.get("query", "")
            a = meta.get("answer", "")
            text = f"历史问答：\nQ: {q}\nA: {a}"
            out.append(Passage(
                text=text, title=q[:40], url="",
                score=score, layer=self.name,
                metadata={k: v for k, v in meta.items() if k != "answer"},
            ))
        return out

    def add(self, query: str, answer: str, sources: Optional[list[dict]] = None) -> bool:
        """向 L3 增量写入一次成功的问答。"""
        text = f"{query}\n{answer}"[:2000]
        vec = self.embedder.embed(text)
        if vec is None:
            return False
        with self._lock:
            self._store.add(
                vectors=[vec],
                metadatas=[{
                    "query": query,
                    "answer": answer,
                    "sources": sources or [],
                    "ts": time.time(),
                }],
            )
            self._store.save()
        return True


# ---------- L4: Web 实时搜索兜底 ----------
class L4WebLayer:
    name: LayerName = "L4_web"

    def search(self, query: str, top_k: int = 5) -> list[Passage]:
        results = web_search(query, top_k=top_k)
        out: list[Passage] = []
        for i, r in enumerate(results):
            out.append(Passage(
                text=r.get("snippet", ""),
                title=r.get("title", ""),
                url=r.get("url", ""),
                score=1.0 - i * 0.05,   # 按顺序衰减（web 端本身无相似度）
                layer=self.name,
                metadata={"rank": i},
            ))
        return out


# ---------- L5: 知识图谱（Wikidata truthy，wiki_rag.KGRetriever） ----------
class L5KGLayer:
    """L5 知识图谱层：封装 wiki_rag 的 :class:`KGRetriever`。

    - 数据来源：Wikidata "truthy" dump → SQLite（09/10/11 步构建）。
    - 能力：mention 抽取（hybrid）→ 实体链接消歧（热门实体 BGE-M3 向量重排）
      → 拉取三元组事实 → 可选多跳 BFS。
    - 惰性加载：首次 search 时才打开 SQLite / 加载热门向量，构造后进程内复用。
    - 多跳开关：由 config.KG_MULTI_HOP 控制（默认关闭以控延迟）。
    """
    name: LayerName = "L5_kg"

    def __init__(self):
        self._retriever: Optional[KGRetriever] = None   # 懒加载
        self._lock = threading.Lock()

    def _lazy_retriever(self) -> KGRetriever:
        """首次调用时构造 KGRetriever（双检锁，进程内单例）。"""
        if self._retriever is None:
            with self._lock:
                if self._retriever is None:
                    # config_path=None → KGRetriever 内部用 vendored 默认 YAML
                    self._retriever = KGRetriever(config_path=None, mention_method="hybrid")
        return self._retriever

    def search(self, query: str, top_k: int = 3) -> list[Passage]:
        docs = self._lazy_retriever().retrieve(
            query,
            top_k=top_k,
            multi_hop=rag_config.KG_MULTI_HOP,
            max_hops=rag_config.KG_MAX_HOPS,
        )
        passages: list[Passage] = []
        for d in docs:
            title = d.get("title", "")
            text = d.get("text", "")
            if not text:
                continue
            passages.append(Passage(
                text=text,
                title=title,
                url="",
                # KG 事实是强先验，给一个较高的基准分（0.9）便于 RRF 中靠前
                score=float(d.get("score") or 0.9),
                layer=self.name,
                metadata={
                    "qid": d.get("qid"),
                    "mention": d.get("mention"),
                    "via": d.get("via"),
                    "predicate": d.get("predicate"),
                },
            ))
        return passages

    def warmup(self) -> None:
        """预热：打开 SQLite + 触发 Linker 热门向量 mmap。"""
        self._lazy_retriever().retrieve("预热", top_k=1, multi_hop=False)
