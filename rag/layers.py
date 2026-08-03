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

P0 改造要点（本次新增）
----------------------
* **P0-2 跨层分数校准**：每层在产出 Passage 时，除了保留层内原始分 `score`，
  额外把校准后的相关概率写进 `metadata["calibrated"]`
  （见 `rag/calibration.py`）。原始分保留用于层内排序 / debug 溯源；
  跨层比较、L4 兜底判定、整体置信度**一律使用 calibrated**。
* **P0-2 修 `or 0.9` bug**：L5 原来写 `score=float(d.get("score") or 0.9)`，
  由于 `traverse_multi_hop` 给多跳实体写死 `score=0.0`，而
  `0.0 or 0.9 == 0.9`，导致所有多跳实体分数被抬到 0.9，进而使
  `retriever` 的 L4 兜底**永不触发**。现改为区分"无分数"与"分数为 0"，
  并按跳数衰减（详见 L5KGLayer.search 内注释）。
* **P0-3 多租户隔离**：L1 / L3 的读写都接受 `namespace` 参数。
  L1 由 QACache 内部按 key 前缀隔离；L3 在 metadata 里存 namespace
  并在 search 时做后过滤。namespace=None 时行为与改造前完全一致。
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

from searcher import web_search

from . import config as rag_config
from . import calibration as rag_calibration   # P0-2：跨层分数校准
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

    def lookup(self, query: str, namespace: Optional[str] = None) -> Optional[str]:
        """直接返回预设答案；命中即可绕过 LLM。

        P0-3：新增 namespace 参数做多租户隔离。QACache 内部会保证
        「精确命中」与「fuzzy 命中」都严格按 namespace 过滤，
        避免 A 用户的答案泄漏给 B 用户。
        """
        return self.qa_cache.get(query, namespace=namespace)

    def search(
        self, query: str, top_k: int = 1, namespace: Optional[str] = None,
    ) -> list[Passage]:
        ans = self.lookup(query, namespace=namespace)
        if ans is None:
            return []
        return [Passage(
            text=ans, title="QA Cache Hit", layer=self.name, score=1.0,
            # P0-2：L1 命中意味着已过「精确匹配」或「0.93 阈值 + 槽位门禁」，
            # 是全系统最高可信的一路。
            metadata={"calibrated": round(
                rag_calibration.calibrate(self.name, 1.0), 4
            )},
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
            score = float(h.get("score", 0.0))
            passages.append(Passage(
                text=h.get("text", ""),
                title=h.get("title", ""),
                url=h.get("url", ""),
                score=score,
                layer=self.name,
                metadata={
                    "source": h.get("source", "wiki"),
                    "chunk_id": h.get("chunk_id"),
                    # P0-2：BGE-M3 余弦 → 统一概率空间（中点 0.55）。
                    # 有了它，"L2 的 0.62" 和 "L4 的 0.95" 才能真正放在一起比。
                    "calibrated": round(
                        rag_calibration.calibrate(self.name, score), 4
                    ),
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

    P0-3（多租户隔离）
    ------------------
    L3 存的是**用户自己的历史问答**，天然是私有数据。改造前所有用户共写
    同一个 FAISS 索引、检索时不做任何过滤，A 用户的历史会作为"外部资料"
    出现在 B 用户的 prompt 里（隐私泄漏 + 答案串味）。

    本次改造在 metadata 里写入 `namespace` 字段，并在 `search()` 里做
    **后过滤**（post-filter）：
        - 向量检索照常取 top-K（FAISS IndexFlatIP 不支持元数据过滤）
        - 命中结果里剔除 namespace 不匹配的条目
        - 为抵消过滤损耗，实际检索 `top_k * OVERFETCH` 条候选

    为什么用后过滤而不是"每个租户一个索引"：
        - 简单、零迁移（历史条目无 namespace 字段 → 视为 None，即全局共享）
        - 单机场景下租户数不多、召回损耗可接受
        - 若未来租户数上千，应改为「按 namespace 分片索引」，
          届时只需替换 VectorStore 的实现，本层接口不变。
    """
    name: LayerName = "L3_history"

    # 后过滤的超取倍数：namespace 过滤会丢弃一部分候选，
    # 多取几倍才能保证过滤后仍有足够的 top_k。
    OVERFETCH: int = 4

    def __init__(self, embedder: Embedder, index_dir: Optional[str] = None):
        self.embedder = embedder
        self.index_dir = index_dir or rag_config.L3_HISTORY_INDEX_DIR
        os.makedirs(self.index_dir, exist_ok=True)
        self._store = VectorStore(self.index_dir, rag_config.RAG_EMBED_DIM)
        self._lock = threading.Lock()

    def search(
        self, query: str, top_k: int = 5, namespace: Optional[str] = None,
    ) -> list[Passage]:
        """检索历史问答。

        Args:
            query:     检索 query。
            top_k:     返回条数。
            namespace: P0-3 多租户隔离。
                       None → 只返回**无 namespace 的全局条目**
                              （与改造前的历史数据兼容）
                       str  → 只返回同 namespace 的条目
                       两种情况都**不会**跨租户串味。
        """
        if len(self._store) == 0:
            return []
        q_vec = self.embedder.embed(query)
        if q_vec is None:
            return []

        # 超取：为 namespace 后过滤预留余量
        fetch_k = top_k * self.OVERFETCH if namespace is not None else top_k

        out: list[Passage] = []
        for meta, score in self._store.search(q_vec, top_k=fetch_k):
            # ---- P0-3：namespace 后过滤 ----
            # 历史条目没有 namespace 字段 → get 返回 None → 只在
            # 查询侧也是 None（全局）时才匹配，保证向后兼容且不泄漏。
            if meta.get("namespace") != namespace:
                continue

            q = meta.get("query", "")
            a = meta.get("answer", "")
            text = f"历史问答：\nQ: {q}\nA: {a}"
            md = {k: v for k, v in meta.items() if k != "answer"}
            # P0-2：L3 与 L2 同为 BGE-M3 余弦，但语义**不等价**——
            # L3 存的是"本系统过去生成的答案"，可能包含过时信息甚至历史幻觉，
            # 不能与 Wikipedia 这类外部权威证据同等看待。
            # 因此 calibration 里 L3 的中点比 L2 高（0.62 vs 0.55），
            # 即"要求更相似才认可"，实现系统性降权。
            md["calibrated"] = round(
                rag_calibration.calibrate(self.name, float(score)), 4
            )
            out.append(Passage(
                text=text, title=q[:40], url="",
                score=score, layer=self.name,
                metadata=md,
            ))
            if len(out) >= top_k:
                break
        return out

    def add(
        self,
        query: str,
        answer: str,
        sources: Optional[list[dict]] = None,
        namespace: Optional[str] = None,
    ) -> bool:
        """向 L3 增量写入一次成功的问答。

        Args:
            namespace: P0-3。写入时打上租户标记，检索时据此过滤。
                       None → 全局共享条目（与改造前一致）。
        """
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
                    # P0-3：租户标记。显式写 None 而不是省略字段，
                    # 让"全局条目"与"缺字段的历史条目"在语义上一致
                    # （meta.get("namespace") 都返回 None）。
                    "namespace": namespace,
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
            # ⚠️ 注意：这个 score 是**位次的线性衰减**，不是相似度。
            #    web 搜索接口不返回可比的相关性分数，只能用排名近似。
            #    正因如此，它绝对不能和 L2/L3 的 BGE 余弦直接比大小——
            #    这正是 P0-2 引入 calibration 的核心动因。
            score = 1.0 - i * 0.05
            out.append(Passage(
                text=r.get("snippet", ""),
                title=r.get("title", ""),
                url=r.get("url", ""),
                score=score,
                layer=self.name,
                metadata={
                    "rank": i,
                    # P0-2：位次 → 概率。校准后 top1≈0.77、rank5≈0.39，
                    # 反映"搜索引擎首条通常靠谱但不保证"的真实可信度。
                    "calibrated": round(
                        rag_calibration.calibrate(self.name, score), 4
                    ),
                },
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

            # ══════════════════════════════════════════════════════════════
            # P0-2：修 `score=float(d.get("score") or 0.9)` 这个隐蔽 bug
            # ══════════════════════════════════════════════════════════════
            # 【原实现的问题】
            #   `traverse_multi_hop()`（kg_retriever.py）给多跳发现的实体
            #   写死了 `"score": 0.0`（注释里明确写着"多跳发现的实体不参与打分"）。
            #   而 Python 的 `0.0 or 0.9` 求值为 `0.9` —— 于是**所有多跳实体
            #   的分数都被静默抬到 0.9**，甚至高于 linker 直接链接到的种子实体。
            #
            #   连锁后果（这是最严重的部分）：
            #   `retriever.retrieve()` 用 `offline_best = max(非 L4 层分数)`
            #   与阈值 0.55 比较来决定是否补 L4 web。只要 KG 抽到任何一个
            #   mention，offline_best 就恒为 0.9 > 0.55 → **L4 web 永不触发**。
            #   用户问时事，系统却只用 Wikidata 的陈旧三元组作答。
            #
            # 【本次修法】
            #   1. 区分「真的没有分数」和「分数就是 0」：
            #      用 `d.get("score") is None` 判断，而不是依赖 `or` 的假值语义。
            #   2. 没有分数时用 `KG_UNSCORED_BASELINE`（0.35，校准后≈0.11）
            #      这个**显著偏低**的保守基准，而不是 0.9。
            #      这样 KG 命中不会再无脑阻止 L4 兜底。
            #   3. 多跳实体：从 `source`（形如 "hop2"）解析跳数，
            #      交给 `calibration.calibrate(..., hop=N)` 按跳衰减。
            #      跳数信息一并写进 metadata，便于上层观测与前端展示。
            raw_score = d.get("score")
            source = str(d.get("source") or "kg")

            # 解析跳数：linker 直接链接的种子实体 source 不是 "hopN"，记为 hop=1；
            # traverse_multi_hop 产出的形如 "hop2"/"hop3"。
            hop = 1
            if source.startswith("hop"):
                try:
                    hop = max(int(source[3:]), 1)
                except ValueError:
                    hop = 2      # 解析失败时保守当 2 跳（宁可降权也不高估）

            if raw_score is None or (hop > 1 and float(raw_score) == 0.0):
                # 情况 A：linker 没给分（未建热门实体向量 / weight 缺失）
                # 情况 B：多跳实体（其 score 被 traverse_multi_hop 写死为 0.0）
                #        → 用种子级基准分，再由 calibrate() 按 hop 衰减
                score = rag_calibration.KG_UNSCORED_BASELINE
            else:
                score = float(raw_score)

            # 校准概率：跨层可比的 P(relevant)，供 retriever 做兜底/置信度判定
            calibrated = rag_calibration.calibrate("L5_kg", score, hop=hop)

            passages.append(Passage(
                text=text,
                title=title,
                url="",
                # score 保留"层内原始分"语义（用于 L5 内部排序、debug 溯源）；
                # 跨层比较统一走 metadata["calibrated"]。
                score=score,
                layer=self.name,
                metadata={
                    "qid": d.get("qid"),
                    "mention": d.get("mention"),
                    "via": d.get("via"),
                    "predicate": d.get("predicate"),
                    "source": source,
                    "hop": hop,                    # P0-2：显式记录跳数
                    "calibrated": round(calibrated, 4),
                },
            ))
        return passages

    def warmup(self) -> None:
        """预热：打开 SQLite + 触发 Linker 热门向量 mmap。"""
        self._lazy_retriever().retrieve("预热", top_k=1, multi_hop=False)
