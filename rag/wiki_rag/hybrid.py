"""混合检索：本地 Wiki 向量库 + Web 搜索 + KG 三元组 + 重排。

对外入口 :func:`hybrid_retrieve` 由四个可选阶段串起来：

1. **KG 事实注入** —— 若传入 ``kg_retriever``（:class:`wiki_rag.kg_retriever.KGRetriever`）
   或传统的 ``linker``（向后兼容），从 query 里抽 mention → link 到 QID → 拉三元组
   → (可选) 多跳图遍历 → 拼成"事实条列"文本，作为高置信度的 ``source="kg"`` 文档置顶。
   
   推荐用 ``kg_retriever``：它支持多跳 (``multi_hop=True``) 和 hybrid 抽取方案。

2. **本地稠密检索** —— 走 :class:`WikiRetriever` 的 FAISS 索引。

3. **外部 Web 搜索** —— 通过注入的 callable 调用，模块本身不绑定任何具体后端。

4. **Cross-Encoder 重排** —— 通过注入的打分函数完成；:func:`build_default_reranker`
   给出一个基于 BGE 的默认实现。KG 事实不参与重排（视为强先验，保持置顶）。

阶段 2 和阶段 3 之间会按 ``(source, title)`` 去重。
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional

from .retriever import WikiRetriever


# ---------------------------------------------------------------- mention 抽取（旧路径兼容）
# 保留：当用户只传旧的 `linker` 参数时使用；新代码请走 kg_retriever。
def _extract_mentions(query: str) -> List[str]:
    """从 query 里抽出可能的 mention 词（旧路径，仅用 jieba 词性）。"""
    try:
        import jieba.posseg as pseg  # type: ignore
        keep = {"n", "nr", "ns", "nt", "nz", "eng"}
        toks = [w.word for w in pseg.cut(query)
                if w.flag[:2] in keep or w.flag in keep]
        seen, out = set(), []
        for t in toks:
            if len(t) < 2 or t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out or [query]
    except ImportError:
        return [query]


def hybrid_retrieve(query: str,
                    wiki_retriever: WikiRetriever,           # WikiRetriever
                    web_search_fn: Optional[Callable[[str, int], List[Dict]]] = None,
                    rerank_fn: Optional[Callable[[str, List[Dict]], List[Dict]]] = None,
                    *,
                    kg_retriever: Optional[object] = None,   # KGRetriever
                    linker: Optional[object] = None,         # 向后兼容：旧接口
                    multi_hop: bool = False,
                    max_hops: int = 2,
                    top_k_wiki: int = 5,
                    top_k_web: int = 5,
                    top_k_kg: int = 3,
                    final_k: int = 5) -> List[Dict]:
    """融合 KG / 本地 / Web 三路结果，可选重排，返回 Top-``final_k``。

    Args:
        query:          用户 query。
        wiki_retriever: 预加载好的本地向量检索器。
        web_search_fn:  可选 ``(query, k) -> List[Dict]``。异常会被捕获并日志。
        rerank_fn:      可选 ``(query, docs) -> reordered docs``。
                        注意：KG 事实不进入这一阶段，避免被打散。
        kg_retriever:   ✅ **推荐**：:class:`KGRetriever` 实例。支持 multi_hop / 多种 mention 抽取。
        linker:         ⚠️ 向后兼容：老的 Linker 实例。当 kg_retriever 未提供时启用。
                        （新代码建议直接传 kg_retriever）
        multi_hop:      是否打开多跳 KG 图遍历（仅 kg_retriever 生效）。
                        默认关闭；打开会额外增加 10-数百 ms 延迟，
                        但对"库克领导的公司在哪个城市"这类多跳问题必需。
        max_hops:       多跳最大跳数（1=只种子；2=种子+1 跳邻居；3=更多层）。
        top_k_wiki:     从本地 FAISS 取几条。
        top_k_web:      从 Web 后端取几条。
        top_k_kg:       KG 侧最多注入几个"实体的事实块"。
        final_k:        wiki+web 部分最终返回的条数上限（KG 事实块另外置顶，不计入）。

    Returns:
        List[Dict]，字段 ``{source, title, text, score, ...}``；
        KG 事实块永远置顶（不参与重排）。
        额外挂在函数属性 ``hybrid_retrieve.last_timing`` 上返回本次每阶段耗时（毫秒）。
    """
    import time as _t
    timing: Dict[str, float] = {}

    # ---- 阶段 1：KG 事实注入（可选）----
    _t0 = _t.perf_counter()
    kg_docs: List[Dict] = []

    if kg_retriever is not None:
        # ---- 新路径：直接用 KGRetriever（支持 multi_hop / hybrid mention 抽取）----
        try:
            kg_docs = kg_retriever.retrieve(   # type: ignore[attr-defined]
                query,
                top_k=top_k_kg,             # top3
                multi_hop=multi_hop,        # false
                max_hops=max_hops,          # true
            )
        except Exception as e:
            print(f"[hybrid] kg_retriever failed: {e}")
            kg_docs = []

    elif linker is not None:
        # ---- 旧路径：单跳、jieba mention 抽取（向后兼容）----
        if multi_hop:
            print("[hybrid] warning: multi_hop 只在传 kg_retriever 时生效，"
                  "当前用的是老 linker 接口，将只跑 1 跳。")
        mentions = _extract_mentions(query)
        seen_qids: set[str] = set()
        for m in mentions:
            try:
                cands = linker.link_and_expand(  # type: ignore[attr-defined]
                    m, query_context=query, top_k=1)
            except Exception as e:
                print(f"[hybrid] linker failed on mention={m!r}: {e}")
                continue
            for c in cands:
                if c["qid"] in seen_qids or not c.get("context"):
                    continue
                seen_qids.add(c["qid"])
                kg_docs.append({
                    "source":  "kg",
                    "title":   c["label_zh"] or c["qid"],
                    "text":    c["context"],
                    "qid":     c["qid"],
                    "mention": m,
                    "score":   float(c.get("score", 0.0)),
                })
                if len(kg_docs) >= top_k_kg:
                    break
            if len(kg_docs) >= top_k_kg:
                break
    timing["kg_ms"] = (_t.perf_counter() - _t0) * 1000

    # ---- 阶段 2：本地稠密检索 ----
    _t0 = _t.perf_counter()
    wiki_hits = wiki_retriever.search(query, top_k=top_k_wiki)
    timing["wiki_total_ms"] = (_t.perf_counter() - _t0) * 1000
    # 从 retriever 拿细分（encode vs faiss）
    timing["wiki_encode_ms"] = float(getattr(wiki_retriever, "_last_encode_ms", 0.0))
    timing["wiki_faiss_ms"]  = float(getattr(wiki_retriever, "_last_faiss_ms",  0.0))

    # ---- 阶段 3：Web 搜索（可选）----
    _t0 = _t.perf_counter()
    web_hits: List[Dict] = []
    if web_search_fn is not None:
        try:
            web_hits = web_search_fn(query, top_k_web) or []
        except Exception as e:
            print(f"[hybrid] web_search failed: {e}")
    timing["web_ms"] = (_t.perf_counter() - _t0) * 1000

    # 按 (source, title) 去重
    _t0 = _t.perf_counter()
    seen, merged = set(), []
    # 先给每一路命中标注"在本源内的排名"（RRF 会用到）
    for src_hits in (wiki_hits, web_hits):
        for rank, d in enumerate(src_hits, start=1):
            d.setdefault("_rank_in_source", rank)
    for d in wiki_hits + web_hits:
        key = (d.get("source"), d.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(d)
    timing["dedup_ms"] = (_t.perf_counter() - _t0) * 1000

    # ---- 阶段 4：重排（可选）----
    _t0 = _t.perf_counter()
    if rerank_fn is not None and merged:
        merged = rerank_fn(query, merged)
    timing["rerank_ms"] = (_t.perf_counter() - _t0) * 1000

    timing["total_ms"] = sum(v for k, v in timing.items()
                             if k in ("kg_ms", "wiki_total_ms", "web_ms",
                                      "dedup_ms", "rerank_ms"))
    hybrid_retrieve.last_timing = timing   # type: ignore[attr-defined]

    # KG 事实置顶（强先验，不参与重排）；再拼向量/Web 结果
    return kg_docs + merged[:final_k]


def build_default_reranker(model_name: str = "BAAI/bge-reranker-v2-m3",
                           use_fp16: bool = True):
    """返回一个基于 BGE Reranker（cross-encoder）的 ``rerank_fn(query, docs)``。

    特点
    ----
    - **强**：cross-encoder 联合建模 query-doc，效果通常明显好于 RRF
    - **慢**：一个 pair 一次前向，GPU 上 3-5 pairs × 512 tokens 约 80-150 ms
    - **依赖模型 + 显存**：需要 GPU / 570MB 权重
    """
    from FlagEmbedding import FlagReranker
    reranker = FlagReranker(model_name, use_fp16=use_fp16)

    def _rerank(query: str, docs: List[Dict]) -> List[Dict]:
        if not docs:
            return docs
        pairs = [[query, d["text"]] for d in docs]
        scores = reranker.compute_score(pairs, normalize=True)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)

    return _rerank


def build_rrf_reranker(k: int = 60):
    """返回一个基于 **Reciprocal Rank Fusion (RRF)** 的 ``rerank_fn``。

    算法
    ----
    对每篇候选文档 d：

        RRF_score(d) = sum_over_sources( 1 / (k + rank_in_source(d)) )

    - ``rank_in_source(d)``：文档在其原始来源（wiki / web / …）里的名次（1-based）
    - ``k``：平滑常数，Cormack 2009 论文的经典取值 60；k 越大不同源的贡献越均衡

    特点
    ----
    - **零依赖、零延迟**：纯 Python 加法，< 1 ms
    - **无监督**：不用任何模型，天然适合融合"分数尺度不可比"的多路召回
      （FAISS 的 cosine 和 BM25、web 搜索给的 popularity score 完全不在一个量纲）
    - **效果稳定**：一般比单路好，但**弱于 cross-encoder rerank**
    - **要求**：每篇候选必须有 ``_rank_in_source`` 字段（``hybrid_retrieve`` 内部
      在合并前会自动打上）；若字段缺失则回退到按 ``score`` 降序的相对名次

    什么时候用 RRF vs BGE
    --------------------
    - ✅ **只有向量单路 + Web 单路**（分数不可比）→ RRF 首选
    - ✅ **延迟敏感、无 GPU**                    → RRF
    - ✅ **候选很多（几十条以上）**              → RRF（BGE 会很慢）
    - ❌ **精度优先，愿意花 100 ms**             → BGE cross-encoder
    - 💡 **两阶段**：先 RRF 粗融合 top-N，再 BGE 精排 top-K，速度和精度兼顾
    """
    def _rrf(query: str, docs: List[Dict]) -> List[Dict]:
        if not docs:
            return docs
        # 若某些 doc 没带 _rank_in_source（例如上游没打），按各自 source 内的 score
        # 排一遍作为兜底 rank。
        by_source: Dict[str, List[Dict]] = {}
        for d in docs:
            by_source.setdefault(d.get("source", "_"), []).append(d)
        for src, lst in by_source.items():
            if any("_rank_in_source" not in d for d in lst):
                lst_sorted = sorted(lst, key=lambda x: x.get("score", 0.0),
                                    reverse=True)
                for rank, d in enumerate(lst_sorted, start=1):
                    d["_rank_in_source"] = rank

        # 累加 RRF 分数（同一 doc 若跨源命中会加多次）
        for d in docs:
            rank = d["_rank_in_source"]
            d["rerank_score"] = d.get("rerank_score", 0.0) + 1.0 / (k + rank)

        return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)

    _rrf.__name__ = f"rrf_reranker(k={k})"
    return _rrf


def build_cascade_reranker(coarse_k: int = 20,
                           rrf_k: int = 60,
                           bge_model_name: str = "BAAI/bge-reranker-v2-m3",
                           use_fp16: bool = True):
    """返回一个**两阶段级联**的 ``rerank_fn``：先 RRF 粗排，再 BGE 精排。

    流程
    ----
        docs ──RRF(zero-cost)──▶ top-`coarse_k` ──BGE cross-encoder──▶ final

    - **第 1 级 RRF**：几乎 0 延迟，把候选从 N 缩到 ``coarse_k``
    - **第 2 级 BGE**：只对 top-``coarse_k`` 跑 cross-encoder，控制精排开销

    什么时候用
    ---------
    - **候选数量 N 较多**（例如接了 web 搜索后每次有 10-50 条）
      → 直接跑 BGE 太慢；纯 RRF 又不够准；级联既能压延迟又能保精度
    - **候选数量 ≤ 5** → 用不上级联，直接 BGE 更简单

    延迟估算
    -------
    - N=30, coarse_k=10：RRF < 1 ms + BGE ~200 ms ≈ 200 ms
    - 对比：纯 BGE 跑 N=30 ≈ 600 ms；纯 RRF ≈ 1 ms

    Args:
        coarse_k:        RRF 粗排后送入 BGE 的候选数上限
        rrf_k:           RRF 的平滑常数
        bge_model_name:  BGE Reranker 权重名
        use_fp16:        BGE 是否 fp16
    """
    rrf_fn = build_rrf_reranker(k=rrf_k)
    bge_fn = build_default_reranker(model_name=bge_model_name, use_fp16=use_fp16)

    def _cascade(query: str, docs: List[Dict]) -> List[Dict]:
        if not docs:
            return docs
        # 阶段 1：RRF 粗排，只保留 top-coarse_k
        coarse = rrf_fn(query, docs)[:coarse_k]
        # 阶段 2：BGE 精排（覆写 rerank_score 为 cross-encoder 输出）
        return bge_fn(query, coarse)

    _cascade.__name__ = f"cascade_reranker(coarse_k={coarse_k},rrf_k={rrf_k})"
    return _cascade


def build_reranker(strategy: str = "bge", **kwargs):
    """统一入口：按名字构造 rerank_fn。

    Args:
        strategy: "bge" | "rrf" | "cascade" | "none"
        kwargs:   透传给具体构造器
                  - bge:      model_name, use_fp16
                  - rrf:      k
                  - cascade:  coarse_k, rrf_k, bge_model_name, use_fp16
    """
    strategy = (strategy or "").lower()
    if strategy in ("none", "off", ""):
        return None
    if strategy == "rrf":
        return build_rrf_reranker(**kwargs)
    if strategy == "cascade":
        return build_cascade_reranker(**kwargs)
    if strategy in ("bge", "cross", "cross-encoder"):
        return build_default_reranker(**kwargs)
    raise ValueError(f"unknown rerank strategy: {strategy!r}")
