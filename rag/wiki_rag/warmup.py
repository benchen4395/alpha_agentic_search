"""统一 warmup：一次把所有会"懒加载"的重资源都拉起来。

背景
----
本项目里有多处**懒加载**（lazy load）：

- :class:`wiki_rag.retriever.WikiRetriever`：``__init__`` 只加载 FAISS + meta，
  BGE-M3 到**首次** ``search()`` 才通过 :func:`embedder.encode` 触发加载（3-8 s）。
- :func:`wiki_rag.hybrid.build_default_reranker`：``FlagReranker(...)`` 构造函数
  返回很快，实际权重也是首次 ``compute_score`` 时才拉进显存。
- :class:`wiki_rag.linker.Linker`：热实体 emb 用 ``np.load(mmap_mode="r")``，
  首次 rerank 时才把页真正 fault 进内存。
- :class:`wiki_rag.kg_retriever.KGRetriever`：走 Linker + KGStore，只要 Linker
  预热了，其自身的开销就只剩 SQLite 首次点查（<10 ms，可忽略）。

因此本模块提供 :func:`warmup_all`，服务启动阶段调一次，就能让**首查询延迟**和
**稳态查询延迟**几乎一致，方便部署（ready 探针、SLA、A/B 对比）。

用法
----
::

    from wiki_rag.retriever import WikiRetriever
    from wiki_rag.kg_retriever import KGRetriever
    from wiki_rag.hybrid import build_default_reranker
    from wiki_rag.warmup import warmup_all

    wiki_retr = WikiRetriever("configs/default.yaml")
    kg_retr   = KGRetriever("configs/default.yaml")
    rerank_fn = build_default_reranker()

    warmup_all(
        wiki_retriever=wiki_retr,
        kg_retriever=kg_retr,
        rerank_fn=rerank_fn,
        probe_query="量子纠缠",   # 用真实业务 query 触发端到端路径
    )
    # 现在服务真正 ready，可以放开 /healthz
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


def _warmup_wiki(wiki_retriever: Any, probe_query: str) -> float:
    """预热 WikiRetriever：加载 BGE-M3 + 触发 CUDA kernel 编译。"""
    t0 = time.perf_counter()
    # 首选：调用检索器自己的 warmup（如果我们已经加上了）
    if hasattr(wiki_retriever, "warmup"):
        wiki_retriever.warmup()
    else:
        # 兜底：直接跑一次 search 触发懒加载
        wiki_retriever.search(probe_query, top_k=1)
    return time.perf_counter() - t0


def _warmup_rerank(rerank_fn: Callable[[str, List[Dict]], List[Dict]],
                   probe_query: str) -> float:
    """预热 Reranker：让 FlagReranker 拉权重到显存 + 编译 kernel。

    关键：用**真实 batch 大小 + 真实 seq_len** 触发 CUDA kernel autotune，
    否则首次真实请求（比如 3 pairs × 512 tokens）会被重新 autotune，多花 1-2 s。
    """
    t0 = time.perf_counter()
    # 用一段接近 max_length 的假文本（Reranker 默认 max_length=512，长文本会被截断，
    # 所以给一大坨中文重复即可覆盖到真实 shape）
    long_text = ("量子纠缠是量子力学中的一种现象。" * 40)   # ~600 字，肯定会撑到 512 tokens
    fake_docs = [{"text": long_text} for _ in range(5)]     # 5 pairs 覆盖常见 top_k
    try:
        rerank_fn(probe_query, fake_docs)
    except Exception as e:
        # 有些 rerank_fn 的实现对文档结构有额外要求，warmup 失败不算致命
        print(f"[warmup] rerank warmup skipped: {e}")
    return time.perf_counter() - t0


def _warmup_kg(kg_retriever: Any, probe_query: str) -> float:
    """预热 KGRetriever + Linker：走一遍端到端路径。

    这里跑一次真正的 ``retrieve()``，会串起：
      - mentions 表精确点查（触发 SQLite page cache 预热）
      - Linker 加载热实体 mmap 页
      - KGStore.to_context 三元组拼装
    """
    t0 = time.perf_counter()
    try:
        _ = kg_retriever.retrieve(probe_query, top_k=1, multi_hop=False)
    except Exception as e:
        print(f"[warmup] kg warmup skipped: {e}")
    return time.perf_counter() - t0


def warmup_all(*,
               wiki_retriever: Any = None,
               kg_retriever: Any = None,
               rerank_fn: Optional[Callable[[str, List[Dict]], List[Dict]]] = None,
               probe_query: str = "预热",
               verbose: bool = True) -> Dict[str, float]:
    """一次性预热所有可选组件。

    组件全部是关键字参数且可选（``None`` 会跳过），方便按部署形态调用。

    Args:
        wiki_retriever: :class:`WikiRetriever` 实例。
        kg_retriever:   :class:`KGRetriever` 实例。
        rerank_fn:      重排 callable，签名 ``(query, docs) -> docs``。
        probe_query:    预热查询串；生产建议传一个典型业务 query，
                        让 CUDA kernel 编译走真实 shape。
        verbose:        打印每一步耗时。

    Returns:
        每一步的耗时（秒）字典，方便记录到监控/日志。
    """
    timing: Dict[str, float] = {}
    t_all = time.perf_counter()

    if verbose:
        print(f"[warmup] start (probe={probe_query!r}) ...")

    if wiki_retriever is not None:
        dt = _warmup_wiki(wiki_retriever, probe_query)
        timing["wiki"] = dt
        if verbose:
            print(f"[warmup]   wiki:   {dt:.2f} s")

    if rerank_fn is not None:
        dt = _warmup_rerank(rerank_fn, probe_query)
        timing["rerank"] = dt
        if verbose:
            print(f"[warmup]   rerank: {dt:.2f} s")

    if kg_retriever is not None:
        dt = _warmup_kg(kg_retriever, probe_query)
        timing["kg"] = dt
        if verbose:
            print(f"[warmup]   kg:     {dt:.2f} s")

    timing["total"] = time.perf_counter() - t_all
    if verbose:
        print(f"[warmup] done in {timing['total']:.2f} s")
    return timing
