# rag/retriever.py
"""LayeredRetriever：整个方案 C 的对外主入口。

流程：
    1) L1 QACache 短路 —— 命中直接返回 cache_answer
    2) Router 决定要激活的离线层集合（L2/L3/L5，可选 L4）
    3) 并行调用各层 search()，每层返回自己 top-k
    4) 检查最高分，若不达标 → 追加 L4 兜底
    5) RRF 融合 + 可选 rerank → FUSION_TOP_K
    6) 组装 RetrievalResult 返回

用法：
    retriever = LayeredRetriever(qa_cache=agent.qa_cache)
    result = retriever.retrieve("量子计算是什么")
    if result.cache_hit:
        return result.cache_answer
    context_block = result.as_context_block()
    ...
    # 回答成功后：
    retriever.archive(query, answer, sources)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from . import config as rag_config
from .embedder import Embedder
from .fusion import rrf_fuse, rerank
from .incremental_worker import ArchiveEvent, IncrementalWorker
from .layers import (
    L1QACacheLayer, L2CommonsenseLayer, L3HistoryLayer,
    L4WebLayer, L5KGLayer,
)
from .router import route, should_fallback_to_web
from .types import LayerName, Passage, RetrievalResult


class LayeredRetriever:
    def __init__(
        self,
        qa_cache=None,
        embedder: Optional[Embedder] = None,
        enable_l2: bool = True,
        enable_l3: bool = True,
        enable_l4: bool = True,
        enable_l5: bool = True,
        strategy: Optional[str] = None,
        fusion_top_k: int = rag_config.FUSION_TOP_K,
    ):
        # 统一 Embedder（BGE-M3）：L3 历史层直接使用；L2/L5 各自内部持有
        # wiki_rag 的 BGE-M3，不依赖这个实例。
        self.embedder = embedder or Embedder()
        self.strategy = strategy or rag_config.ROUTER_STRATEGY
        self.fusion_top_k = fusion_top_k

        # L1
        self.l1: Optional[L1QACacheLayer] = (
            L1QACacheLayer(qa_cache) if qa_cache is not None else None
        )
        # L2（内部自带 BGE-M3 编码器，无需外部 embedder）
        self.l2 = L2CommonsenseLayer() if enable_l2 else None
        # L3（使用统一 embedder）
        self.l3 = L3HistoryLayer(self.embedder) if enable_l3 else None
        # L5（内部自带 KG 检索 + BGE-M3，无需外部 embedder）
        self.l5 = L5KGLayer() if enable_l5 else None
        # L4
        self.l4 = L4WebLayer() if enable_l4 else None

        # 增量 worker（消费 L3 归档事件）
        self._incr: Optional[IncrementalWorker] = (
            IncrementalWorker(self.l3) if self.l3 is not None else None
        )

        self._pool = ThreadPoolExecutor(max_workers=5, thread_name_prefix="rag_layer")

    # ---------- 主接口 ---------- #
    def retrieve(self, query: str) -> RetrievalResult:
        # 1) L1 短路
        if self.l1 is not None:
            ans = self.l1.lookup(query)
            if ans is not None:
                return RetrievalResult(
                    query=query, cache_hit=True, cache_answer=ans,
                    passages=[Passage(text=ans, layer="L1_qa", score=1.0,
                                      title="QA Cache Hit")],
                    layer_hits={"L1_qa": 1},
                )

        # 2) Router
        active: list[LayerName] = route(query, self.strategy)

        # 3) 并行召回
        per_layer: dict[LayerName, list[Passage]] = self._parallel_search(query, active)

        # 4) 兜底 L4
        offline_best = max(
            (p.score for name, ps in per_layer.items()
             for p in ps if name != "L4_web"),
            default=0.0,
        )
        if self.l4 is not None and "L4_web" not in per_layer and should_fallback_to_web(offline_best):
            web = self._safe_search(self.l4, query, rag_config.L4_TOP_K)
            per_layer["L4_web"] = web

        # 5) 融合 + rerank
        fused = rrf_fuse(list(per_layer.values()), top_k=self.fusion_top_k)
        fused = rerank(query, fused, top_k=self.fusion_top_k)

        return RetrievalResult(
            query=query,
            passages=fused,
            layer_hits={k: len(v) for k, v in per_layer.items()},
            cache_hit=False,
        )

    def archive(self, query: str, answer: str, sources: Optional[list[dict]] = None) -> None:
        """由 agent 在成功回答后调用；异步入 L3。"""
        if self._incr is None:
            return
        self._incr.submit(ArchiveEvent(query=query, answer=answer, sources=sources))

    # ---------- 预热（warmup） ---------- #
    def warmup(self, probe_query: str = "预热", verbose: bool = True) -> dict:
        """一次性把所有"懒加载"的重资源拉起来，消除首查询延迟毛刺。

        背景：为了让 agent 启动快、按需付费，L2/L3/L5 及二阶 reranker 都做了
        **懒加载**——真正的大文件（GB 级 FAISS 索引、SQLite KG、BGE-M3 权重、
        热门实体 mmap）都推迟到"首次 search"时才加载。代价是**第一条**真实
        查询会额外多花 3-8s（模型加载 + kernel 编译 + mmap 首次缺页）。

        本方法在服务/CLI 就绪前主动把这些资源全部 touch 一遍，使**首查询延迟**
        与**稳态延迟**几乎一致（对应 rag/scripts/07_demo_retrieve.py 里的 warmup_all）。

        各组件独立 try/except：某层没构建索引（如未跑离线 pipeline）时预热失败
        不致命，只跳过该层，不影响其余层与整体启动。

        Args:
            probe_query: 预热用的探针 query；传一个典型业务问题可让底层按真实
                         shape 触发 kernel 编译，预热更充分。
            verbose:     是否打印每一步耗时。
        Returns:
            每个组件的耗时（秒）字典，便于记录到日志/监控。
        """
        import time as _time
        timing: dict[str, float] = {}
        t_all = _time.perf_counter()
        if verbose:
            print(f"[warmup] start (probe={probe_query!r}) ...")

        # L3 历史层用的统一 Embedder（BGE-M3）：最先预热，L2/L5 各自内部也持有
        # 自己的 BGE-M3，但这里先把共享 embedder 的权重/kernel 拉起来。
        t0 = _time.perf_counter()
        try:
            self.embedder.embed(probe_query)
        except Exception as e:
            print(f"[warmup]   embedder warmup skipped: {e}")
        timing["embedder"] = _time.perf_counter() - t0
        if verbose:
            print(f"[warmup]   embedder: {timing['embedder']:.2f} s")

        # L2 Wiki：加载 GB 级 FAISS 索引 + BGE-M3（层自带 warmup）
        if self.l2 is not None:
            t0 = _time.perf_counter()
            try:
                self.l2.warmup()
            except Exception as e:
                print(f"[warmup]   L2(wiki) warmup skipped: {e}")
            timing["l2_wiki"] = _time.perf_counter() - t0
            if verbose:
                print(f"[warmup]   L2 wiki: {timing['l2_wiki']:.2f} s")

        # L5 KG：打开 SQLite + 触发 Linker 热门实体向量 mmap（层自带 warmup）
        if self.l5 is not None:
            t0 = _time.perf_counter()
            try:
                self.l5.warmup()
            except Exception as e:
                print(f"[warmup]   L5(kg) warmup skipped: {e}")
            timing["l5_kg"] = _time.perf_counter() - t0
            if verbose:
                print(f"[warmup]   L5 kg:   {timing['l5_kg']:.2f} s")

        # 二阶 reranker：仅当 RERANK_STRATEGY=bge/cascade 时才有真实模型；
        # 跑一次 rerank 触发 cross-encoder 权重加载（rrf/none 为空操作，零开销）。
        t0 = _time.perf_counter()
        try:
            from .types import Passage as _P
            _dummy = [_P(text=probe_query, title="", layer="L2_wiki", score=1.0)]
            rerank(probe_query, _dummy, top_k=1)
        except Exception as e:
            print(f"[warmup]   rerank warmup skipped: {e}")
        timing["rerank"] = _time.perf_counter() - t0
        if verbose:
            print(f"[warmup]   rerank:  {timing['rerank']:.2f} s")

        timing["total"] = _time.perf_counter() - t_all
        if verbose:
            print(f"[warmup] done in {timing['total']:.2f} s")
        return timing

    def close(self):
        if self._incr is not None:
            self._incr.shutdown(wait=True)
        self._pool.shutdown(wait=False)

    # ---------- 内部 ---------- #
    def _parallel_search(
        self, query: str, active: list[LayerName],
    ) -> dict[LayerName, list[Passage]]:
        layer_map = {
            "L2_wiki":    (self.l2, rag_config.L2_TOP_K),
            "L3_history": (self.l3, rag_config.L3_TOP_K),
            "L4_web":     (self.l4, rag_config.L4_TOP_K),
            "L5_kg":      (self.l5, rag_config.L5_TOP_K),
        }
        futures = {}
        for name in active:
            layer, k = layer_map.get(name, (None, 0))
            if layer is None:
                continue
            futures[self._pool.submit(self._safe_search, layer, query, k)] = name
        out: dict[LayerName, list[Passage]] = {}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                out[name] = fut.result()
            except Exception as e:
                print(f"[retriever] {name} 层异常: {e}")
                out[name] = []
        return out

    @staticmethod
    def _safe_search(layer, query: str, top_k: int) -> list[Passage]:
        try:
            return layer.search(query, top_k=top_k)
        except Exception as e:
            print(f"[retriever] {layer.name} search 异常: {e}")
            return []
