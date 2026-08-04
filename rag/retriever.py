# rag/retriever.py
"""LayeredRetriever：整个方案 C 的对外主入口。

流程：
    1) L1 QACache 短路 —— 命中直接返回 cache_answer
    2) Router 决定要激活的离线层集合（L2/L3/L5，可选 L4）
    3) 并行调用各层 search()，每层返回自己 top-k
    4) **计算校准聚合置信度**，不达标 → 追加 L4 兜底（P0-2）
    5) RRF 融合 + 可选 rerank → FUSION_TOP_K
    6) 组装 RetrievalResult 返回（含 confidence / low_evidence / web_fallback）

用法：
    retriever = LayeredRetriever(qa_cache=agent.qa_cache)
    result = retriever.retrieve("量子计算是什么", namespace="user:42")
    if result.cache_hit:
        return result.cache_answer
    # P0-4：主链路用 evidence.build_evidence_block 而非 as_context_block
    block, sources = build_evidence_block(result.passages)
    ...
    # 回答成功后：
    retriever.archive(query, answer, sources, namespace="user:42")

P0 改造要点
-----------
* **P0-2**：第 4 步的兜底判定从「max(各层原始 score) < 0.55」改为
  「校准聚合置信度 < WEB_FALLBACK_CONFIDENCE」。原实现有两个 bug
  （L5 的 `or 0.9` + 跨量纲比较）导致 L4 几乎永不触发，详见
  `rag/calibration.py` 顶部说明。
* **P0-3**：`retrieve()` / `archive()` 新增 `namespace` 参数，
  透传给 L1（QACache key 前缀）与 L3（metadata 后过滤），
  实现多租户隔离。namespace=None 时行为与改造前完全一致。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Optional

from . import answerability as rag_answerability
from . import calibration as rag_calibration
from . import config as rag_config
from .embedder import Embedder
from .fusion import rrf_fuse, rerank
from .incremental_worker import ArchiveEvent, IncrementalWorker
from .layers import (
    L1QACacheLayer, L2CommonsenseLayer, L3HistoryLayer,
    L4WebLayer, L5KGLayer,
)
from .router import route, should_abstain, should_fallback_to_web
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

        # 线程池容量说明：
        #   激活层最多 4 个（L2/L3/L4/L5），再加 1 个给"L4 兜底"这条
        #   在主链路之外提交的任务 → 至少要 5 个 worker。
        #   但**超时被放弃的任务仍会占着 worker 把自己跑完**
        #   （ThreadPoolExecutor 无法中断已启动的任务），所以在超时场景下
        #   5 个 worker 可能全被上一轮的僵尸 L4 请求占住，导致下一轮请求
        #   连离线层都排不上队 —— 表现为"越超时越慢"的雪崩。
        #   给到 12 个：留出 2 轮以上的缓冲，代价只是几个空闲线程。
        self._pool = ThreadPoolExecutor(max_workers=12, thread_name_prefix="rag_layer")

    # ---------- 主接口 ---------- #
    def retrieve(
        self,
        query: str,
        namespace: Optional[str] = None,
        route_query: Optional[str] = None,
    ) -> RetrievalResult:
        """执行分层检索。

        Args:
            query:     检索 query（一般是改写后的）。
            namespace: P0-3 多租户隔离命名空间。
                       透传给 L1（QACache key 前缀隔离）与
                       L3（metadata 后过滤），保证不跨用户串味。
                       None → 全局共享（与改造前行为完全一致）。
            route_query: **用于层激活决策**的 query，默认同 `query`。

                ══════════════════════════════════════════════════════
                为什么要把"检索用的 query"和"路由用的 query"分开
                ══════════════════════════════════════════════════════
                这是本次性能优化定位到的**首要瓶颈**。实测链路：

                  用户问   : "美国一共多少位副总统 历史上"
                  改写后   : "美国历史上共有多少位副总统 2026年"
                                                        ↑↑↑↑↑
                  rewriter 的 prompt 原本写着"加入年份…以提高召回"，
                  于是它给一个**历史累计型**问题凭空加上了当前年份。

                后果是一条完整的因果链：
                  ① `cache_policy.is_time_sensitive()` 把「当前年份 ±1」
                     当作强时效信号（这个设计本身是对的）；
                  ② 于是改写后的 query 被判为时效敏感 → `route()`
                     **无条件叠加 L4_web**；
                  ③ L4 走 DuckDuckGo，实测**未命中缓存时 16~38 秒**
                     （两次实测：37957ms / 16412ms）；
                  ④ `_parallel_search` 用 `as_completed` 等**全部**层，
                     所以整个检索被这一层拖到 21 秒。

                而实测离线三层（L2+L3+L5）单独跑：
                     L2 65ms + L3 66ms + L5 434ms  →  聚合置信度 0.9899
                  `should_fallback_to_web(0.9899)` = False
                  —— **本来根本不需要联网**，这 16~38 秒是纯粹的浪费。

                【修法】用**用户的原始 query** 做时效判定。
                理由：时效性是"用户想问什么"的属性，
                而不是"改写器写了什么"的属性。改写是为了提高召回的
                手段，让手段反过来改变意图判定，就是典型的
                **抽象泄漏（leaky abstraction）**。
                原始 query "美国一共多少位副总统 历史上" 的
                `is_time_sensitive()` = False（已实测验证），
                于是 L4 不再被强制激活，只在离线证据确实不足时才兜底。

                向后兼容：不传 route_query 时退化为原行为。
        """
        # 路由决策用原始 query（若未提供则退回改写后的 query）
        route_q = route_query or query

        # ---------- 1) L1 短路 ----------
        if self.l1 is not None:
            ans = self.l1.lookup(query, namespace=namespace)
            if ans is not None:
                return RetrievalResult(
                    query=query, cache_hit=True, cache_answer=ans,
                    passages=[Passage(
                        text=ans, layer="L1_qa", score=1.0,
                        title="QA Cache Hit",
                        metadata={"calibrated": rag_calibration.calibrate("L1_qa", 1.0)},
                    )],
                    layer_hits={"L1_qa": 1},
                    # L1 命中意味着已通过精确匹配或「0.93 阈值 + 槽位门禁」，
                    # 是全系统最高可信路径。
                    confidence=rag_calibration.calibrate("L1_qa", 1.0),
                    low_evidence=False,
                )

        # ---------- 2) Router 决定激活哪些层 ----------
        # ⚠️ 用 route_q（原始 query）而非 query（改写后）：见上方 route_query 说明。
        active: list[LayerName] = route(route_q, self.strategy)

        # ---------- 3) 并行召回 ----------
        per_layer: dict[LayerName, list[Passage]] = self._parallel_search(
            query, active, namespace=namespace
        )

        # ══════════════════════════════════════════════════════════════════
        # 4) L4 兜底判定（P0-2 的核心改动）
        # ══════════════════════════════════════════════════════════════════
        # 【改造前】
        #     offline_best = max(p.score for 非 L4 层的所有 p)
        #     if offline_best < 0.55: 补 L4
        #
        #   两个致命问题：
        #   ① `rag/layers.py` 的 L5 写了 `score=float(d.get("score") or 0.9)`，
        #      而 `traverse_multi_hop` 给多跳实体写死 score=0.0，
        #      `0.0 or 0.9 == 0.9` → 所有多跳实体分数被抬到 0.9
        #      → offline_best 恒 ≥ 0.9 > 0.55 → **L4 永不触发**。
        #   ② 即使没有 bug ①，用一个 0.55 的标量阈值去比较
        #      「L2 的 BGE 余弦」「L4 的位次衰减」「L5 的人工混合分」
        #      这三种完全不同量纲的数，在统计上也是没有意义的。
        #
        # 【改造后】
        #   各层原始分先经 calibration 映射到统一的 P(relevant)，
        #   再用噪声-OR 聚合 top-3 得到 offline_conf，然后与
        #   WEB_FALLBACK_CONFIDENCE 比较。这样：
        #     - "弱 KG 命中"（校准后 ≈0.11）不会再阻止 L4；
        #     - "L2 强命中"（校准后 ≈0.88）会正确跳过 L4，省下 1-3s；
        #     - 多路一致时置信度自然更高（噪声-OR 的性质）。
        # ══════════════════════════════════════════════════════════════════
        offline_layers = {k: v for k, v in per_layer.items() if k != "L4_web"}
        offline_conf = rag_calibration.aggregate_confidence(offline_layers)

        # ══════════════════════════════════════════════════════════════════
        # Stage-1 修复①：置信度**不足以**判断"能不能回答"
        # ══════════════════════════════════════════════════════════════════
        # 上面的 offline_conf 衡量的是**语义相似度**（"召回的东西像不像这个
        # 话题"），但 L4 兜底真正要判断的是**充分性**（"够不够回答问题"）。
        # 这两者在简单事实题上一致，在多跳/聚合题上完全背离。实测：
        #
        #   「小丑鱼 外来物种 USGS 邮编」→ conf 0.9800，证据是"小丑鱼是热带鱼"
        #   「茅盾文学奖 历届 获奖名单」→ conf 0.9286，证据是"1982年首届"
        #   「日本 今年 地震次数」      → conf 0.9839，证据是"2018年大阪地震"
        #
        # 三条证据都**主题相关但不含答案**，却因 conf ≥ 0.93 而
        # `should_fallback_to_web()` = False → **L4 永不触发** → 只能拒答。
        # 根因是噪声-OR 只聚合 top-3，实测「8 条余弦 0.55 的证据」就能
        # 堆出 conf 0.875 —— 有 3 条沾边的就饱和了。
        #
        # 【修法】加一个与相似度**正交**的信号：query 实词覆盖率。
        # 若"邮政编码""获奖名单""次数"在所有证据里一次都没出现，
        # 那无论语义多相似，答案都不可能在里面。实测分离度很干净：
        #   失败案例 0.20 / 0.25 / 0.50   vs   正常案例 1.00
        #
        # 两个信号取 **OR**（而非 AND）：它们各自捕捉一类失败模式 ——
        #   conf 低 + 覆盖高 → 沾了词但语义弱
        #   conf 高 + 覆盖低 → **本次修的正是这类**
        # 用 AND 会让任一信号失效就整体失效，等于没加。
        # 详细标定数据与已知局限见 `rag/answerability.py` 顶部。
        #
        # ⚠️ 用 `query`（改写后）而非 `route_q`（原始）做覆盖率判定：
        # 这里问的是"检索回来的东西够不够"，而检索用的就是改写后的 query，
        # 口径必须一致。这与上面"层激活用原始 query"不矛盾 ——
        # 层激活判的是**用户意图**（时效性），覆盖率判的是**检索结果**。
        insufficient, coverage, missing_terms = (
            rag_answerability.is_evidence_insufficient(query, offline_passages)
            if (offline_passages := [p for v in offline_layers.values() for p in v])
            else (True, 0.0, set())
        )

        low_conf = should_fallback_to_web(offline_conf)
        need_web = low_conf or insufficient

        web_fallback = False
        if (
            self.l4 is not None
            and "L4_web" not in per_layer          # Router 没有已激活 L4
            and need_web
        ):
            # 日志把**触发原因**打清楚：上线后据此区分
            #   "离线库覆盖不够"（coverage 低）与 "检索质量差"（conf 低），
            # 两者的优化方向完全不同（补索引 vs 调检索）。
            why = []
            if low_conf:
                why.append(f"conf={offline_conf:.3f}<{rag_config.WEB_FALLBACK_CONFIDENCE}")
            if insufficient:
                why.append(
                    f"实词覆盖率={coverage:.2f}"
                    f"<{rag_answerability.MIN_TERM_COVERAGE}"
                    f"（缺失 {sorted(missing_terms)[:6]}）"
                )
            print(f"[retriever] 🔍 触发 L4 兜底：{' 且 '.join(why)}")
            # ⚡ 兜底的 L4 也要走延迟预算。
            # 这条路径原本是**直接同步调用** `_safe_search`，完全没有超时保护
            # —— 实测 DDG 未命中缓存时要 16~38 秒（最坏情况因内部
            # 3 次重试 × 15s 超时而远超一分钟），足以让用户以为程序卡死。
            # 复用线程池 + `future.result(timeout=...)`：到点就放弃这一路，
            # 用已有的离线证据作答（此时 low_evidence 会如实标记证据不足，
            # summary prompt 会引导模型承认信息有限，不会编造）。
            _fut = self._pool.submit(
                self._safe_search, self.l4, query, rag_config.L4_TOP_K
            )
            try:
                per_layer["L4_web"] = _fut.result(
                    timeout=rag_config.L4_TIMEOUT_SEC
                )
                web_fallback = True
            except FuturesTimeout:
                # ══════════════════════════════════════════════════════════
                # Stage-1 修复③：超时改为「软放弃」——晚到的结果仍然收下
                # ══════════════════════════════════════════════════════════
                # 【改造前的浪费】
                #   到点直接放弃，`_fut` 的结果被永久丢弃。实测日志：
                #       [retriever] ⏱️ 层检索超预算 8.0s，放弃 ['L4_web']
                #       [searcher] DDG 命中 5 条        ← 紧接着就回来了！
                #   L4 只超时 0.7 秒，DDG 真的召回了 5 条**有用**证据，
                #   却因为"过了截止线"被扔掉 —— 这一题本来是唯一有机会
                #   答对的。既付了 8 秒的等待成本，又没拿到任何收益，
                #   是最坏的一种结果。
                #
                # 【为什么会这样】
                #   ThreadPoolExecutor 无法中断已启动的任务，所以超时后
                #   那个线程**必然会跑完**并把结果存进 future。既然成本
                #   已经付掉了，就没有理由不去取回来。
                #
                # 【软放弃】
                #   给一个很短的"宽限期"（grace period）再看一眼：
                #     * 已经完成 → 直接收下，白赚一路优质证据
                #     * 仍未完成 → 才真正放弃，降级作答
                #   宽限期设得很小（默认 1.5s），所以最坏情况只多等这么久，
                #   前台延迟依然可预测；而收益是把"差一点就成功"的请求救回来。
                #
                #   这是 Bing / Perplexity 的常见做法：deadline 不是一刀切的
                #   硬截断，而是「主预算 + 短宽限」两段式，兼顾尾延迟与召回率。
                grace = rag_config.L4_GRACE_SEC
                try:
                    per_layer["L4_web"] = _fut.result(timeout=grace)
                    web_fallback = True
                    print(
                        f"[retriever] ⏱️→✓ L4 超主预算 "
                        f"{rag_config.L4_TIMEOUT_SEC:.1f}s，但在 {grace:.1f}s "
                        f"宽限期内返回，已收下 "
                        f"{len(per_layer['L4_web'])} 条（软放弃机制）"
                    )
                except FuturesTimeout:
                    print(
                        f"[retriever] ⏱️ L4 web 兜底超预算 "
                        f"{rag_config.L4_TIMEOUT_SEC:.1f}s"
                        f"+{grace:.1f}s 宽限，放弃联网，"
                        f"仅用离线证据作答（offline_conf={offline_conf:.3f}）"
                    )
                except Exception as e:
                    print(f"[retriever] L4 web 兜底异常（宽限期内）: {e}")
            except Exception as e:
                print(f"[retriever] L4 web 兜底异常: {e}")

        # 补完 L4 后重算整体置信度（此时才是"本轮全部证据"的置信度）
        confidence = rag_calibration.aggregate_confidence(per_layer)

        # ---------- 5) 融合 + rerank ----------
        fused = rrf_fuse(list(per_layer.values()), top_k=self.fusion_top_k)
        fused = rerank(query, fused, top_k=self.fusion_top_k)
        # ⚠️ 必须 overwrite=False。
        # `rrf_fuse` 会把 `score` 替换成 RRF 贡献值 Σ1/(60+rank)（典型 0.016~0.03），
        # 那是**排序用的秩倒数，不是相似度**。若在这里重新校准，
        # 相当于拿 0.016 去过 sigmoid，所有 conf 都会被压成 0.00 —— 明明
        # 检索质量很好，来源面板却全显示"置信度 0.00"。
        # 各层的 calibrated 已在层内（score 还是原始分时）算好并随
        # metadata 浅拷贝带过来了，这里只给"万一漏掉的"补算。
        rag_calibration.calibrate_passages(fused, overwrite=False)

        return RetrievalResult(
            query=query,
            passages=fused,
            layer_hits={k: len(v) for k, v in per_layer.items()},
            cache_hit=False,
            confidence=confidence,
            # abstention 信号：即使补了 L4 置信度仍极低 → 让模型明确说资料不足，
            # 而不是基于无关资料硬编答案（幻觉的主要来源之一）。
            low_evidence=should_abstain(confidence),
            web_fallback=web_fallback,
            # Stage-1：把可答性信号也带出去，供 agent 记入 trace / 前端展示。
            # 注意这里报的是**离线证据**的覆盖率（补 L4 之前算的）——
            # 它回答的是"离线库够不够用"，是索引覆盖度的直接指标。
            term_coverage=coverage,
            missing_terms=sorted(missing_terms),
        )

    def archive(
        self,
        query: str,
        answer: str,
        sources: Optional[list[dict]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """由 agent 在成功回答后调用；异步入 L3。

        Args:
            namespace: P0-3。写入 L3 时打上租户标记，
                       检索时 `L3HistoryLayer.search()` 会据此过滤，
                       避免 A 用户的历史问答出现在 B 用户的资料里。
        """
        if self._incr is None:
            return
        self._incr.submit(ArchiveEvent(
            query=query, answer=answer, sources=sources, namespace=namespace,
        ))

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
        self,
        query: str,
        active: list[LayerName],
        namespace: Optional[str] = None,
    ) -> dict[LayerName, list[Passage]]:
        """并行调用各激活层的 search()，带**延迟预算（deadline）**。

        P0-3：L3 需要 namespace 参数做后过滤，而 L2/L4/L5 不需要
        （它们是全局共享的公共知识，不含用户私有数据）。
        因此这里对 L3 做特殊分发，避免给所有层都加一个用不上的参数。

        ══════════════════════════════════════════════════════════════════
        ⚡ 延迟预算：慢层不能拖垮整条链路
        ══════════════════════════════════════════════════════════════════
        【改造前】`as_completed(futures)` 不带 timeout，无条件等**所有**层
        返回。于是整层耗时 = max(各层耗时)，任何一层退化就是全链路退化。

        实测各层耗时（本机，预热后）：
            L2 wiki  65ms | L3 history 66ms | L5 kg 434ms
            L4 web   16412ms ~ 37957ms          ← 慢 2~3 个数量级
        用户观测到的"分层 RAG 检索 21s"，几乎全是在等 L4 这一路。

        【改造后】给 `as_completed` 传 timeout（见 rag/config 的
        LAYER_TIMEOUT_SEC / L4_TIMEOUT_SEC）。超时后：
          * 已完成的层 → 正常收集
          * 未完成的层 → 记为空结果，并打一行 warn 供观测
        融合与置信度计算照常进行，只是少了一路召回 —— 这是**优雅降级**，
        而不是整个请求失败。

        为什么不给 future 发 cancel：Python 的 ThreadPoolExecutor 无法
        中断已经开始执行的任务（没有线程 kill 语义）。超时的线程会继续
        在后台跑完然后被丢弃 —— 浪费一点后台资源，但换来了**可预测的
        前台延迟**，这个交换在交互式产品里是完全值得的。
        （真要彻底可中断，需要把 L4 换成 asyncio + 支持 cancel 的
         HTTP client，属于后续优化项。）
        """
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
            if name == "L3_history":
                # L3 存的是用户私有历史 → 必须按 namespace 隔离
                futures[self._pool.submit(
                    self._safe_search, layer, query, k, namespace
                )] = name
            else:
                # L2/L4/L5 是公共知识，无租户概念
                futures[self._pool.submit(self._safe_search, layer, query, k)] = name

        # 本轮预算 = 各激活层预算的最大值。
        # L4 单独给更长的预算（联网本就慢），但只有它真被激活时才生效 ——
        # 否则纯离线检索会白等 8s 的上限（虽然通常提前返回，但一旦某个
        # 离线层卡住，用户就要多等 3s 才降级）。
        budget = rag_config.LAYER_TIMEOUT_SEC
        if "L4_web" in futures.values():
            budget = max(budget, rag_config.L4_TIMEOUT_SEC)

        out: dict[LayerName, list[Passage]] = {}
        try:
            for fut in as_completed(futures, timeout=budget):
                name = futures[fut]
                try:
                    out[name] = fut.result()
                except Exception as e:
                    print(f"[retriever] {name} 层异常: {e}")
                    out[name] = []
        except FuturesTimeout:
            # ══════════════════════════════════════════════════════════════
            # Stage-1 修复③：预算耗尽时给一个**宽限期**，别浪费已付的成本
            # ══════════════════════════════════════════════════════════════
            # 用户实测日志里最刺眼的一幕：
            #     [retriever] ⏱️ 层检索超预算 8.0s，放弃 ['L4_web']
            #     [searcher] DDG 命中 5 条          ← 0.7 秒后就回来了
            # 只差 0.7 秒，5 条真实有效的 web 证据被丢弃，那一题就此答不出来。
            #
            # 由于 ThreadPoolExecutor 不能中断已启动的任务，这些线程**一定会
            # 跑完**——成本已经付了。所以到点后再快速看一眼（短宽限期）：
            # 已经完成的收下，仍未完成的才真正放弃。
            #
            # 用 `fut.done()` + `result(timeout=0)` 而不是再 `as_completed`：
            #   这里只想"顺手捡走已经躺在那儿的结果"，不想再阻塞等待。
            #   done() 为 True 时 result(timeout=0) 保证不会阻塞。
            # 之后才对仍未完成的层做一次极短的统一等待（grace）。
            missing = [n for f, n in futures.items() if n not in out]
            grace = rag_config.LAYER_GRACE_SEC
            rescued: list[str] = []
            if grace > 0 and missing:
                import time as _t
                deadline = _t.perf_counter() + grace
                for fut, name in futures.items():
                    if name in out:
                        continue
                    remain = deadline - _t.perf_counter()
                    if remain <= 0:
                        break
                    try:
                        out[name] = fut.result(timeout=remain)
                        rescued.append(name)
                    except (FuturesTimeout, Exception):
                        # 宽限期内仍没回来（或本身报错）→ 按放弃处理，
                        # 下面统一补空列表
                        pass
            still_missing = [n for f, n in futures.items() if n not in out]
            if rescued:
                print(
                    f"[retriever] ⏱️→✓ 超预算 {budget:.1f}s 后，在 "
                    f"{grace:.1f}s 宽限期内救回 {rescued}（软放弃机制）"
                )
            if still_missing:
                # 这行日志很重要 —— 上线后它突然变多，说明某个数据源在退化。
                print(
                    f"[retriever] ⏱️ 层检索超预算 {budget:.1f}s"
                    f"+{grace:.1f}s 宽限，放弃未返回的层 {still_missing}"
                    f"（已用 {sorted(out)} 的结果降级作答）"
                )
            for n in still_missing:
                out[n] = []
        return out

    @staticmethod
    def _safe_search(
        layer, query: str, top_k: int, namespace: Optional[str] = None,
    ) -> list[Passage]:
        """调用某层 search()，异常时返回空列表（单层故障不拖垮整体）。

        namespace 只在传入时才透传，保持对「不支持 namespace 的层」
        （L2/L4/L5）的签名兼容。
        """
        try:
            if namespace is not None:
                return layer.search(query, top_k=top_k, namespace=namespace)
            return layer.search(query, top_k=top_k)
        except TypeError:
            # 该层不接受 namespace 参数 → 退回不带 namespace 的调用。
            # 这条分支保证了未来新增自定义层时不会因签名不匹配而整层失效。
            try:
                return layer.search(query, top_k=top_k)
            except Exception as e:
                print(f"[retriever] {getattr(layer, 'name', layer)} search 异常: {e}")
                return []
        except Exception as e:
            print(f"[retriever] {getattr(layer, 'name', layer)} search 异常: {e}")
            return []
