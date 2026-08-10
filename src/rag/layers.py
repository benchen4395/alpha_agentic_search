# rag/layers.py
"""L1 ~ L5 五层检索器。每层实现同一接口：

    class Layer:
        name: LayerName
        def search(self, query: str, top_k: int) -> list[Passage]: ...

分工：
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

设计要点
--------
* **跨层分数校准**：每层在产出 Passage 时，除了保留层内原始分 `score`，
  额外把校准后的相关概率写进 `metadata["calibrated"]`
  （见 `rag/calibration.py`）。原始分保留用于层内排序 / debug 溯源；
  跨层比较、L4 兜底判定、整体置信度**一律使用 calibrated**。
* **L5 分数缺省必须区分"无分数"与"分数为 0"**：写成
  `score=float(d.get("score") or 0.9)` 会踩 Python 假值陷阱 ——
  `traverse_multi_hop` 给多跳实体写死 `score=0.0`，而 `0.0 or 0.9 == 0.9`，
  于是所有多跳实体分数被静默抬到 0.9，进而使 `retriever` 的 L4 兜底
  **永不触发**。因此改用 `is None` 判定并按跳数衰减
  （详见 L5KGLayer.search 内注释）。
* **多租户隔离**：L1 / L3 的读写都接受 `namespace` 参数。
  L1 由 QACache 内部按 key 前缀隔离；L3 在 metadata 里存 namespace
  并在 search 时做后过滤。namespace=None 时只见全局条目。

其他改造
--------
* **L4 snippet 噪声清洗**：L4 产出 Passage 时先过 `rag/textclean.py`。
  实测 Tavily 的 snippet 中位数 1339 字，但含大量页面模板噪声
  （表格骨架 / CTA 按钮 / 评分块 / 图片文件名 / 页脚版权）。
  清洗零联网、零依赖、微秒级，实测去噪 12%（最脏站点 46~54%），
  直接改善证据预算利用率与语义去重质量。开关：`RAG_ENABLE_SNIPPET_CLEAN`。
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

from src.search.searcher import web_search
# L1 命中的读取侧准入复核（判据升级要能回溯到存量条目，见 hit_admissible）
from src.cache.cache_policy import hit_admissible

from . import config as rag_config
from . import calibration as rag_calibration   # 跨层分数校准
from .embedder import Embedder
from .textclean import clean_snippet      # snippet 噪声清洗（L4 用）
from .types import LayerName, Passage
from .vector_store import VectorStore
from .wiki_rag.kg_retriever import KGRetriever
from .wiki_rag.retriever import WikiRetriever


# ---------- L1: 精准 / 模糊 QA 缓存 ----------
class L1QACacheLayer:
    name: LayerName = "L1_qa"

    def __init__(self, qa_cache, enable_admission_recheck: bool = True):
        """qa_cache 需要是 qa_cache.QACache 实例（外部传入以复用配置）。

        Args:
            enable_admission_recheck:
                命中后是否再过一次准入判据（见 `lookup()`）。
                默认 True；置 False 退回旧行为，仅用于 A/B 对照。
        """
        self.qa_cache = qa_cache
        self.enable_admission_recheck = enable_admission_recheck

    def lookup(self, query: str, namespace: Optional[str] = None) -> Optional[str]:
        """直接返回预设答案；命中即可绕过 LLM。

        新增 namespace 参数做多租户隔离。QACache 内部会保证
        「精确命中」与「fuzzy 命中」都严格按 namespace 过滤，
        避免 A 用户的答案泄漏给 B 用户。

        ════════════════════════════════════════════════════════════════
        命中后必须复核准入 —— 「判据升级不回溯」
        ════════════════════════════════════════════════════════════════
        `decide_cacheability()` 只挂在**写入**侧，语义是"今后不准再写"，
        而不是"不准被返回"。差距就是存量脏条目（实测 72 条里有 7 条）。

        ⚠️ 为什么这里也要加，而不是只在 `agent._l1_get()` 加：
        L1 在本系统里有**多个读取入口** —— agent 的 Step 0 短路、
        retriever 的第一层短路、以及参与 RRF 融合的一路。实测只改 agent
        那处时，日志出现自相矛盾的一幕：

            [agent] L1 命中被读取侧复核拒绝 (tier=reject_partial_refusal)
            …
                  RAG 检索完成 [L1_qa:1], 融合 1 段, conf=1.000

        前门拦住了，后门又把同一条拒答放进来，还打上 conf=1.0 的最高
        可信标签 —— 比不拦更糟。「同一机制有多个入口」是本仓库反复踩到的
        疏漏模式，所以判据收敛到 `cache_policy.hit_admissible()` 统一实现。
        """
        ans = self.qa_cache.get(query, namespace=namespace)
        if ans is None or not self.enable_admission_recheck:
            return ans
        rejected = hit_admissible(query, ans)
        if rejected is not None:
            print(f"[L1_qa] 命中被读取侧复核拒绝 (tier={rejected.tier}): "
                  f"{rejected.reason} → 当作未命中")
            return None
        return ans

    def search(
        self, query: str, top_k: int = 1, namespace: Optional[str] = None,
    ) -> list[Passage]:
        ans = self.lookup(query, namespace=namespace)
        if ans is None:
            return []
        return [Passage(
            text=ans, title="QA Cache Hit", layer=self.name, score=1.0,
            # L1 命中意味着已过「精确匹配」或「0.90 阈值 + 槽位门禁」，
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
                    # BGE-M3 余弦 → 统一概率空间（中点 0.55）。
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

    （多租户隔离）
    ------------------
    L3 存的是**用户自己的历史问答**，天然是私有数据。若所有用户共写
    同一个 FAISS 索引、检索时不做任何过滤，A 用户的历史就会作为"外部资料"
    出现在 B 用户的 prompt 里（隐私泄漏 + 答案串味）。

    因此在 metadata 里写入 `namespace` 字段，并在 `search()` 里做
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

    def __init__(self, embedder: Embedder, index_dir: Optional[str] = None,
                 enable_admission_recheck: bool = True):
        """
        Args:
            enable_admission_recheck:
                召回后是否用同一套准入判据剔除拒答类历史条目
                （见 `search()`）。默认 True；置 False 退回旧行为，
                仅用于 A/B 对照与耗时基线测量。
        """
        self.embedder = embedder
        self.index_dir = index_dir or rag_config.L3_HISTORY_INDEX_DIR
        os.makedirs(self.index_dir, exist_ok=True)
        self._store = VectorStore(self.index_dir, rag_config.RAG_EMBED_DIM)
        self._lock = threading.Lock()
        self.enable_admission_recheck = enable_admission_recheck

    def search(
        self, query: str, top_k: int = 5, namespace: Optional[str] = None,
    ) -> list[Passage]:
        """检索历史问答。

        Args:
            query:     检索 query。
            top_k:     返回条数。
            namespace: 多租户隔离。
                       None → 只返回**无 namespace 的全局条目**
                              （无 namespace 的历史数据天然属于这一类）
                       str  → 只返回同 namespace 的条目
                       两种情况都**不会**跨租户串味。
        """
        if len(self._store) == 0:
            return []
        q_vec = self.embedder.embed(query)
        if q_vec is None:
            return []

        # 超取：为 namespace 后过滤 + 准入复核预留余量。
        # ⚠️ 准入复核也会丢弃候选，所以**无论有没有 namespace 都要超取**。
        # 原来只在 namespace 非空时超取；加了复核后如果不超取，一旦 top_k
        # 条里有拒答被剔掉，L3 返回的段数就会少于 top_k —— 相当于把
        # "剔除脏数据"变成了"减少证据量"，反而可能触发 low_evidence。
        fetch_k = top_k * self.OVERFETCH if (
            namespace is not None or self.enable_admission_recheck
        ) else top_k

        out: list[Passage] = []
        for meta, score in self._store.search(q_vec, top_k=fetch_k):
            # ---- namespace 后过滤 ----
            # 历史条目没有 namespace 字段 → get 返回 None → 只在
            # 查询侧也是 None（全局）时才匹配，保证向后兼容且不泄漏。
            if meta.get("namespace") != namespace:
                continue

            q = meta.get("query", "")
            a = meta.get("answer", "")

            # ---- 读取侧准入复核（与 L1 同一套判据）----
            # ⚠️ 这是「判据升级不回溯」在 L3 的重演，而且**比 L1 更隐蔽**：
            # L1 命中是直接返回给用户，错了一眼就能看出来；L3 是作为
            # **证据**送进 prompt，一条旧拒答会安静地把模型带偏。
            #
            # 实测用户四轮对话的第 4 轮（已修完 L1 之后）：融合的 6 段证据里
            # 有 3 段是 L3 里的旧拒答 ——
            #     [1] Q: 美国可饮酒的年龄是多少,法国和日本呢
            #         A: …法国：18 岁。但未直接点名法国，因此是推断…
            #     [3] Q: 请你触发网页搜索…  A: 我无法主动触发网页搜索…
            #     [5] Q: 法国最低饮酒年龄   A: …未能直接确认…
            # 而第 1/2 轮刚生成的**正确**答案（「法国为 18 岁」）根本没被召回。
            # 于是模型照抄了旧拒答的措辞，用户看到的仍是"基于欧洲通行标准的
            # 推断" —— L1 已经修好，但答案质量看起来毫无改善。
            #
            # 更糟的是这类条目还会**骗过 L4 兜底判据**：它存着 query 原文
            # （text 格式为「历史问答：Q: <原问题> A: <拒答>」），于是 query
            # 的每个实词都能在"证据"里找到，实词覆盖率虚高 → 判为"证据够用"
            # → 不触发联网。拒答因此自我强化：拒答 → 入库 → 召回到自己的
            # 拒答 → 覆盖率虚高 → 不联网 → 再次拒答。
            #
            # 写入侧（`_archive_if_enabled`）已经拦了拒答，但那同样只对
            # **今后**的写入有效。存量的 4 条（`scripts/clean_l3_refusals.py`
            # 可枚举）仍在被召回，所以读取侧必须也拦一道。
            if self.enable_admission_recheck and a:
                if hit_admissible(q or query, a) is not None:
                    continue
            text = f"历史问答：\nQ: {q}\nA: {a}"
            # 排除 answer（已拼进 text）与 sources。
            #
            # ⚠️ sources 必须排除，否则会产生**递归嵌套膨胀**：
            #   agent 归档时写的是 `[p.to_dict() for p in passages]`，而
            #   L3 召回的 passage 其 metadata 里又带着上一次归档的 sources。
            #   于是每归档一轮就套进去一层，深度随轮数线性增长、单条大小
            #   指数增长。实测未修复时 670 条 metadata.jsonl 涨到 160 MB
            #   （最大单条 22.9 MB / 嵌套 53 层），仅 json 解析就要 971 ms。
            #   写入侧 `add()` 也做了白名单裁剪，两侧都拦一次。
            md = {
                k: v for k, v in meta.items()
                if k not in ("answer", "sources")
            }
            # L3 与 L2 同为 BGE-M3 余弦，但语义**不等价**——
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

    # 归档 sources 时只保留这些字段。
    #
    # 为什么必须是白名单而不是"排除几个字段"：`Passage.to_dict()` 会带上
    # 整个 `metadata`，而 L3 召回的 passage 其 metadata 里可能又嵌着上一次
    # 归档的 sources。用黑名单容易漏掉新增字段，一旦漏掉就会重新引发
    # 递归膨胀（实测可让单条 metadata 涨到 22.9 MB）。白名单是唯一
    # 能保证"深度恒为 2 层"的写法。
    _SOURCE_FIELDS = ("title", "url", "layer", "score")

    @classmethod
    def _slim_sources(cls, sources: Optional[list[dict]]) -> list[dict]:
        """把归档来源裁剪成扁平的浅层结构（防递归嵌套膨胀）。

        只留可溯源所必需的字段：标题、链接、来源层、分数。
        正文 / metadata / 嵌套 sources 全部丢弃 —— 它们对"这条历史答案
        当初参考了哪些资料"这个用途没有价值，却是膨胀的唯一来源。
        """
        if not sources:
            return []
        out: list[dict] = []
        for s in sources:
            if not isinstance(s, dict):
                continue
            out.append({
                k: s[k] for k in cls._SOURCE_FIELDS if k in s
            })
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
            namespace: 写入时打上租户标记，检索时据此过滤。
                       None → 写成全局共享条目（无租户标记）。
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
                    # 裁剪后再落盘（见 `_slim_sources` 的说明）
                    "sources": self._slim_sources(sources),
                    "ts": time.time(),
                    # 租户标记。显式写 None 而不是省略字段，
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
            #    这正是 引入 calibration 的核心动因。
            score = 1.0 - i * 0.05

            # ══════════════════════════════════════════════════════════
            # snippet 噪声清洗（零成本，见 rag/textclean.py）
            # ══════════════════════════════════════════════════════════
            # 实测 Tavily 的 snippet 中位数 1339 字，但混着大量页面模板
            # 噪声：空表格骨架、`TWD 210 起立即預訂` 这类 CTA、
            # `4.7/51358 reviews` 评分块、图片文件名、页脚版权、
            # 月份选择器控件（`12345678910月1112`）等。实测去噪 12%，
            # 最脏的站点可达 46~54%。
            #
            # 三重代价：① 挤占 evidence.py 的 8000 字硬预算（噪声占一半
            # = 证据条数腰斩）；② 干扰模型注意力；③ 污染 rag/dedup.py
            # 的语义去重 —— 它只取前 512 字算余弦，若前半是导航栏，
            # 算出的是"页面模板像不像"而非"内容像不像"。
            #
            # 【为什么在这里洗、而不是在 searcher 里】
            # searcher 的结果会进 diskcache（TTL 数小时~数天）。在那里洗
            # 等于**缓存里存的是洗过的文本** —— 规则一改老缓存不会重洗，
            # 新旧策略混在一起，而且再也拿不到原文做对照。
            # 放在这里：缓存存原文，每次检索现洗（纯正则，微秒级），
            # 调规则立即全量生效。这是"尽量晚地做有损变换"的一般原则。
            #
            # clean_snippet 内部有兜底：清洗后不足原文 30% 就放弃清洗、
            # 原样返回，所以这一步**不可能**把证据洗空。
            text = clean_snippet(r.get("snippet", ""))

            out.append(Passage(
                text=text,
                title=r.get("title", ""),
                url=r.get("url", ""),
                score=score,
                layer=self.name,
                metadata={
                    "rank": i,
                    # 位次 → 概率。校准后 top1≈0.77、rank5≈0.39，
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
            # 修 `score=float(d.get("score") or 0.9)` 这个隐蔽 bug
            # ══════════════════════════════════════════════════════════════
            # 为什么不能用 `or` 兜默认分
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
            # 做法
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
                    "hop": hop,                    # 显式记录跳数
                    "calibrated": round(calibrated, 4),
                },
            ))
        return passages

    def warmup(self) -> None:
        """预热：打开 SQLite + 触发 Linker 热门向量 mmap。"""
        self._lazy_retriever().retrieve("预热", top_k=1, multi_hop=False)
