# qa_cache.py
"""Q&A 预设缓存模块（增强版）
====================

对常见 / 通用 / 自我介绍类问题提前注入答案。命中时绕过工具路由 / Query 改写 /
联网检索 / LLM 调用，毫秒级返回。

核心组件：
    - normalize_for_qa(query): 强归一化（全半角、大小写、标点、空白）
    - QACache:                  缓存类

设计要点：
    1. 多后端可插拔；未安装依赖(e.g. faiss)时自动降级到内存模式，不会启动失败
    2. 内存索引始终保留一份（即使用 redis 也回填，二次查询零网络开销）
    3. add / get / remove / clear / list / load_from_dict 一站式接口
    4. 外部 API 完全向后兼容（所有增强开关默认关闭 / 保守启用）

已落地的扩展点（内部增强，外部零感知）
    1. 模糊匹配 / 向量相似度命中
       - 用 FlagEmbedding BGE-M3 算 embedding；余弦相似度 >= fuzzy_threshold 视为命中
       - "美国总统是谁 / 谁是美国总统 / 当前美国总统" → 同一个答案
       - 开关：enable_fuzzy=True（与 L2/L5 共享同一 BGE-M3 向量空间）
    2. 命中统计 / Hit-rate 上报
       - _stats 记录 exact_hits / fuzzy_hits / misses / L1/L2/L3 命中分布
       - 通过 stats() 返回快照，reset_stats() 清零
    3. 多级缓存（L1 内存 → L2 → L3）
       - layers 参数支持 ["diskcache", "redis"] 之类的组合；未指定则退化为单后端
       - get 时按层从上往下查，命中后回填上层；write 时全层写入
    4. 异步写后端（写穿透 + 异步刷盘）
       - async_write=True 时启用 daemon worker 线程，主流程仅同步写内存 L1
       - 通过内部 queue.Queue 把 set/del/clear 事件派发给底层持久化后端

════════════════════════════════════════════════════════════════════════
P0 改造（本次新增，见 cache_policy.py）
════════════════════════════════════════════════════════════════════════
5. **per-entry TTL**（原来只有一个全局 TTL）
   `add(query, answer, ttl=...)` 支持按条目指定过期时间，配合
   `cache_policy.decide_cacheability()` 的分级 TTL：
     时效类 → 拒绝写入 / web 兜底 → 6h / 易变槽位 → 24h / 常识 → 30d
   为什么必要：全局 30 天 TTL 会把「今天天气」冻结一个月。

6. **槽位一致性门禁（防线 B）**
   fuzzy 命中不再"分数过线即返回"，而要再过一道
   `cache_policy.slots_compatible()`：抽取双方 query 的数字/年份/否定词/
   比较级/限定词/疑问焦点/命名实体/主题名词做比对，不一致即拒绝。
   为什么必要：BGE-M3 下「苹果CEO vs 苹果CFO」「总统 vs 副总统」
   余弦常在 0.85~0.88，单靠调阈值无法区分，会毫秒级返回错误答案。
   实测 12 个危险负样本里，门禁独立拦下 11 个（91.7%）。

   默认阈值从 0.8 改为 `cache_policy.FUZZY_THRESHOLD`（0.90，经正负样本
   实测标定，详见该常量处的长注释；一度设为 0.93，但实测证明那是纯粹的
   净损失——挡不掉任何额外负样本，只会误杀同义改述）。

   ⚠️ 各槽位的比对方式**并不统一**，这是设计而非疏漏：
     · 数字/年份/否定词/比较级/限定词 → 精确相等（一字之差换答案）
     · 疑问焦点 → **兼容判定**。陈述式提问（如「美国历届总统名单」）
       抽不到疑问词、焦点为 ∅，若要求相等会把所有「陈述式 ↔ 疑问式」
       的改述全部误拒。详见 `cache_policy._focus_compatible()`。
     · 主题名词 → Jaccard 相似度（容忍分词波动）

7. **namespace 多租户隔离**
   `add/get/remove(..., namespace="user:42")` 会把 key 前缀化为
   `user:42::<norm_q>`，实现 L1 的用户/会话级隔离，避免 A 用户的答案
   被 fuzzy 命中返回给 B 用户。namespace=None 时行为与改造前**完全一致**
   （key 就是裸 norm_q），历史落盘数据无需迁移。

8. **原始 query 元数据存储**
   槽位门禁需要拿到「缓存条目当初的原始 query」，而 `_mem` 的 key 是
   归一化后（去标点、小写）的串。因此新增 `_orig` 映射并持久化到
   `<cache_dir>/_meta`，冷启动可恢复；取不到时降级用 norm_key 本身
   （槽位抽取对去标点文本依然有效，只是实体识别精度略降）。

9. **embedding 维度基准改为运行时自描述**（修静默性能回归）
   原实现用**配置常量** `rag_config.RAG_EMBED_DIM`（写死 1024）判定
   磁盘 embedding 是否"脏"。一旦配置与当前编码器的真实维度不一致
   （换 bge-small/base、Matryoshka 降维、单测注入轻量编码器…），
   每次冷启动都会把**全部** embedding 判为脏数据并全量重算 ——
   只打一行 warn、不报错，属于静默性能回归（几千条就是几十秒启动开销）。

   现在改为：`_emb_store` 里存一条维度戳 `__emb_meta__`，基准由
   **编码器的真实输出**校准（`_learn_emb_dim`）；无戳的老 store 用
   首个读到的向量推断（`_infer_emb_dim`），历史数据零迁移即可复用；
   真的换了模型则**一次性**清空重建，而不是每次启动重算。
   配置里的维度降级为纯日志提示，不再参与任何丢弃决策。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import threading
import time
import unicodedata
from typing import Any, Iterable, Optional

from configs import config

# ---- P0-1：L1 准入策略 / 槽位门禁（详见 cache_policy.py 模块 docstring）----
# 放在这里 import 而非函数内，是因为 cache_policy 只依赖 re + 可选 jieba，
# 极轻量、无循环依赖风险（它不 import qa_cache）。
from cache_policy import FUZZY_THRESHOLD as _POLICY_FUZZY_THRESHOLD
from cache_policy import slots_compatible as _slots_compatible

# 可选后端（按需启用）
try:
    import diskcache as _diskcache  # type: ignore
except Exception:
    _diskcache = None  # type: ignore

try:
    import redis as _redis  # type: ignore
except Exception:
    _redis = None  # type: ignore

# ============================================================
#   向量后端（D1：全库统一到 FlagEmbedding BGE-M3）
# ============================================================
# 直接复用 rag/ 下 vendored 的 BGE-M3 编码器（与 L2/L5 同一向量空间），
# 这样 L1 的模糊命中和 L2/L5 检索共享同一语义空间，跨层可比。
from rag.wiki_rag import embedder as _bge

# numpy 可选：有则矩阵化 cosine，无则回退纯 Python
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None  # type: ignore

# ANN 后端（可选）：优先 faiss，其次 hnswlib
try:
    import faiss as _faiss  # type: ignore
except Exception:
    _faiss = None  # type: ignore

try:
    import hnswlib as _hnswlib  # type: ignore
except Exception:
    _hnswlib = None  # type: ignore


# ============================================================
#                         归一化工具
# ============================================================

_QA_NORM_PUNCT_RE = re.compile(
    r"[\s，。！？；：、,.?!;:\"'“”‘’（）()【】\[\]<>《》「」…—\-_/\\|`~@#$%^&*+=]+"
)


def normalize_for_qa(query: str) -> str:
    """强归一化：NFKC + 小写 + 去除全部标点/空白。"""
    if not query:
        return ""
    q = unicodedata.normalize("NFKC", query).lower()
    q = _QA_NORM_PUNCT_RE.sub("", q)
    return q


# ============================================================
#                         缓存主类
# ============================================================

BACKEND_MEMORY = "memory"
BACKEND_DISKCACHE = "diskcache"
BACKEND_REDIS = "redis"

_REDIS_KEY_PREFIX = "qa_cache:"

# namespace 与归一化 key 之间的分隔符。
# 选 "::" 是因为 normalize_for_qa() 会把所有标点（含 ":"）去掉，
# 所以正常的用户 query 归一化后**不可能**包含 "::"，不会与 namespace 冲突。
_NS_SEP = "::"

# 异步写事件类型
_OP_SET = "set"
_OP_DEL = "del"
_OP_CLEAR = "clear"


class QACache:
    """Q&A 预设缓存（增强版）。

    向后兼容用法：
        cache = QACache(backend="diskcache")
        cache.add("中国的首都是哪里？", "北京")
        cache.get("中国 的首都是哪里")    # → "北京"

    多级缓存：
        cache = QACache(layers=["diskcache", "redis"])   # L1 内存 → L2 disk → L3 redis

    模糊向量命中：
        cache = QACache(backend="diskcache", enable_fuzzy=True,
                        fuzzy_threshold=0.90, embed_model="bge-m3")

    异步写后端：
        cache = QACache(backend="diskcache", async_write=True)

    P0 新增用法：
        # 1) per-entry TTL（配合 cache_policy.decide_cacheability）
        cache.add("量子计算是什么", ans, ttl=30*24*3600)

        # 2) 多租户隔离（namespace=None 时与改造前完全一致）
        cache.add("我叫什么", ans, namespace="user:42")
        cache.get("我叫什么", namespace="user:42")     # 命中
        cache.get("我叫什么", namespace="user:99")     # 不命中（跨用户隔离）

        # 3) 槽位门禁（默认开启，可关闭做 A/B 对比）
        cache = QACache(enable_slot_gate=False)        # 退回旧行为
    """

    def __init__(
        self,
        backend: Optional[str] = None,      # cache读取的方式: discache, redis, memory
        cache_dir: Optional[str] = None,
        redis_url: Optional[str] = None,
        ttl: Optional[int] = None,
        verbose: bool = False,
        # ---- 新增：扩展点开关（全部可选，默认保守） ----
        layers: Optional[list[str]] = None,
        enable_fuzzy: bool = True,
        fuzzy_threshold: Optional[float] = None,
        embed_model: str = "bge-m3",
        async_write: bool = False,
        # ---- P0-1：槽位一致性门禁 ----
        enable_slot_gate: bool = True,   # fuzzy 命中后是否再过槽位门禁（强烈建议 True）
        strict_entities: bool = True,    # 门禁中实体集合是否要求完全相等
        # ---- ANN（HNSW/FAISS）参数 ----
        ann_backend: str = "auto",     # "auto" | "faiss" | "hnswlib" | "none"
        ann_min_size: int = 500,       # N 少于此值继续用 numpy 矩阵乘（建索引反而更贵）
        ann_hnsw_m: int = 16,          # HNSW 图内邻居数：越大召回越高、内存越大
        ann_ef_construction: int = 200,
        ann_ef_search: int = 64,       # 查询时探索因子：越大越准、越慢
    ):
        # 未显式传入的参数统一回退到 config.py 默认值
        backend = backend or config.QA_CACHE_BACKEND
        cache_dir = cache_dir or config.QA_CACHE_DIR
        redis_url = redis_url or config.QA_REDIS_URL
        if ttl is None:
            ttl = config.QA_CACHE_TTL

        self.ttl = ttl
        self.verbose = verbose
        self.cache_dir = cache_dir
        self._mem: dict[str, str] = {}      # L1 内存（最快）；key = [ns::]norm_q
        self._disk = None
        self._redis = None

        # ---- 多级缓存 layers 归一化 ----
        # 兼容：未传 layers 时按老逻辑用单一 backend；传了 layers 时以 layers 为准
        if layers:
            self.layers = [b.lower() for b in layers if b]
            self.backend = self.layers[0]  # 兼容 __repr__ / 老代码
        else:
            self.layers = [backend]     # default: ['diskcache']
            self.backend = backend            # default: diskcache

        # ---- 初始化各后端连接 ----
        for b in list(self.layers):
            if b == BACKEND_DISKCACHE:
                if _diskcache is None:
                    self._warn("diskcache 未安装，跳过该层")
                    self.layers.remove(b)
                elif self._disk is None:
                    self._disk = _diskcache.Cache(cache_dir)
            elif b == BACKEND_REDIS:
                if _redis is None:
                    self._warn("redis 未安装，跳过该层")
                    self.layers.remove(b)
                elif self._redis is None:
                    self._redis = _redis.Redis.from_url(redis_url, decode_responses=True)
            elif b == BACKEND_MEMORY:
                pass  # 内存层始终在
            else:
                self._warn(f"未知后端 '{b}'，忽略")
                self.layers.remove(b)

        if not self.layers:
            self.layers = [BACKEND_MEMORY]
            self.backend = BACKEND_MEMORY

        # ---- 命中统计 ----
        self._stats_lock = threading.Lock()
        self._stats: dict[str, Any] = {
            "exact_hits": 0,
            "fuzzy_hits": 0,
            "misses": 0,
            "layer_hits": {"L1_mem": 0, "L2": 0, "L3": 0, "fuzzy": 0},
            "writes": 0,
            "async_dropped": 0,
            # P0-1 新增：被槽位门禁拦下的 fuzzy 候选数。
            # 这个指标很重要——它直接量化了"本次改造避免了多少次错误答案"，
            # 上线后应该重点观察。若长期为 0 说明门禁没起作用（或阈值太严）；
            # 若相对 fuzzy_hits 过高（>50%）说明阈值可以适当放宽。
            "slot_gate_rejects": 0,
        }

        # ---- 向量模糊匹配 ----
        # D1：向量编码统一走 FlagEmbedding BGE-M3（与 L2/L5 同一向量空间）。
        self.enable_fuzzy = bool(enable_fuzzy)
        # P0-1：阈值默认改为 cache_policy.FUZZY_THRESHOLD（0.90，原来 0.8）。
        # 显式传参仍然优先，便于单测 / A-B 对比。
        self.fuzzy_threshold = float(
            fuzzy_threshold if fuzzy_threshold is not None
            else _POLICY_FUZZY_THRESHOLD
        )
        # P0-1：槽位门禁开关
        self.enable_slot_gate = bool(enable_slot_gate)
        self.strict_entities = bool(strict_entities)
        self.embed_model = embed_model  # 只有在add, reloaded_all时才会触发写入

        # ══════════════════════════════════════════════════════════════════
        # 向量维度基准 `_emb_dim`：用于识别「换了模型后的历史脏缓存」
        # ══════════════════════════════════════════════════════════════════
        # 【改造前的实现及其缺陷】
        #     self._expected_dim = int(rag_config.RAG_EMBED_DIM)   # 写死 1024
        #     ...
        #     if len(cached) != self._expected_dim: 丢弃并重算
        #
        #   问题在于**基准取自配置常量，而不是当前真正在用的编码器**。
        #   一旦两者不一致，每次冷启动都会把**全部** embedding 判为脏数据、
        #   全量重算 —— 而且只打一行 warn，没有任何报错，属于静默性能回归。
        #
        #   什么时候会不一致（都是很正常的操作）：
        #     · 换成 bge-small(512) / bge-base(768) 却忘了同步改 RAG_EMBED_DIM
        #     · 对 bge-m3 做 Matryoshka 降维（模型名不变、维度变了）
        #     · 单测/离线脚本注入自己的轻量编码器
        #   这也正是 `test_embedding_cached_on_disk_and_reused` 一直失败的原因：
        #   测试用 3 维假编码器，而基准是配置里的 1024。
        #
        # 【改造后】基准改为「**运行时自描述**」，配置不再参与判定：
        #     1. `_emb_store` 里存一条维度戳 `__emb_meta__`；
        #     2. 冷启动时读它作为基准（老数据没有戳 → 从首个读到的向量推断）；
        #     3. 真正调编码器算出向量时调 `_learn_emb_dim()`：
        #        - 基准未知   → 记录并写戳
        #        - 与基准不符 → 说明**模型确实换了** → 清空整个 embedding store
        #                       并换上新基准（一次性代价，而非每次启动重算）
        #
        #   这样做的三个好处：
        #     · 基准永远等于「当前编码器的真实维度」，不会被错配的配置误伤；
        #     · 换模型仍能被检测到，且是**一次性清理**而不是反复重算；
        #     · 混维向量喂给 np.asarray 导致 `inhomogeneous shape` 的崩溃
        #       依然被防住（这是原实现真正想解决的问题，现在解决得更准）。
        self._emb_dim: Optional[int] = None
        # 配置里的维度只作为**日志提示**（用于提醒配置与实际不一致），
        # 绝不再参与"是否丢弃缓存"的判定。
        try:
            from rag import config as _rag_config
            self._configured_dim: Optional[int] = int(_rag_config.RAG_EMBED_DIM)
        except Exception:
            self._configured_dim = None
        self._embeddings: dict[str, list[float]] = {}   # norm_key -> 已归一化 vec
        self._embed_lock = threading.Lock()

        # 矩阵化缓存（矩阵 + 行号索引）
        self._np_matrix = None                    # np.ndarray (N, D) 或 None
        self._np_keys: list[str] = []                   # 行号 → norm_key
        self._np_dirty = True

        # ---- Embedding 持久化存储（避免冷启动重算） ----
        self._emb_store = None
        if self.enable_fuzzy and _diskcache is not None:
            try:
                emb_dir = os.path.join(cache_dir, "_embeddings")
                self._emb_store = _diskcache.Cache(emb_dir)
            except Exception as e:
                self._warn(f"embedding 持久化目录初始化失败: {e}")

        # 读取维度戳作为本次的维度基准（详见上方 `_emb_dim` 的长注释）。
        # 必须放在 `reload_all()` **之前** —— 否则 reload 时基准还是 None，
        # 会退化成"用首个读到的向量推断"，虽然也能工作但不如显式戳可靠。
        self._load_emb_dim()

        # ---- P0-1：原始 query 元数据存储 ----
        # 为什么需要：槽位门禁要比对「候选缓存条目当初的原始 query」与
        # 「当前用户 query」。但 self._mem 的 key 是 normalize_for_qa() 之后的串
        # （去掉了全部标点、转小写），信息有损——比如 "GPT-4" 会变成 "gpt4"，
        # jieba 对无标点长串的实体切分也会变差。所以额外存一份原文。
        #
        # 存储位置：<cache_dir>/_meta（独立 diskcache），进程重启后可恢复。
        # 降级策略：diskcache 不可用 / 取不到时，退化为用 norm_key 本身做槽位
        # 抽取——数字、否定词、比较级、疑问焦点这几类槽位对去标点文本依然有效，
        # 只有实体识别精度略降，门禁仍然生效（保守方向）。
        self._orig: dict[str, str] = {}      # [ns::]norm_key -> 原始 query
        self._meta_store = None
        if _diskcache is not None:
            try:
                meta_dir = os.path.join(cache_dir, "_meta")
                self._meta_store = _diskcache.Cache(meta_dir)
            except Exception as e:
                self._warn(f"meta 持久化目录初始化失败: {e}")

        # ---- ANN 索引状态 ----
        # backend 选择：auto → 优先 faiss，其次 hnswlib，否则 none（走 numpy 矩阵乘）
        req = (ann_backend or "auto").lower()
        if req == "auto":
            if _faiss is not None:
                self.ann_backend = "faiss"
            elif _hnswlib is not None:
                self.ann_backend = "hnswlib"
            else:
                self.ann_backend = "none"
        elif req == "faiss" and _faiss is None:
            self._warn("faiss 未安装，ANN 回退为 numpy 矩阵乘")
            self.ann_backend = "none"
        elif req == "hnswlib" and _hnswlib is None:
            self._warn("hnswlib 未安装，ANN 回退为 numpy 矩阵乘")
            self.ann_backend = "none"
        else:
            self.ann_backend = req if req in ("faiss", "hnswlib", "none") else "none"

        self.ann_min_size = int(ann_min_size)
        self.ann_hnsw_m = int(ann_hnsw_m)
        self.ann_ef_construction = int(ann_ef_construction)
        self.ann_ef_search = int(ann_ef_search)
        self._ann_index = None                          # faiss.Index / hnswlib.Index
        self._ann_keys: list[str] = []                  # row_id -> norm_key
        self._ann_alive: list[bool] = []                # 软删除位图
        self._ann_dirty = True
        self._ann_removed = 0                           # 已软删条数；超阈值触发全量 rebuild

        # ---- 异步写 ----
        self.async_write = bool(async_write)
        self._async_q: Optional["queue.Queue[tuple[str, tuple[Any, ...]]]"] = None
        self._worker: Optional[threading.Thread] = None
        if self.async_write:
            self._async_q = queue.Queue(maxsize=10000)  # type: ignore[assignment]
            self._worker = threading.Thread(
                target=self._async_worker_loop, name="qa_cache_writer", daemon=True
            )
            self._worker.start()
        
        # 需要先读取diskcache/redis, 保证self._mem读取到各级存储的值
        self._num_mem = self.reload_all()
        print(f"[qa_cache] enable_fuzzy={self.enable_fuzzy}, "
            f"fuzzy_threshold={self.fuzzy_threshold}, "
            f"slot_gate={self.enable_slot_gate}, "
            f"mem_size={self._num_mem}, emb_size={len(self._embeddings)}")

    # ---------- helpers ---------- #
    def _warn(self, msg: str) -> None:
        print(f"[qa_cache] {msg}")

    @staticmethod
    def _redis_key(norm_q: str) -> str:
        return f"{_REDIS_KEY_PREFIX}{norm_q}"

    # ---------- P0-3：namespace（多租户隔离）---------- #
    @staticmethod
    def _make_key(query: str, namespace: Optional[str] = None) -> str:
        """把 (query, namespace) 归一化成统一的缓存 key。

        - namespace=None  → 返回裸 `norm_q`，**与改造前完全一致**，
          历史落盘的 diskcache/redis 数据无需迁移即可继续命中。
        - namespace="u:42" → 返回 `u:42::norm_q`，实现租户隔离。

        为什么用 "::" 做分隔符：`normalize_for_qa()` 会剔除所有标点（含 ":"），
        因此任何真实 query 归一化后都不会含 "::"，不存在 key 冲突/伪造风险。
        """
        norm = normalize_for_qa(query)
        if not norm:
            return ""
        if not namespace:
            return norm
        return f"{namespace}{_NS_SEP}{norm}"

    @staticmethod
    def _key_namespace(key: str) -> Optional[str]:
        """从完整 key 里解析出 namespace；无前缀则返回 None。"""
        idx = key.find(_NS_SEP)
        return key[:idx] if idx > 0 else None

    def _same_namespace(self, key: str, namespace: Optional[str]) -> bool:
        """判断某个已存 key 是否属于给定 namespace。

        用于 fuzzy 检索时过滤候选：**绝不允许跨 namespace 命中**，
        否则 A 用户的私有答案会通过向量相似度泄漏给 B 用户。
        """
        return self._key_namespace(key) == (namespace or None)

    # ---------- P0-1：原始 query 元数据 ---------- #
    def _meta_key(self, key: str) -> str:
        """meta store 的 key（加前缀避免与其它用途混淆）。"""
        return f"orig::{key}"

    def _remember_orig(self, key: str, query: str) -> None:
        """记录 key 对应的原始 query 文本（供槽位门禁比对）。"""
        self._orig[key] = query
        if self._meta_store is not None:
            try:
                # 不设 expire：meta 是"辅助信息"，即使主条目过期后残留
                # 也只占极少空间，且下次同 key 写入会覆盖。
                self._meta_store.set(self._meta_key(key), query, expire=None)
            except Exception as e:
                self._warn(f"meta 写入失败: {e}")

    def _get_orig(self, key: str) -> str:
        """取回 key 的原始 query；取不到时降级返回 norm_key 本身。

        降级是安全的：槽位抽取对"去标点的归一化串"依然能识别数字/年份/
        否定词/比较级/限定词/疑问焦点，只是中文实体切分精度略降。
        """
        cached = self._orig.get(key)
        if cached:
            return cached
        if self._meta_store is not None:
            try:
                v = self._meta_store.get(self._meta_key(key))
                if v:
                    self._orig[key] = v      # 回填内存，避免反复读盘
                    return v
            except Exception:
                pass
        # 降级：用 norm_key（可能带 namespace 前缀，这里剥掉）
        idx = key.find(_NS_SEP)
        return key[idx + len(_NS_SEP):] if idx > 0 else key

    def _incr(self, path: str, sub: Optional[str] = None) -> None:
        """ 用于记录cache命中次数，path：主key，sub：次key"""
        with self._stats_lock:
            if sub is None:
                self._stats[path] = self._stats.get(path, 0) + 1
            else:
                self._stats[path][sub] = self._stats[path].get(sub, 0) + 1

    # ============================================================
    #                      向量相似度模糊匹配
    # ============================================================
    @staticmethod
    def _hash_cache_key(model: str, norm_key: str) -> str:
        """embedding 持久化存储的 key， 使用hash：含模型名，避免换模型脏读。"""
        h = hashlib.sha1(f"{model}::{norm_key}".encode("utf-8")).hexdigest()
        return f"emb::{h}"

    # ---------- 向量维度基准管理（见 __init__ 里 `_emb_dim` 的长注释）---------- #
    # 维度戳在 _emb_store 里的 key。加 `__` 前后缀避免与 `emb::<sha1>` 冲突。
    _EMB_DIM_KEY = "__emb_meta__"

    def _load_emb_dim(self) -> None:
        """冷启动时从磁盘读取维度戳，作为本次运行的维度基准。

        戳的结构：`{"model": "<embed_model>", "dim": 1024}`
        带上 model 名是因为 `_hash_cache_key()` 已经把模型名混进了条目 key，
        同一个 store 里理论上可以并存多个模型的向量；戳只对**当前** model 生效。

        兼容性：改造前写入的 store 里没有这条戳。此时 `_emb_dim` 保持 None，
        `_embed()` 会用"首个成功读到的向量长度"来推断基准（见 `_infer_emb_dim`），
        因此**历史缓存不会被误判为脏数据**，无需任何迁移。
        """
        if self._emb_store is None:
            return
        try:
            meta = self._emb_store.get(self._EMB_DIM_KEY)
            if isinstance(meta, dict) and meta.get("model") == self.embed_model:
                dim = int(meta.get("dim") or 0)
                if dim > 0:
                    self._emb_dim = dim
                    # 配置与实际不符时给一次明确提示。这只是提示，
                    # 不影响缓存复用 —— 修掉"配置说了算"正是本次改动的核心。
                    if (self._configured_dim is not None
                            and self._configured_dim != dim):
                        self._warn(
                            f"提示：磁盘 embedding 维度为 {dim}，而配置 "
                            f"RAG_EMBED_DIM={self._configured_dim}。"
                            f"以实际维度为准；如非预期请检查 RAG_EMBED_DIM。"
                        )
        except Exception as e:
            self._warn(f"embedding 维度戳读取失败（按未知维度处理）: {e}")

    def _save_emb_dim(self, dim: int) -> None:
        """把维度戳写回磁盘，供下次冷启动直接读取。"""
        if self._emb_store is None:
            return
        try:
            self._emb_store.set(
                self._EMB_DIM_KEY,
                {"model": self.embed_model, "dim": int(dim)},
                expire=None,
            )
        except Exception as e:
            self._warn(f"embedding 维度戳写入失败: {e}")

    def _infer_emb_dim(self, dim: int) -> None:
        """基准未知时，用**磁盘上读到的**首个向量长度推断基准。

        这是对"改造前写入、没有维度戳"的老 store 的兼容路径：
        既然那批向量本来就是同一个模型算出来的，取第一个的长度即可，
        不需要重算任何东西。
        """
        if self._emb_dim is None and dim > 0:
            self._emb_dim = dim
            self._save_emb_dim(dim)

    def _learn_emb_dim(self, dim: int) -> None:
        """用**编码器刚算出的**向量长度校准基准；不符则清空整个 store。

        这是"换模型"的唯一可信信号：编码器实际输出的维度才是真相。

        三种情况：
          1. 基准未知      → 记录 + 写戳（首次运行 / 老 store 首次编码）
          2. 与基准一致    → 什么都不做（绝大多数情况）
          3. 与基准不一致  → **确实换了模型/维度**：旧向量已不可比，
             清空 store 并换上新基准。

        为什么第 3 种要清空而不是逐条丢弃：
          旧维度的向量与新向量**处在不同的向量空间**，混在一起做余弦
          在数学上没有意义（长度不同还会让 np.asarray 抛
          `inhomogeneous shape`）。而逐条丢弃的老做法会导致每次冷启动
          都重算一遍相同的条目 —— 把"一次性迁移成本"变成了"永久性能税"。
          清空是一次性的：之后所有条目按新维度正常落盘、正常复用。
        """
        if dim <= 0:
            return
        if self._emb_dim is None:
            self._emb_dim = dim
            self._save_emb_dim(dim)
            return
        if dim == self._emb_dim:
            return

        # 维度变了 → 旧缓存整体失效
        self._warn(
            f"检测到 embedding 维度变化（{self._emb_dim} → {dim}），"
            f"判定为编码模型/维度已更换：清空 embedding 磁盘缓存并重建。"
            f"（这是一次性操作，后续冷启动可正常复用）"
        )
        if self._emb_store is not None:
            try:
                self._emb_store.clear()
            except Exception as e:
                self._warn(f"embedding 磁盘缓存清空失败: {e}")
        with self._embed_lock:
            self._embeddings.clear()
            self._np_dirty = True
            self._ann_dirty = True
        self._emb_dim = dim
        self._save_emb_dim(dim)

    def _embed(self, text: str, cache_key: Optional[str] = None) -> Optional[list[float]]:
        """获取 text 的 embedding（已 L2 归一化）。

        优先级：
            1. 磁盘持久化存储 self._emb_store（避免重算）
            2. 调 BGE-M3 重算 → 写盘

        Args:
            text:      待编码文本。
            cache_key: 传入（通常是 norm_key）则启用磁盘缓存；
                       临时 query（fuzzy 查询侧）不传，仅实时算不落盘。

        维度处理（本次修复的重点）：
          * 读盘命中 → 若基准未知就用它推断基准（兼容无戳的老 store）；
            若与基准不符则**只跳过这一条**（同一 store 内的个别脏条目），
            不会因此把全部缓存作废。
          * 编码器实算 → 调 `_learn_emb_dim()`。只有编码器的真实输出
            才有资格改写基准，也只有它能证明"模型真的换了"。
        """
        if not self.enable_fuzzy:
            return None

        # ---- 1) 磁盘命中 ----
        if cache_key is not None and self._emb_store is not None:
            try:
                cached = self._emb_store.get(
                    self._hash_cache_key(self.embed_model, cache_key)
                )
                if cached is not None:
                    cached = list(cached)
                    n = len(cached)
                    if n > 0:
                        if self._emb_dim is None:
                            # 老 store 没有维度戳 → 用它自己推断基准，
                            # 从而**不再误判整批历史缓存为脏数据**（本次修复的核心）
                            self._infer_emb_dim(n)
                            return cached
                        if n == self._emb_dim:
                            return cached
                        # 与基准不符：只跳过这一条（同 store 内的个别脏条目）
                        self._warn(
                            f"跳过维度异常的磁盘 embedding（{n}≠{self._emb_dim}），"
                            f"该条将重算"
                        )
            except Exception as e:
                self._warn(f"embedding 磁盘读取失败: {e}")

        # ---- 2) 磁盘未命中 → 走编码器实算（与 L2/L5 同一向量空间）----
        arr = _bge.encode([text], normalize=True)
        vec: Optional[list[float]] = arr[0].tolist()
        if not vec:
            return None

        # 编码器的真实输出维度才是唯一可信基准；不符则一次性清空重建
        self._learn_emb_dim(len(vec))

        # 预归一化：存盘/内存都存 unit vector，后续 cos = dot
        vec = self._l2_normalize(vec)

        # ---- 3) 回写磁盘 ----
        if cache_key is not None and self._emb_store is not None:
            try:
                self._emb_store.set(
                    self._hash_cache_key(self.embed_model, cache_key), vec, expire=None
                )
            except Exception as e:
                self._warn(f"embedding 磁盘写入失败: {e}")
        return vec

    @staticmethod
    def _l2_normalize(vec: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in vec))
        if n == 0:
            return vec
        return [x / n for x in vec]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """纯 Python 余弦（回退路径，仅在 numpy 不可用时使用）。"""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _rebuild_matrix_locked(self) -> None:
        """将 self._embeddings 拼成 (N, D) 矩阵。需在 _embed_lock 内调用。

        维度防御（最后一道保险）：若内存索引里混入了长度不一的向量，
        直接喂给 ``np.asarray`` 会抛 ``ValueError: inhomogeneous shape``，
        导致整个 fuzzy 功能不可用。这里以"出现次数最多的维度"为准，
        剔除少数派坏条目。

        注：正常路径下这里**不应该**有活儿可干 —— `_embed()` 已经按
        `_emb_dim` 基准过滤了读盘条目，`_learn_emb_dim()` 也会在检测到
        模型更换时清空重建（见模块 docstring 第 9 条）。保留本段是为了
        兜住"手工往 store 里塞数据"之类的异常情况，属于防御性编程。
        """
        if _np is None or not self._embeddings:
            self._np_matrix = None
            self._np_keys = []
            self._np_dirty = False
            return

        # 以众数维度为基准，剔除维度不一致的脏向量
        from collections import Counter
        dim_counter = Counter(len(v) for v in self._embeddings.values() if v)
        target_dim = dim_counter.most_common(1)[0][0]
        bad = [k for k, v in self._embeddings.items() if not v or len(v) != target_dim]
        if bad:
            self._warn(
                f"检测到 {len(bad)} 条维度异常的 embedding（期望 {target_dim} 维），"
                f"已从内存索引剔除（不影响精确命中）。建议清理磁盘缓存 _embeddings 目录。"
            )
            for k in bad:
                self._embeddings.pop(k, None)

        keys = list(self._embeddings.keys())
        if not keys:
            self._np_matrix = None
            self._np_keys = []
            self._np_dirty = False
            return
        mat = _np.asarray(
            [self._embeddings[k] for k in keys], dtype=_np.float32
        )
        # 保险起见，再归一化一次（写入时已归一，这里幂等）
        norms = _np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        self._np_matrix = mat
        self._np_keys = keys
        self._np_dirty = False

    # ---------- ANN 索引：HNSW / FAISS ---------- #
    def _ann_available(self) -> bool:
        if self.ann_backend == "faiss" and _faiss is not None:
            return True
        if self.ann_backend == "hnswlib" and _hnswlib is not None:
            return True
        return False

    def _rebuild_ann_locked(self) -> None:
        """全量重建 ANN 索引。需在 _embed_lock 内调用。

        向量已 L2 归一化 → 用内积度量即余弦相似度，O(log N) 查询。
        """
        if not self._ann_available() or _np is None or not self._embeddings:
            self._ann_index = None
            self._ann_keys = []
            self._ann_alive = []
            self._ann_dirty = False
            self._ann_removed = 0
            return

        keys = list(self._embeddings.keys())
        mat = _np.asarray(
            [self._embeddings[k] for k in keys], dtype=_np.float32
        )
        # 保险再归一
        norms = _np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        d = int(mat.shape[1])
        n = int(mat.shape[0])

        if self.ann_backend == "faiss":
            # HNSW + 内积（余弦）
            index = _faiss.IndexHNSWFlat(d, self.ann_hnsw_m, _faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.ann_ef_construction
            index.hnsw.efSearch = self.ann_ef_search
            index.add(mat)
            self._ann_index = index
        elif self.ann_backend == "hnswlib":
            index = _hnswlib.Index(space="ip", dim=d)  # inner product
            index.init_index(
                max_elements=max(n, 1024),
                ef_construction=self.ann_ef_construction,
                M=self.ann_hnsw_m,
            )
            index.add_items(mat, list(range(n)))
            index.set_ef(self.ann_ef_search)
            self._ann_index = index

        self._ann_keys = keys
        self._ann_alive = [True] * n
        self._ann_dirty = False
        self._ann_removed = 0

    def _ann_search(
        self, q_vec: list[float], top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """在 ANN 索引中查最相似的活条目，返回 [(norm_key, score), ...]。

        P0-1 改造：从「只返回 top-1」改为「返回 top-k 候选列表」。
        原因：加了 namespace 过滤 + 槽位门禁后，top-1 很可能被拒；
        必须能继续看 top-2/3…，否则会把本可正确命中的条目也一起丢掉
        （召回率无谓下降）。
        """
        if not self._ann_available() or _np is None:
            return []
        with self._embed_lock:
            if self._ann_dirty or self._ann_index is None:
                self._rebuild_ann_locked()
            index = self._ann_index
            keys = list(self._ann_keys)
            alive = list(self._ann_alive)
        if index is None or not keys:
            return []

        q = _np.asarray([q_vec], dtype=_np.float32)
        k = min(max(top_k, 1), len(keys))
        if self.ann_backend == "faiss":
            sims, idxs = index.search(q, k)
            sims, idxs = sims[0], idxs[0]
        else:  # hnswlib
            labels, dists = index.knn_query(q, k=k)
            idxs, sims = labels[0], dists[0]  # ip space 返回的就是内积

        out: list[tuple[str, float]] = []
        for row, s in zip(idxs, sims):
            row = int(row)
            if row < 0 or row >= len(keys):
                continue
            if alive[row]:
                out.append((keys[row], float(s)))
        return out

    # ---------- P0-1：fuzzy 命中的三道校验 ---------- #
    def _accept_fuzzy_candidate(
        self,
        cand_key: str,
        score: float,
        query: str,
        namespace: Optional[str],
    ) -> Optional[str]:
        """校验一个 fuzzy 候选是否可以真正命中，通过则返回答案。

        三道校验（顺序即成本从低到高）：

          ① **分数门槛**：`score >= self.fuzzy_threshold`（默认 0.90）
             ——最廉价，先挡掉绝大多数。

          ② **namespace 隔离**（P0-3）：候选必须与当前 namespace 完全一致。
             这是安全要求：绝不允许 A 用户的答案通过向量相似度泄漏给 B。

          ③ **槽位一致性门禁**（P0-1 核心）：
             `cache_policy.slots_compatible(候选原始query, 当前query)`
             抽取数字/年份/否定词/比较级/限定词/疑问焦点/命名实体做精确
             集合比对，任一不同即拒绝。

             为什么必须有 ③：即使把阈值抬得很高，下面这些 pair 在 BGE-M3 上
             依然可能过线，而它们的答案完全不同：
                「美国总统是谁」  vs 「美国副总统是谁」
                「苹果的CEO是谁」 vs 「苹果的CFO是谁」
                「2024年GDP」    vs 「2025年GDP」
                「推荐保健品」   vs 「不要推荐保健品」
             向量相似度是连续的、软的，无法可靠区分"一字之差"；
             槽位比对是离散的、硬的，恰好互补。

        Returns:
            命中时返回答案字符串；被任一道拒绝时返回 None。
        """
        # ① 分数门槛
        if score < self.fuzzy_threshold:
            return None

        # ② namespace 隔离
        if not self._same_namespace(cand_key, namespace):
            if self.verbose:
                self._warn(
                    f"fuzzy 候选跨 namespace 被拒: key={cand_key!r} "
                    f"ns={namespace!r}"
                )
            return None

        # ③ 槽位一致性门禁
        if self.enable_slot_gate:
            cand_query = self._get_orig(cand_key)
            ok, reason = _slots_compatible(
                cand_query, query, strict_entities=self.strict_entities
            )
            if not ok:
                self._incr("slot_gate_rejects")
                if self.verbose:
                    self._warn(
                        f"fuzzy 候选被槽位门禁拒绝 (score={score:.3f}): "
                        f"{cand_query!r} vs {query!r} —— {reason}"
                    )
                return None

        ans = self._mem.get(cand_key)
        if ans is None:
            return None
        if self.verbose:
            self._warn(f"fuzzy hit '{cand_key}' score={score:.3f} ✓ 通过全部校验")
        return ans

    def _fuzzy_lookup(
        self, query: str, namespace: Optional[str] = None,
    ) -> Optional[str]:
        """向量模糊匹配（带 namespace 过滤 + 槽位门禁）。

        P0-1 改造前后的行为差异：
            改造前：算相似度 → top-1 分数 >= 0.8 → **直接返回答案**
            改造后：算相似度 → 取 top-K 候选 → 逐个过
                    `_accept_fuzzy_candidate()` 三道校验 → 第一个通过的才返回

        取 top-K（而非 top-1）的原因见 `_ann_search` 注释：
        加了过滤后 top-1 常被拒，需要继续看后续候选以保住召回率。
        """
        if not self.enable_fuzzy or not self._embeddings:
            return None
        # 查询向量不存盘（实时 query）；已归一化
        q_vec = self._embed(query, cache_key=None)
        if q_vec is None:
            return None

        # 候选数：多取几个以抵消 namespace/槽位过滤造成的损耗
        _CAND_K = 10

        # 快路径 A：ANN（HNSW/FAISS），O(log N)
        if self._ann_available() and len(self._embeddings) >= self.ann_min_size:
            for cand_key, score in self._ann_search(q_vec, top_k=_CAND_K):
                ans = self._accept_fuzzy_candidate(
                    cand_key, score, query, namespace
                )
                if ans is not None:
                    return ans
            return None

        # 回退路径 B：numpy 矩阵乘（N 较小时比建 ANN 索引更划算）
        if _np is not None:
            with self._embed_lock:
                if self._np_dirty or self._np_matrix is None:
                    self._rebuild_matrix_locked()
                mat = self._np_matrix
                keys = list(self._np_keys)
            if mat is not None and len(keys) > 0:
                q = _np.asarray(q_vec, dtype=_np.float32)
                # q 已归一，mat 已归一，余弦 = 点积
                sims = mat @ q                              # (N,)
                # 取 top-K（argsort 降序），逐个过校验
                k = min(_CAND_K, len(keys))
                # argpartition 比全量 argsort 快，但 K 很小时差别可忽略，
                # 这里用 argsort 保证顺序严格正确、代码更易读
                top_idx = _np.argsort(-sims)[:k]
                for i in top_idx:
                    i = int(i)
                    ans = self._accept_fuzzy_candidate(
                        keys[i], float(sims[i]), query, namespace
                    )
                    if ans is not None:
                        return ans
                return None

        # 回退路径 C：纯 Python（numpy 都没有时）
        with self._embed_lock:
            items = list(self._embeddings.items())
        scored = sorted(
            ((k, self._cosine(q_vec, v)) for k, v in items),
            key=lambda kv: -kv[1],
        )[:_CAND_K]
        for cand_key, score in scored:
            ans = self._accept_fuzzy_candidate(cand_key, score, query, namespace)
            if ans is not None:
                return ans
        return None

    def _index_embedding(self, key: str, text: str) -> None:
        """新增/更新条目时建 embedding 索引。key: norm_query, text: query

        - 优先从 self._emb_store 拿历史 embedding（避免重算）
        - 拿不到才调 BGE-M3 重算，且写回盘
        - 已归一化，便于后续矩阵点积即余弦
        - 标记矩阵脏 → 下次 fuzzy 时重建
        """
        if not self.enable_fuzzy:
            return
        vec = self._embed(text, cache_key=key)
        if vec is None:
            return
        with self._embed_lock:
            self._embeddings[key] = vec
            self._np_dirty = True
            self._ann_dirty = True

    # ============================================================
    #                       异步写 worker
    # ============================================================
    def _async_worker_loop(self) -> None:
        assert self._async_q is not None
        while True:
            try:
                op, args = self._async_q.get()
            except Exception:
                time.sleep(0.05)
                continue
            try:
                if op == _OP_SET:
                    # P0-1：事件 payload 从 (key, answer) 扩展为 (key, answer, ttl)。
                    # 兼容旧的 2 元 tuple，避免热升级期间队列里的存量事件报错。
                    if len(args) == 3:
                        key, answer, _ttl = args
                    else:
                        key, answer = args      # type: ignore[misc]
                        _ttl = None
                    self._sync_write_all_backends(key, answer, ttl=_ttl)
                elif op == _OP_DEL:
                    (key,) = args
                    self._sync_delete_all_backends(key)
                elif op == _OP_CLEAR:
                    self._sync_clear_all_backends()
            except Exception as e:
                self._warn(f"async worker error: {e}")
            finally:
                try:
                    self._async_q.task_done()
                except Exception:
                    pass

    def _enqueue(self, op: str, args: tuple[Any, ...]) -> None:
        """把写事件放入异步队列；队满则丢弃并记录。"""
        if self._async_q is None:
            return
        try:
            self._async_q.put_nowait((op, args))
        except queue.Full:
            self._incr("async_dropped")
            self._warn("async 写队列已满，事件被丢弃")

    # 低层：真正写各持久化后端（供同步/异步共用）
    def _sync_write_all_backends(
        self, key: str, answer: str, ttl: Optional[int] = None,
    ) -> None:
        """写入所有持久化后端。

        P0-1 改造：新增 `ttl` 参数支持 **per-entry TTL**。
            ttl=None  → 沿用实例级 self.ttl（向后兼容）
            ttl=<int> → 本条目单独的过期秒数
            ttl=0     → 视为"永不过期"（diskcache/redis 都用 None 表示）

        为什么必须支持 per-entry：改造前所有条目共享一个 30 天 TTL，
        「今天天气」和「量子计算是什么」被一视同仁。现在由
        cache_policy.decide_cacheability() 给出分级 TTL（6h/24h/30d），
        必须能逐条落地才有意义。
        """
        # ttl 显式传 None → 回退实例默认；传 0 → 明确表示永不过期
        eff_ttl = self.ttl if ttl is None else (ttl or None)
        for b in self.layers:
            if b == BACKEND_DISKCACHE and self._disk is not None:
                try:
                    self._disk.set(key, answer, expire=eff_ttl)
                except Exception as e:
                    self._warn(f"diskcache 写入失败: {e}")
            elif b == BACKEND_REDIS and self._redis is not None:
                try:
                    if eff_ttl:
                        self._redis.setex(self._redis_key(key), eff_ttl, answer)
                    else:
                        self._redis.set(self._redis_key(key), answer)
                except Exception as e:
                    self._warn(f"redis 写入失败: {e}")

    def _sync_delete_all_backends(self, key: str) -> None:
        for b in self.layers:
            if b == BACKEND_DISKCACHE and self._disk is not None:
                try: self._disk.delete(key)
                except Exception: pass
            elif b == BACKEND_REDIS and self._redis is not None:
                try: self._redis.delete(self._redis_key(key))
                except Exception: pass

    def _sync_clear_all_backends(self) -> None:
        for b in self.layers:
            if b == BACKEND_DISKCACHE and self._disk is not None:
                try: self._disk.clear()
                except Exception: pass
            elif b == BACKEND_REDIS and self._redis is not None:
                try:
                    for k in self._redis.scan_iter(f"{_REDIS_KEY_PREFIX}*"):
                        self._redis.delete(k)
                except Exception as e:
                    self._warn(f"redis 清空失败: {e}")

    # ============================================================
    #                          写接口
    # ============================================================
    def add(
        self,
        query: str,
        answer: str,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> bool:
        """写入一条 Q&A。

        Args:
            query:     原始 query（**不要传归一化后的**，槽位门禁需要原文）。
            answer:    答案文本。
            ttl:       本条目的过期秒数（P0-1）。
                       None → 沿用实例级 self.ttl（向后兼容旧调用）
                       int  → 本条独立 TTL，典型值来自
                              `cache_policy.decide_cacheability().ttl`
                       0    → 永不过期
            namespace: 租户/会话隔离命名空间（P0-3）。
                       None → 全局共享（与改造前完全一致）
                       str  → 只有同 namespace 的查询才能命中

        Returns:
            是否成功写入。

        ⚠️ 注意：本方法**不做**准入判定。是否该缓存、给多长 TTL，
        由调用方（`agent._archive_if_enabled`）先调
        `cache_policy.decide_cacheability()` 决定。
        这样分层的好处：QACache 保持"纯存储"语义，策略可独立演进/单测。
        """
        if not query or not answer:
            return False
        key = self._make_key(query, namespace)
        if not key:
            return False

        # L1 内存索引始终同步更新
        self._mem[key] = answer
        self._incr("writes")

        # P0-1：记录原始 query，供 fuzzy 命中时的槽位门禁比对
        self._remember_orig(key, query)

        # 建向量索引（enable_fuzzy=True 时才做）, 存储为:(model_name+key, query_emb)
        self._index_embedding(key, query)

        # 后端持久化：同步 or 异步
        if self.async_write:
            self._enqueue(_OP_SET, (key, answer, ttl))
        else:
            self._sync_write_all_backends(key, answer, ttl=ttl)
        return True

    def add_batch(
        self,
        pairs: dict[str, str] | Iterable[tuple[str, str]],
        ttl: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> int:
        iterator = pairs.items() if isinstance(pairs, dict) else pairs
        cnt = 0
        for q, a in iterator:
            if self.add(q, a, ttl=ttl, namespace=namespace):
                cnt += 1
        return cnt

    def load_from_dict(
        self,
        data: dict[str, str],
        replace: bool = False,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> int:
        if replace:
            self.clear()
        return self.add_batch(data, ttl=ttl, namespace=namespace)

    # ============================================================
    #                          读接口
    # ============================================================
    def get(self, query: str, namespace: Optional[str] = None) -> Optional[str]:
        """精确命中优先；miss 且启用 fuzzy 时做向量兜底。

        查找顺序：
            L1 内存 → L2/L3（按 layers 顺序）→ [fuzzy 兜底 + 槽位门禁]
        命中后回填上层，热点二次查询零 IO。

        Args:
            query:     用户原始 query。
            namespace: 租户/会话隔离命名空间（P0-3）。必须与写入时一致才能命中；
                       fuzzy 路径同样严格按 namespace 过滤候选。
        """
        key = self._make_key(query, namespace)
        if not key:
            return None

        # L1: 内存
        ans = self._mem.get(key)
        if ans is not None:
            self._incr("exact_hits")
            self._incr("layer_hits", "L1_mem")
            return ans

        # L2/L3: 按 layers 顺序查后端
        layer_names = ("L2", "L3")
        idx = 0
        for b in self.layers:
            if b == BACKEND_MEMORY: # 前面已使用self._mem缓存
                continue
            layer_tag = layer_names[idx] if idx < len(layer_names) else "L3"
            idx += 1
            if b == BACKEND_DISKCACHE and self._disk is not None:
                try:
                    ans = self._disk.get(key)
                    if ans:
                        self._mem[key] = ans   # 回填 L1
                        self._incr("exact_hits")
                        self._incr("layer_hits", layer_tag)
                        return ans
                except Exception as e:
                    self._warn(f"diskcache 读取失败: {e}")
            elif b == BACKEND_REDIS and self._redis is not None:
                try:
                    ans = self._redis.get(self._redis_key(key))
                    if ans:
                        self._mem[key] = ans   # 回填 L1
                        self._incr("exact_hits")
                        self._incr("layer_hits", layer_tag)
                        return ans
                except Exception as e:
                    self._warn(f"redis 读取失败: {e}")

        # Fuzzy 兜底：向量相似度 + namespace 过滤 + 槽位一致性门禁
        if self.enable_fuzzy:
            fuzzy_ans = self._fuzzy_lookup(query, namespace=namespace)
            if fuzzy_ans is not None:
                self._incr("fuzzy_hits")
                self._incr("layer_hits", "fuzzy")
                return fuzzy_ans

        self._incr("misses")
        return None

    def __contains__(self, query: str) -> bool:
        return self.get(query) is not None

    # ============================================================
    #                          删接口
    # ============================================================
    def remove(self, query: str, namespace: Optional[str] = None) -> bool:
        key = self._make_key(query, namespace)
        existed = key in self._mem
        self._mem.pop(key, None)
        self._orig.pop(key, None)          # P0-1：同步清理原文元数据
        if self._meta_store is not None:
            try:
                self._meta_store.delete(self._meta_key(key))
            except Exception:
                pass
        with self._embed_lock:
            self._embeddings.pop(key, None)
            self._np_dirty = True
            # ANN 软删除：把该 row 标记为 dead；累积过多再全量 rebuild
            if self._ann_index is not None and key in self._ann_keys:
                try:
                    row = self._ann_keys.index(key)
                    if 0 <= row < len(self._ann_alive) and self._ann_alive[row]:
                        self._ann_alive[row] = False
                        self._ann_removed += 1
                    # 软删累积 > 20% → 强制 rebuild
                    if self._ann_removed * 5 >= max(len(self._ann_keys), 1):
                        self._ann_dirty = True
                except ValueError:
                    self._ann_dirty = True
            else:
                self._ann_dirty = True
        # 同步删除 embedding 磁盘缓存（避免脏数据）
        if self._emb_store is not None:
            try:
                self._emb_store.delete(self._hash_cache_key(self.embed_model, key))
            except Exception:
                pass

        if self.async_write:
            self._enqueue(_OP_DEL, (key,))
        else:
            self._sync_delete_all_backends(key)
        return existed

    def clear(self) -> None:
        self._mem.clear()
        self._orig.clear()                 # P0-1：一并清理原文元数据
        if self._meta_store is not None:
            try:
                self._meta_store.clear()
            except Exception:
                pass
        with self._embed_lock:
            self._embeddings.clear()
            self._np_matrix = None
            self._np_keys = []
            self._np_dirty = False
            self._ann_index = None
            self._ann_keys = []
            self._ann_alive = []
            self._ann_removed = 0
            self._ann_dirty = False
        if self._emb_store is not None:
            try:
                self._emb_store.clear()
                # clear() 会连维度戳一起删掉。但内存里的 `_emb_dim` 仍然有效
                # （编码器没变），而 `_learn_emb_dim()` 在"维度一致"时会直接
                # return，不会补写戳 —— 那样下次冷启动就退化成"靠首个向量推断"。
                # 所以这里立刻把戳写回，保持磁盘与内存状态一致。
                if self._emb_dim:
                    self._save_emb_dim(self._emb_dim)
            except Exception:
                pass

        if self.async_write:
            self._enqueue(_OP_CLEAR, ())
        else:
            self._sync_clear_all_backends()

    # ============================================================
    #                          元信息
    # ============================================================
    def list_all(self, full: bool = False) -> dict[str, str]:
        if not full:
            return dict(self._mem)

        # 全量视图：优先从更权威的后端（layers 中最后一个持久化后端）读
        for b in reversed(self.layers):
            if b == BACKEND_DISKCACHE and self._disk is not None:
                try:
                    return {k: self._disk.get(k) for k in self._disk}
                except Exception as e:
                    self._warn(f"diskcache 全量读取失败: {e}")
            elif b == BACKEND_REDIS and self._redis is not None:
                try:
                    d: dict[str, str] = {}
                    for k in self._redis.scan_iter(f"{_REDIS_KEY_PREFIX}*"):
                        norm = k[len(_REDIS_KEY_PREFIX):] if k.startswith(_REDIS_KEY_PREFIX) else k
                        d[norm] = self._redis.get(k)
                    return d
                except Exception as e:
                    self._warn(f"redis 全量读取失败: {e}")
        return dict(self._mem)

    def reload_all(self) -> int:
        """把后端真身重新加载到 L1 内存索引。"""
        loaded: dict[str, str] = {}
        for b in self.layers:
            if b == BACKEND_DISKCACHE and self._disk is not None:
                try:
                    for k in self._disk:
                        v = self._disk.get(k)
                        if v is not None:
                            loaded.setdefault(k, v)
                except Exception as e:
                    self._warn(f"diskcache 预热失败: {e}")
            elif b == BACKEND_REDIS and self._redis is not None:
                try:
                    for k in self._redis.scan_iter(f"{_REDIS_KEY_PREFIX}*"):
                        norm = k[len(_REDIS_KEY_PREFIX):] if k.startswith(_REDIS_KEY_PREFIX) else k
                        v = self._redis.get(k)
                        if v is not None:
                            loaded.setdefault(norm, v)
                except Exception as e:
                    self._warn(f"redis 预热失败: {e}")
        if loaded:
            self._mem.clear()
            self._mem.update(loaded)
            # 若启用 fuzzy，重建 embedding 索引：
            #   - 优先从 self._emb_store 拿历史向量（零重算）
            #   - 拿不到才重算 + 写盘
            if self.enable_fuzzy:
                for k in loaded.keys():
                    # P0-1：这里用 _get_orig(k) 而不是裸 k。
                    # 原因：冷启动时 _meta 里若存有原始 query（带标点/大小写），
                    # 用它编码得到的向量与 add() 时写入的完全一致（都是原文编码），
                    # 避免"写入用原文编码、重载用 norm_key 编码"造成的向量漂移。
                    # 取不到 meta 时 _get_orig 会降级返回 norm_key，与旧行为一致。
                    self._index_embedding(k, self._get_orig(k))
                # 满载后制作矩阵一次，首次 fuzzy 查询 0 延迟
                with self._embed_lock:
                    self._rebuild_matrix_locked()   # 将 self._embeddings 拼成 (N, D) 矩阵。keys转成一个list
                    # 若数据量够，也顺便建 ANN (<500条使用矩阵乘， >500才)
                    if self._ann_available() and len(self._embeddings) >= self.ann_min_size:
                        self._rebuild_ann_locked()
        return len(self._mem)

    def size(self, full: bool = False) -> int:
        if not full:
            return len(self._mem)
        for b in reversed(self.layers):
            if b == BACKEND_DISKCACHE and self._disk is not None:
                try: return len(self._disk)
                except Exception: return len(self._mem)
            if b == BACKEND_REDIS and self._redis is not None:
                try:
                    return sum(1 for _ in self._redis.scan_iter(f"{_REDIS_KEY_PREFIX}*"))
                except Exception: return len(self._mem)
        return len(self._mem)

    def __len__(self) -> int:
        return len(self._mem)

    def __repr__(self) -> str:
        # 展示QACache实例的关键状态，（层数、缓存条目数、TTL、是否启用模糊匹配、是否异步写入）
        # >>> cache = QACache(...)
        # >>> cache          # ← 这里自动调用 __repr__
        # <QACache layers=3 size=0 ttl=3600 fuzzy=True async=False>
        # >>> repr(cache)    # ← python内建函数，这里自动调用 __repr__
        return (
            f"<QACache layers={self.layers} size={len(self._mem)} "
            f"ttl={self.ttl} fuzzy={self.enable_fuzzy}"
            f"@{self.fuzzy_threshold} slot_gate={self.enable_slot_gate} "
            f"async={self.async_write}>"
        )

    # ============================================================
    #                     命中统计 / 运维接口
    # ============================================================
    def stats(self) -> dict[str, Any]:
        """返回命中统计快照，含 hit_rate。"""
        with self._stats_lock:
            exact = int(self._stats["exact_hits"])
            fuzzy = int(self._stats["fuzzy_hits"])
            miss = int(self._stats["misses"])
            writes = int(self._stats["writes"])
            dropped = int(self._stats["async_dropped"])
            gate_rejects = int(self._stats.get("slot_gate_rejects", 0))
            layer_hits = dict(self._stats["layer_hits"])
        total_hits = exact + fuzzy
        total_reads = total_hits + miss
        snap: dict[str, Any] = {
            "exact_hits": exact,
            "fuzzy_hits": fuzzy,
            "misses": miss,
            "layer_hits": layer_hits,
            "writes": writes,
            "async_dropped": dropped,
            # P0-1 新增：槽位门禁拦截数 + 拦截率。
            # 运维意义：`slot_gate_rejects` 近似等于"如果没有这道门禁，
            # 系统会返回多少次错误答案"，是本次改造收益的直接量化指标。
            "slot_gate_rejects": gate_rejects,
            "slot_gate_reject_rate": (
                round(gate_rejects / (gate_rejects + fuzzy), 4)
                if (gate_rejects + fuzzy) else 0.0
            ),
            "total_reads": total_reads,
            "hit_rate": round(total_hits / total_reads, 4) if total_reads else 0.0,
            "fuzzy_rate": round(fuzzy / total_hits, 4) if total_hits else 0.0,
        }
        return snap

    def reset_stats(self) -> None:
        with self._stats_lock:
            self._stats = {
                "exact_hits": 0,
                "fuzzy_hits": 0,
                "misses": 0,
                "layer_hits": {"L1_mem": 0, "L2": 0, "L3": 0, "fuzzy": 0},
                "writes": 0,
                "async_dropped": 0,
                "slot_gate_rejects": 0,
            }

    def flush(self, timeout: Optional[float] = None) -> bool:
        """等待异步写队列落盘完成（关机/测试用）。"""
        if not self.async_write or self._async_q is None:
            return True
        try:
            if timeout is None:
                self._async_q.join()
                return True
            end = time.time() + timeout
            while not self._async_q.empty() and time.time() < end:
                time.sleep(0.02)
            return self._async_q.empty()
        except Exception:
            return False


# ============================================================
#       便捷函数 —— 后端固定的全局单例 QACache（懒加载）
# ============================================================

_redis_cache: Optional["QACache"] = None
_disk_cache: Optional["QACache"] = None

_redis_config: dict[str, Any] = {"redis_url": config.QA_REDIS_URL, "ttl": config.QA_CACHE_TTL}
_disk_config: dict[str, Any] = {"cache_dir": config.QA_CACHE_DIR, "ttl": config.QA_CACHE_TTL}


# ---------- Redis 后端 ---------- #
def configure_redis(
    redis_url: Optional[str] = None, ttl: Optional[int] = None
) -> None:
    global _redis_cache, _redis_config
    _redis_config["redis_url"] = redis_url
    _redis_config["ttl"] = ttl
    _redis_cache = None


def _get_redis_cache() -> "QACache":
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = QACache(
            backend=BACKEND_REDIS,
            redis_url=_redis_config["redis_url"],
            ttl=_redis_config["ttl"],
        )
    return _redis_cache


def add_search_reddis(query: str, answer: str) -> bool:
    return _get_redis_cache().add(query, answer)


def add_search_reddis_batch(
    pairs: dict[str, str] | Iterable[tuple[str, str]],
) -> int:
    return _get_redis_cache().add_batch(pairs)


def remove_search_reddis(query: str) -> bool:
    return _get_redis_cache().remove(query)


def remove_search_reddis_batch(queries: Iterable[str]) -> int:
    cache = _get_redis_cache()
    cnt = 0
    for q in queries:
        if cache.remove(q):
            cnt += 1
    return cnt


def clear_search_reddis() -> None:
    _get_redis_cache().clear()


def list_search_reddis(full: bool = True) -> dict[str, str]:
    return _get_redis_cache().list_all(full=full)


def reload_search_reddis() -> int:
    return _get_redis_cache().reload_all()


# ---------- DiskCache 后端 ---------- #
def configure_diskcache(
    cache_dir: Optional[str] = None, ttl: Optional[int] = None
) -> None:
    global _disk_cache, _disk_config
    _disk_config["cache_dir"] = cache_dir
    _disk_config["ttl"] = ttl
    _disk_cache = None


def _get_disk_cache() -> "QACache":
    global _disk_cache
    if _disk_cache is None:
        _disk_cache = QACache(
            backend=BACKEND_DISKCACHE,
            cache_dir=_disk_config["cache_dir"],
            ttl=_disk_config["ttl"],
        )
    return _disk_cache


def add_search_diskcache(query: str, answer: str) -> bool:
    return _get_disk_cache().add(query, answer)


def add_search_diskcache_batch(
    pairs: dict[str, str] | Iterable[tuple[str, str]],
) -> int:
    return _get_disk_cache().add_batch(pairs)


def remove_search_diskcache(query: str) -> bool:
    return _get_disk_cache().remove(query)


def remove_search_diskcache_batch(queries: Iterable[str]) -> int:
    cache = _get_disk_cache()
    cnt = 0
    for q in queries:
        if cache.remove(q):
            cnt += 1
    return cnt


def clear_search_diskcache() -> None:
    _get_disk_cache().clear()


def list_search_diskcache(full: bool = True) -> dict[str, str]:
    return _get_disk_cache().list_all(full=full)


def reload_search_diskcache() -> int:
    return _get_disk_cache().reload_all()


# ---------- 全局单例获取 ---------- #
def get_redis_singleton() -> "QACache":
    return _get_redis_cache()


def get_diskcache_singleton() -> "QACache":
    return _get_disk_cache()
