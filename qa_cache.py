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
                        fuzzy_threshold=0.92, embed_model="bge-m3")

    异步写后端：
        cache = QACache(backend="diskcache", async_write=True)
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
        fuzzy_threshold: float = 0.8,
        embed_model: str = "bge-m3",
        async_write: bool = False,
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
        self._mem: dict[str, str] = {}      # L1 内存（最快）
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
        }

        # ---- 向量模糊匹配 ----
        # D1：向量编码统一走 FlagEmbedding BGE-M3（与 L2/L5 同一向量空间）。
        self.enable_fuzzy = bool(enable_fuzzy)
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.embed_model = embed_model  # 只有在add, reloaded_all时才会触发写入
        # 期望向量维度：用于过滤历史脏缓存（非当前模型维度的向量）。
        # 读 rag 侧统一配置（BGE-M3 = 1024）；取不到则设 None（不做维度校验，退化为原行为）。
        try:
            from rag import config as _rag_config
            self._expected_dim: Optional[int] = int(_rag_config.RAG_EMBED_DIM)
        except Exception:
            self._expected_dim = None
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
            f"mem_size={self._num_mem}, emb_size={len(self._embeddings)}")

    # ---------- helpers ---------- #
    def _warn(self, msg: str) -> None:
        print(f"[qa_cache] {msg}")

    @staticmethod
    def _redis_key(norm_q: str) -> str:
        return f"{_REDIS_KEY_PREFIX}{norm_q}"

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

    def _embed(self, text: str, cache_key: Optional[str] = None) -> Optional[list[float]]:
        """获取text的embedding（已归一化）。如果磁盘存储了(开机热加载)就直接读取，如果磁盘未存储，就计算emb并存储在磁盘中

        优先级：
            1. 磁盘持久化存储 self._emb_store（避免重算）
            2. 调 BGE-M3 重算 → 写盘
        cache_key: 若传入（通常是 norm_key）则启用磁盘缓存；
                   临时 query（fuzzy 查询）不传，仅实时算。
        """
        if not self.enable_fuzzy:
            return None

        # 1) 磁盘命中
        if cache_key is not None and self._emb_store is not None:
            try:
                # 根据model_name+cache_key的hash值，从_embed_store中获取对应的embeeding值
                cached = self._emb_store.get(self._hash_cache_key(self.embed_model, cache_key))
                if cached is not None:
                    cached = list(cached)
                    # 维度校验：历史脏缓存可能是非当前模型维度（如 4/8/128 维），
                    # 读到就丢弃并往下重算，避免污染 fuzzy 矩阵导致 np.asarray 崩溃。
                    if self._expected_dim is None or len(cached) == self._expected_dim:
                        return cached
                    self._warn(
                        f"忽略维度异常的磁盘 embedding（{len(cached)}≠{self._expected_dim}），重算"
                    )
            except Exception as e:
                self._warn(f"embedding 磁盘读取失败: {e}")

        # 2) 磁盘没有命中，走 BGE-M3 实时编码（与 L2/L5 同一向量空间）
        #    BGE-M3 原生 batch 编码，这里包单元素 batch；已 L2 归一化
        arr = _bge.encode([text], normalize=True)
        vec: Optional[list[float]] = arr[0].tolist()
        if not vec:
            return None

        # 预归一化：存盘/内存都存 unit vector，后续 cos = dot
        vec = self._l2_normalize(vec)

        # 3) 将本次读取的内容，回写磁盘
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

        维度防御：历史遗留的磁盘缓存里可能混入**非当前模型维度**的脏向量
        （如早期用别的编码/降维配置写入的 4/8/16/128 维向量）。若直接把长度
        不齐的向量喂给 ``np.asarray`` 会抛
        ``ValueError: inhomogeneous shape``。这里以"出现次数最多的维度"为准
        （正常即 BGE-M3 的 1024），过滤掉维度不符的坏条目，避免整个缓存不可用。
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

    def _ann_search(self, q_vec: list[float]) -> Optional[tuple[str, float]]:
        """在 ANN 索引中查最相似的活条目，返回 (norm_key, score)。"""
        if not self._ann_available() or _np is None:
            return None
        with self._embed_lock:
            if self._ann_dirty or self._ann_index is None:
                self._rebuild_ann_locked()
            index = self._ann_index
            keys = list(self._ann_keys)
            alive = list(self._ann_alive)
        if index is None or not keys:
            return None

        q = _np.asarray([q_vec], dtype=_np.float32)
        # 取 top-k 里第一个仍 alive 的（应对软删除）
        k = min(5, len(keys))
        if self.ann_backend == "faiss":
            sims, idxs = index.search(q, k)
            sims, idxs = sims[0], idxs[0]
        else:  # hnswlib
            labels, dists = index.knn_query(q, k=k)
            idxs, sims = labels[0], dists[0]  # ip space 返回的就是内积

        for row, s in zip(idxs, sims):
            if row < 0 or row >= len(keys):
                continue
            if alive[row]:
                return keys[row], float(s)
        return None

    def _fuzzy_lookup(self, query: str) -> Optional[str]:
        """矩阵化模糊匹配：一次 numpy 矩乘完成所有余弦计算。"""
        if not self.enable_fuzzy or not self._embeddings:
            return None
        # 查询向量不存盘（实时 query）；已归一化
        q_vec = self._embed(query, cache_key=None)
        if q_vec is None:
            return None

        # 快路径 A：ANN（HNSW/FAISS），O(log N)
        if self._ann_available() and len(self._embeddings) >= self.ann_min_size:
            found = self._ann_search(q_vec)
            if found is not None:
                best_key, best_score = found
                if best_score >= self.fuzzy_threshold:
                    ans = self._mem.get(best_key)
                    if ans is not None:
                        if self.verbose:
                            self._warn(
                                f"fuzzy hit '{best_key}' score={best_score:.3f} "
                                f"(ann:{self.ann_backend})"
                            )
                        return ans
                return None

        # 回退路径：numpy 矩阵乘
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
                best_idx = int(_np.argmax(sims))
                best_score = float(sims[best_idx])
                best_key = keys[best_idx]
                if best_score >= self.fuzzy_threshold:
                    ans = self._mem.get(best_key)
                    if ans is not None:
                        if self.verbose:
                            self._warn(f"fuzzy hit '{best_key}' score={best_score:.3f} (np)")
                        return ans
                return None

        # 回退路径：纯 Python for
        best_key, best_score = None, -1.0
        with self._embed_lock:
            items = list(self._embeddings.items())
        for key, vec in items:
            s = self._cosine(q_vec, vec)
            if s > best_score:
                best_key, best_score = key, s
        if best_key is not None and best_score >= self.fuzzy_threshold:
            ans = self._mem.get(best_key)
            if ans is not None:
                if self.verbose:
                    self._warn(f"fuzzy hit '{best_key}' score={best_score:.3f} (py)")
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
                    key, answer = args
                    self._sync_write_all_backends(key, answer)
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
    def _sync_write_all_backends(self, key: str, answer: str) -> None:
        for b in self.layers:
            if b == BACKEND_DISKCACHE and self._disk is not None:
                try:
                    self._disk.set(key, answer, expire=self.ttl)
                except Exception as e:
                    self._warn(f"diskcache 写入失败: {e}")
            elif b == BACKEND_REDIS and self._redis is not None:
                try:
                    if self.ttl:
                        self._redis.setex(self._redis_key(key), self.ttl, answer)
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
    def add(self, query: str, answer: str) -> bool:
        if not query or not answer:
            return False
        key = normalize_for_qa(query)
        if not key:
            return False

        # L1 内存索引始终同步更新
        self._mem[key] = answer
        self._incr("writes")

        # 建向量索引（enable_fuzzy=True 时才做）, 存储为:(model_name+key, query_emb)
        self._index_embedding(key, query)

        # 后端持久化：同步 or 异步
        if self.async_write:
            self._enqueue(_OP_SET, (key, answer))
        else:
            self._sync_write_all_backends(key, answer)
        return True

    def add_batch(
        self, pairs: dict[str, str] | Iterable[tuple[str, str]]
    ) -> int:
        iterator = pairs.items() if isinstance(pairs, dict) else pairs
        cnt = 0
        for q, a in iterator:
            if self.add(q, a):
                cnt += 1
        return cnt

    def load_from_dict(self, data: dict[str, str], replace: bool = False) -> int:
        if replace:
            self.clear()
        return self.add_batch(data)

    # ============================================================
    #                          读接口
    # ============================================================
    def get(self, query: str) -> Optional[str]:
        """精确命中优先；miss 且启用 fuzzy 时做向量兜底。

        查找顺序：
            L1 内存 → L2/L3（按 layers 顺序）→ [fuzzy 兜底]
        命中后回填上层，热点二次查询零 IO。
        """
        key = normalize_for_qa(query)
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

        # Fuzzy 兜底：向量相似度
        if self.enable_fuzzy:
            fuzzy_ans = self._fuzzy_lookup(query)
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
    def remove(self, query: str) -> bool:
        key = normalize_for_qa(query)
        existed = key in self._mem
        self._mem.pop(key, None)
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
                    self._index_embedding(k, k)
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
            f"ttl={self.ttl} fuzzy={self.enable_fuzzy} async={self.async_write}>"
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
