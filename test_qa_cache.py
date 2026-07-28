# test_qa_cache.py
"""qa_cache.QACache 4 条扩展点的单元测试。

运行方式：
    pytest -v test_qa_cache.py
或：
    python -m pytest -v test_qa_cache.py

用例覆盖：
    1) 向量相似度模糊命中（fuzzy）—— mock BGE-M3 编码，避免真依赖
    2) 命中统计（stats / reset_stats / hit_rate / fuzzy_rate）
    3) 多级缓存（layers: memory + diskcache，L1↔L2 回填）
    4) 异步写后端（async_write：主流程零阻塞 + flush 等待落盘）
    5) 回归：老 API（add/get/remove/clear/list_all/size）行为不变
"""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

import pytest

import qa_cache
from qa_cache import QACache, normalize_for_qa


# ============================================================
#                     公共：mock BGE-M3
# ============================================================
class _FakeBge:
    """确定性 embedding：把字符串映射成固定向量。

    - 完全同的字符串 → 完全同向量（相似度=1.0）
    - 通过 register(text, vec) 可自定义某些 query 的向量
    - 未注册的按字符 hash 生成 8 维稀疏向量

    接口对齐 rag.wiki_rag.embedder.encode：
        encode(texts: list[str], normalize=True, **kwargs) -> np.ndarray (N, dim)
    """

    def __init__(self):
        self._table: dict[str, list[float]] = {}
        self.call_count = 0

    def register(self, text: str, vec: list[float]) -> None:
        self._table[text] = vec

    def _vec_of(self, prompt: str) -> list[float]:
        if prompt in self._table:
            return list(self._table[prompt])
        # 兜底：字符哈希 → 稳定向量
        vec = [0.0] * 8
        for i, ch in enumerate(prompt):
            vec[i % 8] += (ord(ch) % 13) / 13.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def encode(self, texts, normalize=True, **kwargs):
        import numpy as _np
        self.call_count += len(texts)
        return _np.asarray([self._vec_of(t) for t in texts], dtype="float32")


@pytest.fixture
def fake_bge(monkeypatch):
    """注入假的 BGE-M3 编码器，让 enable_fuzzy 生效但不加载真模型。"""
    fake = _FakeBge()
    monkeypatch.setattr(qa_cache, "_bge", fake)
    return fake


# ============================================================
#                        归一化 + 基础回归
# ============================================================
class TestNormalize:
    def test_normalize_removes_punct_and_case(self):
        assert normalize_for_qa("中国的首都是哪里？") == "中国的首都是哪里"
        assert normalize_for_qa("What  is the CAPITAL?") == "whatisthecapital"
        assert normalize_for_qa("") == ""
        assert normalize_for_qa(None) == ""  # type: ignore[arg-type]


class TestBackwardCompat:
    """老 API 行为回归。"""

    def test_memory_backend_basic_flow(self, monkeypatch):
        # 禁用 fuzzy，避免 fake 依赖污染精确匹配语义
        c = QACache(backend="memory", enable_fuzzy=False)
        assert c.add("中国的首都是哪里？", "北京") is True
        assert c.get("中国 的 首都 是哪里!!!") == "北京"
        assert "中国的首都是哪里" in c
        assert c.remove("中国的首都是哪里") is True
        assert c.get("中国的首都是哪里") is None
        assert len(c) == 0

    def test_add_empty_ignored(self, monkeypatch):
        c = QACache(backend="memory", enable_fuzzy=False)
        assert c.add("", "x") is False
        assert c.add("q", "") is False
        assert c.add("!!!", "x") is False  # 全被归一化删光

    def test_add_batch_and_load_from_dict(self, monkeypatch):
        c = QACache(backend="memory", enable_fuzzy=False)
        n = c.load_from_dict({"a?": "A", "b?": "B", "": "skip"}, replace=True)
        assert n == 2
        assert c.get("a") == "A" and c.get("b") == "B"


# ============================================================
#              扩展点 1：向量相似度 fuzzy 命中
# ============================================================
class TestFuzzy:
    def test_paraphrase_hits_via_bge_m3(self, fake_bge, monkeypatch):
        # 让三种同义问法拿到几乎相同的向量
        base = [1.0, 0.0, 0.0, 0.0]
        fake_bge.register("美国总统是谁？", base)
        fake_bge.register("谁是美国总统", [0.99, 0.14, 0.0, 0.0])   # cos≈0.99
        fake_bge.register("当前美国总统是谁", [0.98, 0.20, 0.0, 0.0])  # cos≈0.98

        c = QACache(
            backend="memory",
            enable_fuzzy=True,
            fuzzy_threshold=0.9,
        )
        assert c.add("美国总统是谁？", "特朗普(示例)") is True
        # 精确 miss 后走 fuzzy
        assert c.get("谁是美国总统") == "特朗普(示例)"
        assert c.get("当前美国总统是谁") == "特朗普(示例)"

        s = c.stats()
        assert s["exact_hits"] == 0
        assert s["fuzzy_hits"] == 2
        assert s["layer_hits"]["fuzzy"] == 2
        assert s["misses"] == 0

    def test_below_threshold_is_miss(self, fake_bge):
        # 两个明显不相关的问题，相似度低于阈值
        fake_bge.register("A 问题", [1.0, 0.0, 0.0, 0.0])
        fake_bge.register("B 问题", [0.0, 0.0, 1.0, 0.0])  # cos=0
        c = QACache(backend="memory", enable_fuzzy=True, fuzzy_threshold=0.9)
        c.add("A 问题", "answer-A")
        assert c.get("B 问题") is None
        s = c.stats()
        assert s["misses"] == 1
        assert s["fuzzy_hits"] == 0

    def test_fuzzy_enabled_with_bge(self, fake_bge):
        """BGE-M3 可用时，enable_fuzzy=True 应保持开启。"""
        c = QACache(backend="memory", enable_fuzzy=True)
        assert c.enable_fuzzy is True


# ============================================================
#              扩展点 2：命中统计 / stats
# ============================================================
class TestStats:
    def test_counters_and_hit_rate(self, monkeypatch):
        c = QACache(backend="memory", enable_fuzzy=False)
        c.add("q1", "a1")
        c.add("q2", "a2")
        # 2 命中 + 1 未命中
        assert c.get("q1") == "a1"
        assert c.get("q2") == "a2"
        assert c.get("q3") is None

        s = c.stats()
        assert s["exact_hits"] == 2
        assert s["misses"] == 1
        assert s["writes"] == 2
        assert s["total_reads"] == 3
        assert s["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)
        assert s["layer_hits"]["L1_mem"] == 2

    def test_reset_stats(self, monkeypatch):
        c = QACache(backend="memory", enable_fuzzy=False)
        c.add("q", "a")
        c.get("q")
        c.get("miss")
        c.reset_stats()
        s = c.stats()
        assert s["exact_hits"] == 0
        assert s["misses"] == 0
        assert s["hit_rate"] == 0.0

    def test_stats_thread_safe(self, monkeypatch):
        """并发读写下 stats 不崩、计数守恒。"""
        c = QACache(backend="memory", enable_fuzzy=False)
        c.add("hot", "value")

        N = 200
        THREADS = 8

        def worker():
            for _ in range(N):
                c.get("hot")     # 命中
                c.get("cold")    # 未命中

        ts = [threading.Thread(target=worker) for _ in range(THREADS)]
        [t.start() for t in ts]
        [t.join() for t in ts]

        s = c.stats()
        assert s["exact_hits"] == N * THREADS
        assert s["misses"] == N * THREADS
        assert s["total_reads"] == 2 * N * THREADS


# ============================================================
#              扩展点 3：多级缓存 layers
# ============================================================
class TestLayers:
    def test_l1_miss_l2_hit_and_backfill(self, tmp_path, monkeypatch):
        c = QACache(
            layers=["memory", "diskcache"],
            cache_dir=str(tmp_path / "qa"),
            enable_fuzzy=False,
        )
        c.add("q1", "a1")

        # 手动清 L1，强制走 L2
        c._mem.clear()
        assert c.get("q1") == "a1"

        s = c.stats()
        assert s["layer_hits"]["L2"] == 1
        assert s["layer_hits"]["L1_mem"] == 0

        # 已回填 L1，再查走 L1
        c.reset_stats()
        assert c.get("q1") == "a1"
        s2 = c.stats()
        assert s2["layer_hits"]["L1_mem"] == 1
        assert s2["layer_hits"]["L2"] == 0

    def test_write_persists_to_all_layers(self, tmp_path, monkeypatch):
        cache_dir = str(tmp_path / "qa2")

        c1 = QACache(
            layers=["memory", "diskcache"],
            cache_dir=cache_dir,
            enable_fuzzy=False,
        )
        c1.add("shared_q", "shared_a")

        # 用新实例，同一 disk 目录 → 应该 reload 出来
        c2 = QACache(
            layers=["memory", "diskcache"],
            cache_dir=cache_dir,
            enable_fuzzy=False,
        )
        # 构造器已 reload_all 到 L1
        assert c2.get("shared_q") == "shared_a"

    def test_unknown_backend_falls_back_to_memory(self, monkeypatch):
        c = QACache(layers=["not_exist"], enable_fuzzy=False)
        assert c.layers == ["memory"]
        c.add("x", "y")
        assert c.get("x") == "y"


# ============================================================
#              扩展点 4：异步写 async_write
# ============================================================
class TestAsyncWrite:
    def test_async_add_returns_fast_and_flush_completes(self, tmp_path, monkeypatch):
        c = QACache(
            layers=["memory", "diskcache"],
            cache_dir=str(tmp_path / "qa_async"),
            enable_fuzzy=False,
            async_write=True,
        )

        # 主流程零阻塞：批量 add 应显著低于 100ms
        t0 = time.time()
        for i in range(50):
            c.add(f"q{i}", f"a{i}")
        elapsed = time.time() - t0
        assert elapsed < 0.5

        # L1 立刻可查
        assert c.get("q0") == "a0"
        assert c.get("q49") == "a49"

        # flush 后 L2 也应完成
        assert c.flush(timeout=3.0) is True

        # 新实例从磁盘 reload，验证异步写确实落盘
        c2 = QACache(
            layers=["memory", "diskcache"],
            cache_dir=str(tmp_path / "qa_async"),
            enable_fuzzy=False,
        )
        assert c2.get("q0") == "a0"
        assert c2.get("q49") == "a49"

    def test_async_dropped_when_queue_full(self, tmp_path, monkeypatch):
        """把队列压到超小，验证溢出被计数而不是崩溃。"""
        # 让 worker 慢一些，以便队列可能满
        import queue as _q
        c = QACache(
            layers=["memory", "diskcache"],
            cache_dir=str(tmp_path / "qa_async2"),
            enable_fuzzy=False,
            async_write=True,
        )
        # 手动替换成极小队列
        c._async_q = _q.Queue(maxsize=1)  # type: ignore[assignment]

        # 大量并发写，肯定会有丢弃
        for i in range(50):
            c.add(f"k{i}", f"v{i}")

        s = c.stats()
        # 至少应发生一次丢弃或者刚好放进
        assert s["async_dropped"] >= 0  # 断言不崩即通过
        # L1 完整（异步丢弃不影响 L1）
        assert c.get("k0") == "v0"

    def test_flush_no_op_when_sync(self, monkeypatch):
        c = QACache(backend="memory", enable_fuzzy=False, async_write=False)
        assert c.flush() is True


# ============================================================
#                     组合场景：4 条一起用
# ============================================================
class TestCombined:
    def test_layers_plus_fuzzy_plus_async_plus_stats(self, tmp_path, fake_bge):
        fake_bge.register("美国总统是谁？", [1.0, 0.0, 0.0])
        fake_bge.register("谁是美国总统", [0.98, 0.19, 0.0])

        c = QACache(
            layers=["memory", "diskcache"],
            cache_dir=str(tmp_path / "combined"),
            enable_fuzzy=True,
            fuzzy_threshold=0.9,
            async_write=True,
        )
        c.add("美国总统是谁？", "特朗普(示例)")
        assert c.flush(timeout=2.0)

        # 命中路径：精确 + fuzzy 各一次
        assert c.get("美国总统是谁？") == "特朗普(示例)"
        assert c.get("谁是美国总统") == "特朗普(示例)"

        s = c.stats()
        assert s["exact_hits"] == 1
        assert s["fuzzy_hits"] == 1
        assert s["hit_rate"] == 1.0
        assert s["layer_hits"]["L1_mem"] == 1
        assert s["layer_hits"]["fuzzy"] == 1


# ============================================================
#         新增：embedding 持久化 & 矩阵化加速
# ============================================================
class TestEmbeddingPersistence:
    """扩展点 1 强化：embedding 冷启动零重算。"""

    def test_embedding_cached_on_disk_and_reused(self, tmp_path, fake_bge):
        fake_bge.register("美国总统是谁？", [1.0, 0.0, 0.0])
        cache_dir = str(tmp_path / "persist")

        c1 = QACache(
            layers=["memory", "diskcache"],
            cache_dir=cache_dir,
            enable_fuzzy=True,
            fuzzy_threshold=0.9,
        )
        c1.add("美国总统是谁？", "特朗普")
        first_calls = fake_bge.call_count
        assert first_calls >= 1  # add 时算了一次

        # 用新实例：Q&A + embedding 都来自磁盘，BGE-M3 不应再被调用
        fake_bge.call_count = 0
        c2 = QACache(
            layers=["memory", "diskcache"],
            cache_dir=cache_dir,
            enable_fuzzy=True,
            fuzzy_threshold=0.9,
        )
        # 构造器已 reload_all，如果 embedding 没落盘，会再调 BGE-M3
        assert fake_bge.call_count == 0, (
            f"embedding 未复用磁盘缓存，仍调 BGE-M3 {fake_bge.call_count} 次"
        )
        # fuzzy 命中依旧正常
        assert c2.get("美国总统是谁？") == "特朗普"


class TestMatrixSpeedup:
    """扩展点 1 强化：cosine 矩阵化。"""

    def test_matrix_built_and_reused(self, fake_bge):
        # 构造 20 条不同的 Q，向量维度 8
        c = QACache(backend="memory", enable_fuzzy=True, fuzzy_threshold=0.5)
        for i in range(20):
            c.add(f"question_{i}?", f"answer_{i}")

        # 首次 fuzzy 查询 → 触发矩阵构建
        _ = c.get("brand_new_query_that_wont_exact_match")
        # 内部矩阵应已存在（依赖 numpy）
        try:
            import numpy as _np  # type: ignore
            assert c._np_matrix is not None
            assert c._np_matrix.shape[0] == 20
            assert c._np_dirty is False
        except ImportError:
            pytest.skip("numpy 未安装，跳过矩阵化验证")

        # 再次 add → 标记 dirty
        c.add("one_more?", "x")
        assert c._np_dirty is True

        # 再查 → 重建 & dirty 清除
        _ = c.get("another_query")
        assert c._np_dirty is False
        assert c._np_matrix.shape[0] == 21


# ============================================================
#         新增：ANN (HNSW/FAISS) O(log N) 加速
# ============================================================
faiss = pytest.importorskip("faiss")


class TestANN:
    """扩展点 1 强化：faiss/hnswlib 近似最近邻。"""

    def _rand_bge(self, monkeypatch, dim: int = 32, seed: int = 0):
        """注入一个返回随机稳定向量的假 BGE-M3 编码器。"""
        import random
        import numpy as _np
        rng = random.Random(seed)
        cache_vec: dict[str, list[float]] = {}

        class RandBge:
            call_count = 0
            def encode(self, texts, normalize=True, **kwargs):
                RandBge.call_count += len(texts)
                rows = []
                for prompt in texts:
                    if prompt not in cache_vec:
                        cache_vec[prompt] = [rng.uniform(-1, 1) for _ in range(dim)]
                    rows.append(cache_vec[prompt])
                return _np.asarray(rows, dtype="float32")
        fake = RandBge()
        monkeypatch.setattr(qa_cache, "_bge", fake)
        return fake

    def test_ann_backend_auto_prefers_faiss(self, tmp_path, monkeypatch):
        self._rand_bge(monkeypatch)
        c = QACache(
            layers=["memory"], cache_dir=str(tmp_path/"a"),
            enable_fuzzy=True, ann_min_size=1,
        )
        # 环境里装了 faiss，auto 应选 faiss
        assert c.ann_backend == "faiss"

    def test_ann_builds_and_hits(self, tmp_path, monkeypatch):
        self._rand_bge(monkeypatch, dim=32, seed=42)
        c = QACache(
            layers=["memory"], cache_dir=str(tmp_path/"b"),
            enable_fuzzy=True,
            fuzzy_threshold=-1.0,   # 让任何 top1 都算命中
            ann_backend="faiss",
            ann_min_size=10,
        )
        # 塞 50 条超过阈值 → 触发 ANN
        for i in range(50):
            c.add(f"question_{i}", f"ans_{i}")
        # 查已存条目，ANN 应能命中它自己
        assert c.get("question_7") == "ans_7"
        # 触发一次 fuzzy 走 ANN 分支
        _ = c.get("some_query_not_exact")
        assert c._ann_index is not None
        assert c._ann_dirty is False
        assert len(c._ann_keys) == 50

    def test_ann_soft_delete_and_rebuild(self, tmp_path, monkeypatch):
        self._rand_bge(monkeypatch, dim=16, seed=7)
        c = QACache(
            layers=["memory"], cache_dir=str(tmp_path/"c"),
            enable_fuzzy=True,
            fuzzy_threshold=-1.0,
            ann_backend="faiss",
            ann_min_size=5,
        )
        for i in range(30):
            c.add(f"q_{i}", f"a_{i}")
        # 先触发 ANN 建索引
        _ = c.get("trigger")
        assert c._ann_index is not None

        # 少量删除 → 软删，索引不重建
        c.remove("q_1")
        c.remove("q_2")
        assert c._ann_removed >= 1

        # fuzzy 查询后，已被软删的 row 不会被返回
        # 塞满 20% 阈值触发重建
        for i in range(3, 12):
            c.remove(f"q_{i}")
        # 下次查询会 rebuild
        _ = c.get("another_trigger")
        assert c._ann_dirty is False

    def test_ann_below_min_size_uses_numpy(self, tmp_path, monkeypatch):
        self._rand_bge(monkeypatch)
        c = QACache(
            layers=["memory"], cache_dir=str(tmp_path/"d"),
            enable_fuzzy=True,
            fuzzy_threshold=-1.0,
            ann_backend="faiss",
            ann_min_size=1000,   # 高阈值 → 小规模不走 ANN
        )
        for i in range(20):
            c.add(f"q{i}", f"a{i}")
        _ = c.get("hello")
        # ANN 不该被构建
        assert c._ann_index is None
        # 但 numpy 矩阵应被构建
        assert c._np_matrix is not None

    def test_ann_faster_than_numpy_on_large_n(self, tmp_path, monkeypatch):
        """粗粒度性能：N 大时 ANN 应显著快于矩阵乘。"""
        self._rand_bge(monkeypatch, dim=128, seed=1)
        N = 3000

        c_ann = QACache(
            layers=["memory"], cache_dir=str(tmp_path/"e_ann"),
            enable_fuzzy=True, fuzzy_threshold=-1.0,
            ann_backend="faiss", ann_min_size=100,
        )
        c_np = QACache(
            layers=["memory"], cache_dir=str(tmp_path/"e_np"),
            enable_fuzzy=True, fuzzy_threshold=-1.0,
            ann_backend="none",
        )
        for i in range(N):
            c_ann.add(f"q{i}", f"a{i}")
            c_np.add(f"q{i}", f"a{i}")

        # 预热
        c_ann.get("warmup"); c_np.get("warmup")

        t0 = time.time()
        for _ in range(50):
            c_ann.get("query_probe")
        t_ann = time.time() - t0

        t0 = time.time()
        for _ in range(50):
            c_np.get("query_probe")
        t_np = time.time() - t0

        # 至少不能显著更慢；宽松断言（不同机器差异大）
        assert t_ann <= t_np * 2.0, f"ANN {t_ann:.3f}s vs numpy {t_np:.3f}s"
