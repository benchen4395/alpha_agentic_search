# rag/vector_store.py
"""可插拔向量存储：优先 faiss（IndexFlatIP），无 faiss 时降级到 numpy 矩阵。

设计目标：
    - 极简 API：add / search / save / load / __len__
    - 元数据与向量同盘保存（jsonl + faiss/npz）
    - 已归一化向量 → 用 inner product 索引，等价 cosine
    - 支持增量 add（追加而非重建）
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional

try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None  # type: ignore

try:
    import faiss as _faiss  # type: ignore
except Exception:
    _faiss = None  # type: ignore


class VectorStore:
    """归一化向量 + 元数据的持久化存储。

    文件布局：
        {dir}/vectors.faiss     # faiss 后端时
        {dir}/vectors.npy       # numpy 后端时
        {dir}/metadata.jsonl    # 每行一条元数据 {"id": ..., "text": ..., ...}
    """

    def __init__(self, index_dir: str, dim: int):
        if _np is None:
            raise RuntimeError("VectorStore 需要 numpy，请 `pip install numpy`。")
        self.index_dir = index_dir
        self.dim = dim
        os.makedirs(index_dir, exist_ok=True)
        self._lock = threading.Lock()

        # faiss 优先
        self._use_faiss = _faiss is not None
        self._index = None            # faiss index
        self._matrix: Optional["_np.ndarray"] = None   # numpy fallback (N, dim)
        self._metadata: list[dict] = []
        self._load()

    # ---------- 持久化 ---------- #
    def _faiss_path(self) -> str: return os.path.join(self.index_dir, "vectors.faiss")
    def _npy_path(self)   -> str: return os.path.join(self.index_dir, "vectors.npy")
    def _meta_path(self)  -> str: return os.path.join(self.index_dir, "metadata.jsonl")

    def _load(self) -> None:
        # metadata
        if os.path.exists(self._meta_path()):
            with open(self._meta_path(), "r", encoding="utf-8") as f:
                self._metadata = [json.loads(line) for line in f if line.strip()]
        # vectors
        if self._use_faiss and os.path.exists(self._faiss_path()):
            self._index = _faiss.read_index(self._faiss_path())
        elif os.path.exists(self._npy_path()):
            self._matrix = _np.load(self._npy_path())
        # 冷启动
        if self._use_faiss and self._index is None:
            self._index = _faiss.IndexFlatIP(self.dim)

    def save(self) -> None:
        with self._lock:
            with open(self._meta_path(), "w", encoding="utf-8") as f:
                for m in self._metadata:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            if self._use_faiss and self._index is not None:
                _faiss.write_index(self._index, self._faiss_path())
            elif self._matrix is not None:
                _np.save(self._npy_path(), self._matrix)

    # ---------- 增删查 ---------- #
    def add(self, vectors: list[list[float]], metadatas: list[dict]) -> None:
        assert len(vectors) == len(metadatas)
        if not vectors:
            return
        arr = _np.asarray(vectors, dtype=_np.float32)
        with self._lock:
            if self._use_faiss:
                assert self._index is not None
                self._index.add(arr)
            else:
                if self._matrix is None:
                    self._matrix = arr
                else:
                    self._matrix = _np.concatenate([self._matrix, arr], axis=0)
            self._metadata.extend(metadatas)

    def search(self, query_vec: list[float], top_k: int = 5) -> list[tuple[dict, float]]:
        """返回 [(metadata, score), ...]，score 为 cosine（已归一化）。"""
        if not query_vec:
            return []
        q = _np.asarray([query_vec], dtype=_np.float32)
        with self._lock:
            if self._use_faiss and self._index is not None and self._index.ntotal > 0:
                k = min(top_k, self._index.ntotal)
                sims, idxs = self._index.search(q, k)
                sims, idxs = sims[0], idxs[0]
            elif self._matrix is not None and len(self._matrix) > 0:
                sims_all = self._matrix @ q[0]
                k = min(top_k, len(sims_all))
                idxs = _np.argsort(-sims_all)[:k]
                sims = sims_all[idxs]
            else:
                return []
            out: list[tuple[dict, float]] = []
            for i, s in zip(idxs, sims):
                i = int(i)
                if 0 <= i < len(self._metadata):
                    out.append((self._metadata[i], float(s)))
            return out

    def __len__(self) -> int:
        with self._lock:
            if self._use_faiss and self._index is not None:
                return int(self._index.ntotal)
            if self._matrix is not None:
                return int(self._matrix.shape[0])
            return 0
