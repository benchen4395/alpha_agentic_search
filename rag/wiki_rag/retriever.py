"""基于本地 FAISS 索引的稠密检索。

:class:`WikiRetriever` 打包了两份由离线流水线产出的、行号严格对齐的产物：

- ``index_file``: "data/wiki_zh_emb_hnsw.faiss"   —— FAISS 索引（IVF-Flat 或 HNSW-Flat，内积度量）。
- ``chunks_file``  —— JSONL 元数据，其第 *i* 行对应索引第 *i* 行的向量。

对齐关系由 ``05_build_chunks_and_embed`` 在构建阶段保证，所以运行时
"搜索索引 + 切片元数据"就能拿到一致的命中结果。
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np

from .config import load_config
from .embedder import encode, get_model


class WikiRetriever:
    """加载 FAISS 索引与其对应的每行元数据，供查询时联合使用。"""

    def __init__(self, config_path: str | Path | None = None):
        cfg = load_config(config_path)
        self.cfg = cfg
        paths = cfg["paths"]

        print("[wiki] loading FAISS index ...")
        self.index = faiss.read_index(str(paths["index_file"]))     # "data/wiki_zh_emb_hnsw.faiss"

        # `nprobe` 只在 IVF 系列索引上存在；HNSW 会忽略它。这里统一在这里
        # 设一次，两种索引类型走同一份代码路径。
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = cfg["faiss"]["nprobe"]      # 32, 每次查询扫描的桶数；越大召回越高、延迟越高

        print("[wiki] loading chunk meta ...")
        self.meta: List[Dict] = []
        with open(paths["chunks_file"], "r", encoding="utf-8") as f:    #  "data/wiki_zh_chunks.jsonl"
            for line in f:
                # line: (chunk_id, doc_id, title, text)
                self.meta.append(json.loads(line))

        assert self.index.ntotal == len(self.meta), (   # 判断两边数据是否一致
            f"index({self.index.ntotal}) != chunks({len(self.meta)})"
        )
        print(f"[wiki] ready. total chunks = {len(self.meta)}")

    def warmup(self) -> None:
        """服务启动时主动预热：把 BGE-M3 权重加载到显存，避免首查询延迟毛刺。

        懒加载模式下，模型会在第一次 :meth:`search` 时才加载（3-8s）；
        生产环境建议在服务 ready 之前显式调用一次 warmup，使首查询延迟
        和稳态一致。多次调用无副作用（get_model 内部单例）。
        """
        print("[wiki] warming up embedder ...")
        get_model(
            model_name=self.cfg["embedder"]["model_name"],
            use_fp16=self.cfg["embedder"].get("use_fp16", True),
        )
        # 打一次真实 encode 触发 CUDA kernel 编译 / 图捕获
        _ = self.search("warmup", top_k=1)
        print("[wiki] warmup done.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """对 ``query`` 做编码并返回 Top-``k`` 命中。"""
        import time as _t
        _t0 = _t.perf_counter()
        q = encode(
            [query],
            model_name=self.cfg["embedder"]["model_name"],
            max_length=self.cfg["embedder"]["max_length"],
        )
        self._last_encode_ms = (_t.perf_counter() - _t0) * 1000

        _t0 = _t.perf_counter()
        scores, idxs = self.index.search(q, top_k)
        self._last_faiss_ms = (_t.perf_counter() - _t0) * 1000

        hits: List[Dict] = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:                          # FAISS 在结果不足时用 -1 补齐
                continue
            rec = self.meta[i]
            hits.append({
                "source":   "wiki",
                "title":    rec["title"],
                "text":     rec["text"],
                "chunk_id": rec["chunk_id"],
                "score":    float(score),
            })
        return hits
