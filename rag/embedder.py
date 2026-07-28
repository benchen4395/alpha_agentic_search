# rag/embedder.py
"""统一的 embedding 接入层（全库统一到 FlagEmbedding BGE-M3）。

背景与决策（D1）
----------------
生产环境要求「一套编码方案贯穿 L1-L5」，因此本模块底层统一使用
:class:`FlagEmbedding.BGEM3FlagModel`（原 wiki_rag 的编码路径）：

* L2 wiki FAISS 索引、L5 KG 热门实体向量都是用 BGE-M3 离线构建的；
  查询侧走同一个模型，才能保证内积 = 精确 cosine，召回不掉点。
* L1（qa_cache 模糊命中）、L3（历史归档）也复用同一向量空间，
  跨层 RRF 融合、相似度阈值判定才有一致语义。

对外接口
--------
暴露 :class:`Embedder`，方法 ``embed`` / ``embed_batch``：返回「已 L2 归一化的
Python list[float]」。批量路径 ``embed_batch`` 直接走 BGE-M3 的原生 batch
编码，比逐条快一个数量级。设备选择委托 wiki_rag.embedder 的 ``_resolve_device``
（CUDA → MPS → CPU）。
"""
from __future__ import annotations

from typing import Iterable, Optional

from . import config as rag_config
from .wiki_rag import embedder as _bge


class Embedder:
    """统一 embedder 包装（底层 = FlagEmbedding BGE-M3）。

    用法：
        emb = Embedder()
        v = emb.embed("你好")            # -> list[float]，已 L2 归一化
        vs = emb.embed_batch(["a","b"])  # -> list[Optional[list[float]]]
    """

    def __init__(self, model: Optional[str] = None, device: str = "auto"):
        # 模型名：默认取 config.RAG_EMBED_MODEL（已统一为 BAAI/bge-m3）
        self.model = model or rag_config.RAG_EMBED_MODEL
        self.device = device

    # ---------- 单条 ---------- #
    def embed(self, text: str) -> Optional[list[float]]:
        """单条编码，返回已 L2 归一化的 list[float]；空串返回 None。"""
        if not text:
            return None
        # BGE-M3 原生按 batch 编码，这里包一个单元素 batch
        vecs = _bge.encode(
            [text],
            model_name=self.model,
            device=self.device,
            normalize=True,
        )
        return vecs[0].tolist()

    # ---------- 批量 ---------- #
    def embed_batch(self, texts: Iterable[str]) -> list[Optional[list[float]]]:
        """批量编码。走 BGE-M3 原生 batch，比逐条快很多。

        空串会被占位为 None，保持返回列表与输入一一对应。
        """
        text_list = list(texts)
        if not text_list:
            return []

        # 记录空串位置，非空串统一送入模型
        idx_nonempty = [i for i, t in enumerate(text_list) if t]
        result: list[Optional[list[float]]] = [None] * len(text_list)
        if not idx_nonempty:
            return result

        vecs = _bge.encode(
            [text_list[i] for i in idx_nonempty],
            model_name=self.model,
            device=self.device,
            normalize=True,
        )
        for slot, vec in zip(idx_nonempty, vecs):
            result[slot] = vec.tolist()
        return result
