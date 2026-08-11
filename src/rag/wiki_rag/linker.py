"""Linker：mention → qid 消歧（可选 embedding 重排）。

工作流
------
1. 用 :class:`KGStore` 拿到"精确/别名/FTS5"三级 fallback 的候选列表；
2. 若命中方案"只对热门 Top-30 万实体（能 join 上 L2 的 article_rank）算 embedding，其余 220 万实体只走 label 精查。" 的"热门实体索引"（11 步产物），
   用 BGE-M3 对 query 编码，与候选中"能查到向量"的那些实体做内积重排；
3. 未命中热门索引的候选（冷门实体）依旧走 weight 排序，直接拼在向量重排结果之后。

在线阶段的额外开销：
    - query 侧多一次 BGE-M3 编码（本来 wiki 检索时就已经在做，可复用）
    - 候选侧从 emb.npy mmap 读若干行，1 次 numpy 内积；
    - 若走 FAISS 索引则是一次 top-k 检索。

热门实体不存在时（比如没跑 11 步），Linker 自动退化成"纯 weight 排序"。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import load_config
from .embedder import encode
from .kg_store import KGStore


class Linker:
    """基于 KGStore + 可选热门实体向量库的实体链接器。"""

    def __init__(self,
                 kg_store: KGStore | None = None,
                 config_path: str | Path | None = None):
        cfg = load_config(config_path)
        self.cfg = cfg
        self.kg_cfg = cfg["kg"]
        self.emb_cfg = cfg["embedder"]
        self.kg = kg_store or KGStore(config_path)

        # ---- 加载热门实体向量（若存在）----
        # 我们不加载 FAISS 索引，而是用最小实现：mmap 读 emb.npy + numpy 内积。
        # 好处：这一小份向量最多几百万条，mmap 完全够用；且候选通常只有几十个，
        # 直接切片做 (K, dim) @ (dim,) 内积比 FAISS 快且无 nprobe/ef 参数烦恼。
        self.qid_to_row: Dict[str, int] = {}
        self.emb: Optional[np.ndarray] = None

        emb_file: Path = cfg["paths"]["kg_hot_emb_file"]        # "data/wikidata_zh_kg_hot_emb.npy"
        qids_file: Path = cfg["paths"]["kg_hot_qids_file"]      # "data/wiki_zh_kg_hot_qids.txt"
        if emb_file.exists() and qids_file.exists():
            print(f"[linker] loading hot entity embeddings: {emb_file}")
            # mmap 只读，不占常驻内存；实际用到的行才被 OS 页缓存进来
            self.emb = np.load(emb_file, mmap_mode="r")         # shape=(2871, 1024)
            with open(qids_file, "r", encoding="utf-8") as f:
                for i, ln in enumerate(f):
                    q = ln.strip()
                    if q:
                        self.qid_to_row[q] = i
            assert self.emb.shape[0] == len(self.qid_to_row), (
                f"emb rows({self.emb.shape[0]}) != qids({len(self.qid_to_row)})"
            )
            print(f"[linker] hot embeddings ready: {self.emb.shape}")
        else:
            print("[linker] hot embeddings NOT found; fallback to weight-only ranking. "
                  "(run scripts/11_encode_hot_entities.py to enable rerank)")

    # ------------------------------------------------------ core API
    def link(self, mention: str, top_k: Optional[int] = None,
             query_context: Optional[str] = None) -> List[Dict]:
        """把 mention 消歧成 top_k 个候选实体。

        Args:
            mention:       要链接的字面串（比如 "苹果"）
            top_k:         最终返回条数上限；默认 config.kg.link_final_k
            query_context: 可选的 query 全句（比如 "苹果公司的 CEO 是谁"）。
                           若提供，就用它做 embedding 重排——这样比只用 mention
                           更能区分 "苹果(水果)" 和 "苹果(公司)"。若不提供，
                           就用 mention 本身。

        Returns:
            List[{qid, label_zh, description, weight, source, score}]
            按 score 降序。
        """
        top_k = top_k or self.kg_cfg["link_final_k"]    # 5
        cands = self.kg.link(mention)   # 查询，输入mention → 候选实体列表
        if not cands:
            return []

        # KGStore.link() 已经算好了 `weight × log1p(popularity)` 的先验分。
        # 下面的重排必须**把它带上**，否则会把入度信号整个丢掉。
        # 归一化到 0~1：先验分是对数尺度（最大约 log(106万)≈14），
        # 而 cosine 在 0~1，不归一化直接加权会让先验分碾压语义相似度。
        _MAX_PRIOR = 14.0
        for c in cands:
            c["prior"] = min(float(c.get("score") or 0.0), _MAX_PRIOR) / _MAX_PRIOR

        # ---- 没有热门 embedding 时，退化成先验分排序 ----
        # ⚠️ 不能退化成纯 weight：weight 只有 1.0/0.8/0.6/0.5 四档，
        # 大量同名候选全是 1.0，排不出顺序来，等于把 KGStore 里刚算好的
        # 入度信息丢掉 —— “中国”会回到指向 1972 年的意大利电影。
        if self.emb is None or len(cands) <= 1:
            for c in cands:
                c["score"] = c["prior"]
            cands.sort(key=lambda x: x["score"], reverse=True)
            return cands[:top_k]

        # ---- 有热门 embedding 时，做 embedding 重排 ----
        query_text = query_context if query_context else mention
        q_vec = encode(
            [query_text],
            model_name=self.emb_cfg["model_name"],
            max_length=self.emb_cfg["max_length"],
            device=self.emb_cfg["device"],
            normalize=True,
        )[0]  # (dim,)

        # 拆两拨：命中热门向量库的 vs 没命中的
        hot_idx: list[int] = []
        hot_pos: list[int] = []       # 在 cands 列表里的原始位置
        cold_pos: list[int] = []
        for i, c in enumerate(cands):
            row = self.qid_to_row.get(c["qid"])
            if row is not None:
                hot_idx.append(row)
                hot_pos.append(i)
            else:
                cold_pos.append(i)

        # 热门候选：一次性 gather + 内积（内积等价 cosine，因两侧都已归一化）
        if hot_idx:
            # np.take + reshape 是最省事的 gather
            mat = self.emb[hot_idx]                       # (K, dim)
            sims = mat @ q_vec.astype("float32")          # (K,)
            for pos, sim in zip(hot_pos, sims):
                # 融合分：0.6 * cosine + 0.4 * 先验分
                # 先验分 = weight × log1p(入度) 归一化，既包含 label/alias
                # 的证据强度，也包含实体重要度。
                # ⚠️ 原实现是 `0.7*cosine + 0.3*weight`，直接用裸 weight，
                # 把 KGStore 里算好的入度信号**整个覆盖掉了** —— 这是
                # 为什么修了 kg_store 后，link("中国") 已经能排对，
                # 但走完整管线的 retrieve() 仍然返回「1972 年的意大利电影」。
                cands[pos]["score"] = (0.6 * float(sim)
                                       + 0.4 * cands[pos]["prior"])
        # 冷门候选：没有 embedding，只能用先验分，乘一个系数避免和热门尺度差异过大
        for p in cold_pos:
            cands[p]["score"] = 0.5 * cands[p]["prior"]

        cands.sort(key=lambda x: x["score"], reverse=True)
        return cands[:top_k]

    # ------------------------------------------------------ 便捷 API
    def link_and_expand(self, mention: str,
                        query_context: Optional[str] = None,
                        top_k: int = 1,
                        triples_per_entity: Optional[int] = None) -> List[Dict]:
        """一步到位：链接 + 拉出候选的三元组，返回每个候选一段可读文本。

        典型下游用法：把返回的 ``context`` 字段拼接到 LLM system prompt 里。
        """
        cands = self.link(mention, top_k=top_k, query_context=query_context)
        out = []
        for c in cands:
            ctx = self.kg.to_context(c["qid"])
            out.append({
                **c,
                "context": ctx,
            })
        return out
