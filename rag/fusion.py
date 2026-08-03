# rag/fusion.py
"""结果融合器：RRF (Reciprocal Rank Fusion) + 可选 rerank。

RRF 公式：
    score(d) = Σ_l  1 / (k + rank_l(d))
其中 k 常取 60，对分数量级不敏感，非常适合融合"来源打分口径不一致"的层。
"""
from __future__ import annotations

from typing import Optional

from . import config as rag_config
from .types import Passage


def _passage_key(p: Passage) -> str:
    """去重键：优先 url，其次 (title, text[:80])。"""
    if p.url:
        return f"url::{p.url}"
    return f"txt::{p.title[:40]}|{p.text[:80]}"


def rrf_fuse(
    per_layer_results: list[list[Passage]],
    top_k: int = 6,
    k: int = 60,
) -> list[Passage]:
    """对多路结果做 RRF 融合。

    Args:
        per_layer_results: 每一路（每一层）已按各自分数排序的 Passage 列表
        top_k: 融合后保留的段落数
        k    : RRF 常数
    Returns:
        融合后按分数降序的 Passage 列表（保留最高分那一份 metadata）
    """
    fused: dict[str, tuple[float, Passage]] = {}
    for layer_result in per_layer_results:
        for rank, p in enumerate(layer_result, start=1):
            key = _passage_key(p)
            contrib = 1.0 / (k + rank)
            prev = fused.get(key)
            if prev is None:
                # 用一个新 Passage 承载融合分（保留原 layer 元数据）
                new_p = Passage(
                    text=p.text, title=p.title, url=p.url,
                    score=contrib, layer=p.layer,
                    metadata={**p.metadata, "layers": [p.layer]},
                )
                fused[key] = (contrib, new_p)
            else:
                prev_score, prev_p = prev
                prev_p.score = prev_score + contrib
                layers = prev_p.metadata.setdefault("layers", [])
                if p.layer not in layers:
                    layers.append(p.layer)
                # P0-2：同一文档被多层命中时，保留**最高**的校准置信度。
                # 否则会取决于 as_completed 的到达顺序（不确定），
                # 可能把 L4 top1 的 0.77 覆盖成 L3 的 0.12 —— 同一次查询
                # 跑两遍得到不同的置信度，这在可观测性上是不可接受的。
                cal_new = p.metadata.get("calibrated")
                if cal_new is not None:
                    cal_old = prev_p.metadata.get("calibrated")
                    if cal_old is None or float(cal_new) > float(cal_old):
                        prev_p.metadata["calibrated"] = cal_new
                fused[key] = (prev_p.score, prev_p)
    ordered = [pp for _, pp in sorted(fused.values(), key=lambda x: -x[0])]
    return ordered[:top_k]


# ---------- 二阶 rerank（D5：默认 RRF 零依赖，BGE/cascade 走环境变量） ----------
# 策略由 config.RERANK_STRATEGY 决定：
#   "rrf"     : 已在 rrf_fuse 阶段完成融合排序，这里直接返回（零依赖、零延迟）——默认
#   "bge"     : 用 FlagEmbedding 的 BGE cross-encoder 精排
#   "cascade" : 先 RRF 粗排 top-N 再 BGE 精排
#   "none"    : 不做二阶 rerank
# BGE/cascade 复用 vendored 的 wiki_rag.hybrid 里的构造器，避免重复实现。
_reranker_fn = None          # 缓存构造好的 rerank_fn（(query, docs)->docs）
_reranker_tried = False


def _try_load_reranker():
    """按 config.RERANK_STRATEGY 懒加载二阶 rerank 函数（进程内构造一次）。

    仅 bge/cascade 需要真正加载模型；rrf/none 返回 None（走轻量路径）。
    """
    global _reranker_fn, _reranker_tried
    if _reranker_tried:
        return _reranker_fn
    _reranker_tried = True

    strategy = rag_config.RERANK_STRATEGY
    if strategy in ("rrf", "none", "off", ""):
        # RRF 已在 rrf_fuse 完成；这里无需额外模型
        _reranker_fn = None
        return None

    from .wiki_rag.hybrid import build_reranker
    if strategy == "cascade":
        _reranker_fn = build_reranker("cascade", bge_model_name=rag_config.RERANK_MODEL)
    else:  # "bge" / "cross" / "cross-encoder"
        _reranker_fn = build_reranker("bge", model_name=rag_config.RERANK_MODEL)
    print(f"[fusion] 二阶 rerank 已启用：strategy={strategy}")
    return _reranker_fn


def rerank(query: str, passages: list[Passage], top_k: Optional[int] = None) -> list[Passage]:
    """对候选做二阶 rerank。

    - 默认策略 rrf/none：直接截断返回（rrf_fuse 已排好序，零延迟）。
    - bge/cascade：把 Passage 转成 wiki_rag rerank_fn 需要的 dict，精排后写回分数。
    """
    if not passages:
        return passages
    rerank_fn = _try_load_reranker()
    if rerank_fn is None:
        # RRF-only 路径：rrf_fuse 已按融合分排序，直接截断
        return passages[: top_k or len(passages)]

    # BGE/cascade 路径：适配 dict 接口
    docs = [{"text": p.text[:512], "_p": p, "score": p.score} for p in passages]
    reordered = rerank_fn(query, docs)
    out: list[Passage] = []
    for d in reordered:
        p = d["_p"]
        # 把 cross-encoder 分数写回 Passage.score（便于上层观测）
        if "rerank_score" in d:
            p.score = float(d["rerank_score"])
        out.append(p)
    return out[: top_k or len(out)]
