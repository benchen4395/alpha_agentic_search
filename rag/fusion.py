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


def quota_fuse(
    groups: list[list[Passage]],
    entities: list[str],
    top_k: int = 6,
    k: int = 60,
    min_per_entity: int = 2,
) -> list[Passage]:
    """带**实体配额**的融合：先 RRF 排序，再保证每个并列实体都有保底席位。

    ════════════════════════════════════════════════════════════════════
    要解决的问题：winner-takes-most
    ════════════════════════════════════════════════════════════════════
    实测故障：
        提问「国庆期间，俄罗斯、希腊、巴厘岛的气候和景色分别如何？」
        融合后 6 段的实体分布: {'俄罗斯': 4, '希腊': 1, '巴厘岛': 1}
        模型回答「希腊完全没有相关资料」

    证据**其实检索到了**，是在这个函数所在的融合阶段被挤掉的。
    根因是 RRF 只有"全局相关度"一个维度，它不知道这条 query 里有 3 个
    **并列**对象、每个都必须有料才能回答。哪个实体的资料更丰富/更权威，
    它就独占 FUSION_TOP_K 这个固定预算。

    注意这不是"检索次数不够"。我实测过：只把 query 拆成 3 个子 query
    并发搜、不改融合，饥饿只是**换了个实体**：
        单次长 query :  俄罗斯 1 / 希腊 0 / 巴厘岛 3
        拆 3 个子query:  俄罗斯 0 / 希腊 1 / 巴厘岛 3
    6 个席位分给 3 个对象，没有配额约束照样有人归零。所以配额是**前置**修复。

    ════════════════════════════════════════════════════════════════════
    参数为什么这样设计
    ════════════════════════════════════════════════════════════════════
    `groups` 刻意**不叫 per_layer**、也刻意不是 dict[LayerName, ...]。

    它的语义就是"多路已排序的结果"，至于这些路是按**层**分的（今天）
    还是按**子 query** 分的（第二步做多 query 并发时），这个函数
    一行都不用改。我实测过同一个 RRF 实现喂两种分组方式都正常工作。
    如果签名写成 `per_layer: dict[str, list[Passage]]`，第二步就得
    重构签名和所有调用点 —— 那才是"先做第一步反而让第二步更麻烦"。

    Args:
        groups:         多路已按各自分数排序的 Passage 列表
        entities:       并列实体列表（由 rag.entities.extract_parallel_entities 给出）。
                        **长度 < 2 时本函数完全退化为 rrf_fuse**，逐段等价。
        top_k:          最终保留段数（硬上限，配额不会突破它）
        k:              RRF 常数
        min_per_entity: 每个实体的保底席位数。

                        为什么默认 2 而不是 1：1 段往往只够证明"这个实体
                        存在"，不够回答"它的气候和景色如何"这种双维度问题。
                        2 段是实测下来"够说一句有内容的话"的下限。
                        当 top_k // len(entities) < min_per_entity 时会
                        自动下调（见下），所以实体很多时不会把预算撑爆。

    Returns:
        融合后的 Passage 列表，长度 ≤ top_k，无重复段落。
    """
    # ══════════════════════════════════════════════════════════════════
    # 第 0 步：先跑原样的 RRF。
    #
    # ⚠️ 关键设计：配额是在 RRF **结果之上**做的一次重排，而不是另写一套
    # 打分逻辑。这带来两个性质：
    #   ① 单实体/无实体时可以直接把 RRF 结果原样返回 → 天然逐段等价，
    #      不需要"小心地保持一致"，而是**根本走同一段代码**；
    #   ② 实体内部的相对顺序仍由 RRF 决定 —— 配额只管"分几个席位给谁"，
    #      不管"哪一段更好"。职责单一，出问题好定位。
    #
    # 注意这里传的 top_k 不是最终的 top_k，而是一个放大的候选池。
    # 原因：要给弱实体腾位置，就必须能看到 RRF 前 6 名**之外**的段落
    # —— 希腊的那段本来就排在第 7 名以后，只取 6 个的话它根本不在候选里。
    # ══════════════════════════════════════════════════════════════════
    valid = [e for e in (entities or []) if e]
    if len(valid) < 2:
        # 单实体 / 无实体 —— 也就是线上 99% 的流量。
        # 直接走原函数，零行为差异、零额外开销。
        return rrf_fuse(groups, top_k=top_k, k=k)

    # 候选池大小：每个实体理论上最多需要 min_per_entity 段，再留一倍余量
    # 给"归属判断没命中任何实体"的通用段落。取 top_k 的上界保证不会
    # 因为池子太小而找不到弱实体的证据。
    pool_size = max(top_k, len(valid) * min_per_entity * 2, top_k * 3)
    pool = rrf_fuse(groups, top_k=pool_size, k=k)
    if not pool:
        return []

    from .entities import attribute_to_entity

    # ══════════════════════════════════════════════════════════════════
    # 第 1 步：算每个实体的配额。
    #
    # 实体多、预算少时（比如 5 个实体抢 6 个席位），min_per_entity=2
    # 是不可能满足的。此时下调到 top_k // n，且至少给 1 —— 宁可每个实体
    # 只有 1 段，也不要有实体是 0 段：0 段会让模型说"完全没有资料"，
    # 1 段至少能说点什么。这个"每个都有 ≥1"是本次修复的核心目标。
    # ══════════════════════════════════════════════════════════════════
    quota = max(1, min(min_per_entity, top_k // len(valid)))

    # ══════════════════════════════════════════════════════════════════
    # 第 2 步：按 RRF 顺序扫候选池，给每个实体填满配额。
    #
    # 为什么按 RRF 顺序扫而不是按实体分组后各取 top-n：
    #   保证被选中的段落在**该实体内部**也是最优的那几段
    #   （RRF 顺序天然就是相关度降序），不需要二次排序。
    #
    # `taken` 按段落去重：一段"俄罗斯和希腊都在北半球"会同时满足两个
    # 实体的配额需求，但**只能进结果一次**。这是这段逻辑最容易写错的
    # 地方 —— 按实体分别 append 就会出现重复段落，
    # 下游 evidence block 里会出现两个内容一样的 <doc>。
    # ══════════════════════════════════════════════════════════════════
    filled: dict[str, int] = {e: 0 for e in valid}
    taken: dict[str, Passage] = {}      # _passage_key -> Passage，保序靠 dict
    for p in pool:
        owners = attribute_to_entity(p.text + " " + p.title, valid)
        if not owners:
            continue
        # 只有当这段能帮到**至少一个还没吃饱的实体**时才收
        hungry = [e for e in owners if filled[e] < quota]
        if not hungry:
            continue
        key = _passage_key(p)
        if key in taken:
            continue
        taken[key] = p
        # 一段同时讲多个实体时，给**所有**它讲到的实体记账（不只 hungry）。
        # 否则会出现这种浪费：这段已经讲了希腊，但因为没记账，
        # 后面又专门为希腊留一个席位，等于同一份信息占了两个位置。
        for e in owners:
            filled[e] += 1
        if len(taken) >= top_k:
            break

    # ══════════════════════════════════════════════════════════════════
    # 第 3 步：剩余席位按 RRF 原顺序回填。
    #
    # 配额只保证**下限**，不是把预算平均分完就结束。剩下的位置仍然交给
    # 全局相关度 —— 这样强实体依然能拿到较多份额（它本来就资料多、
    # 用户也可能更关心），只是不能把别人饿死。
    #
    # 这一步也顺便兜住了两种情况：
    #   * 某实体一条证据都没检索到（"冰岛"）→ 它的配额自然落空，
    #     席位在这里被其它段落用掉，不会白白浪费预算；
    #   * 池子里有不归属任何实体的通用段落（"十月的欧洲旅游总览"）→
    #     它们在第 2 步被跳过，在这里按相关度正常参与竞争。
    # ══════════════════════════════════════════════════════════════════
    if len(taken) < top_k:
        for p in pool:
            if len(taken) >= top_k:
                break
            key = _passage_key(p)
            if key not in taken:
                taken[key] = p

    # ══════════════════════════════════════════════════════════════════
    # 第 4 步：恢复成 RRF 分数降序。
    #
    # 第 2/3 步的插入顺序是"配额优先"的，直接返回会让弱实体的低分段落
    # 排在最前面。而下游 `evidence.build_evidence_block` 是按顺序编号
    # <doc id=1..n> 的，模型对靠前的证据权重更高 —— 顺序有语义。
    # 所以最后按分数重排：配额决定**谁进来**，分数决定**排第几**。
    # ══════════════════════════════════════════════════════════════════
    out = sorted(taken.values(), key=lambda p: -p.score)

    dropped = [e for e, n in filled.items() if n == 0]
    if dropped:
        # 这行日志很重要：它说明"配额已生效但该实体确实一条证据都没有"，
        # 是真正需要走第二步（拆 query 并发检索）的信号。
        # 只有这条日志频繁出现，才值得再加多 query 路由的复杂度。
        print(f"[fusion] ⚠️ 并列实体 {dropped} 无任何证据（配额无法兑现）")
    else:
        print(f"[fusion] 📊 实体配额生效 quota={quota}/实体，分布={filled}")
    return out[:top_k]


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
