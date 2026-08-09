# rag/calibration.py
"""跨层分数校准。

════════════════════════════════════════════════════════════════════════
为什么需要校准：直接比较各层原始分在统计上没有意义
════════════════════════════════════════════════════════════════════════
一个直觉的兜底判定写法是：

    offline_best = max(所有非 L4 层的 p.score)
    if offline_best < 0.55:  →  追加 L4 web

这里有**两个独立的 bug**，叠加导致 L4 几乎永远不会被触发：

┏━ Bug ①：`or 0.9` 把 KG 分数全部抬到 0.9 ━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ `rag/layers.py` L5KGLayer.search() 里写的是：                     ┃
┃     score=float(d.get("score") or 0.9)                           ┃
┃ 而 `traverse_multi_hop()` 给多跳发现的实体写死了 `"score": 0.0`。  ┃
┃ Python 里 `0.0 or 0.9 == 0.9` —— 所以**所有多跳实体的分数都变成   ┃
┃ 0.9**。只要 KG 抽到任何一个 mention，offline_best 就 = 0.9 > 0.55，┃
┃ L4 永不启动。用户问时事，系统却只拿 Wikidata 的陈旧三元组作答。     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┏━ Bug ②：拿四个不同量纲的原始分做标量比较 ━━━━━━━━━━━━━━━━━━━━━━━┓
┃   L2 wiki    : BGE-M3 余弦，相关命中典型落在 0.45 ~ 0.75          ┃
┃   L3 history : BGE-M3 余弦，同上但分布更集中（都是自己写的答案）    ┃
┃   L5 kg      : 0.7*cosine + 0.3*weight 的人工混合分（linker.py）  ┃
┃   L4 web     : `1.0 - i*0.05` 的位次线性衰减（完全不是相似度！）   ┃
┃ 用一个 0.55 的阈值同时裁决这四种分数，等于"用摄氏度阈值判断华氏温度"。┃
┃ 后果：L2 的 0.6（其实很相关）被判"不够"，L5 的 0.9（可能只是热门   ┃
┃ 实体先验高）被判"很够"。                                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

════════════════════════════════════════════════════════════════════════
解法：把各层原始分映射到统一的「相关概率」空间 P(relevant)
════════════════════════════════════════════════════════════════════════
                     各层原始分（量纲互不相同）
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        L2 cosine 0.62   L5 mixed 0.90   L4 rank 0.95
              │               │               │
              └────► calibrate(layer, score) ◄┘   ← 本模块
                              │
                              ▼
                      P(relevant) ∈ [0,1]
                     （跨层可比、可做阈值、可做 abstention）

具体做法：**分层 Logistic（Platt scaling）**

    p = sigmoid( a_layer * (s - b_layer) )

* `b_layer` 是该层的"决策中点"：原始分等于 b 时 p=0.5，
  即"该层给出这个分数时，一半概率相关"。
* `a_layer` 是斜率（陡峭度）：越大表示该层分数越有判别力
  （分数稍微高一点就明显更可信）。

为什么用 Platt scaling 而不是别的：
  1. **两参数、单调、可解释** —— 不会打乱层内排序（这点很重要：
     校准只改变"跨层可比性"，不应改变层内的相对优劣）。
  2. **可从少量标注拟合** —— 500 条标注就够（见 `fit_platt()`），
     不需要大规模训练数据。
  3. **业界标准** —— Platt(1999) / Guo et al.(2017) 的标准做法。

默认参数是**基于各层分数分布的先验估计**（见下方每层注释里的推导），
不是拍脑袋：它们让「同一个 p 值」在各层含义一致。上线后应该用真实
标注数据跑 `fit_platt()` 覆盖（本模块支持环境变量/JSON 文件热覆盖）。

════════════════════════════════════════════════════════════════════════
使用方式
════════════════════════════════════════════════════════════════════════
    from rag.calibration import calibrate, calibrate_passages, aggregate_confidence

    p = calibrate("L2_wiki", 0.62)              # -> 0.71
    calibrate_passages(passages)                # 就地写入 p.metadata["calibrated"]
    conf = aggregate_confidence(per_layer)      # -> 0.0~1.0 整体证据强度
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, Optional


# ════════════════════════════════════════════════════════════════════════
#                        校准参数（每层一组）
# ════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PlattParams:
    """单层的 Platt scaling 参数。

    p = sigmoid(a * (s - b))

    Fields:
        a: 斜率。越大 → sigmoid 越陡 → 该层分数判别力越强。
        b: 决策中点（原始分尺度）。s == b 时 p == 0.5。
        note: 该组参数的推导依据，写清楚便于后人调参时理解取值来源。
    """

    a: float
    b: float
    note: str = ""


# —— 默认参数：基于各层分数分布的先验估计 ——
# ⚠️ 这些是**冷启动默认值**，用于让系统立刻可用且行为可解释。
#    生产环境请用 fit_platt() 在自建标注集上重新拟合后覆盖。
_DEFAULT_PARAMS: dict[str, PlattParams] = {
    # ---- L1 QA Cache ----
    # 能命中 L1 意味着已经过了「精确匹配」或「0.90 阈值 + 槽位门禁」，
    # 是全系统置信度最高的一路，直接给接近 1 的概率。
    "L1_qa": PlattParams(
        a=20.0, b=0.5,
        note="命中即高可信（精确匹配或过了 0.90 阈值+槽位门禁）",
    ),

    # ---- L2 Wikipedia（BGE-M3 余弦）----
    # 分布依据：BGE-M3 在中文短 query vs wiki chunk 上，
    #   不相关   ≈ 0.30 ~ 0.45
    #   弱相关   ≈ 0.45 ~ 0.55
    #   相关     ≈ 0.55 ~ 0.70
    #   强相关   ≈ 0.70+
    # 因此把中点 b 设在 0.55（弱/强相关的分界），
    # a=12 让 0.45→0.23、0.55→0.50、0.65→0.77，区分度合适。
    "L2_wiki": PlattParams(
        a=12.0, b=0.55,
        note="BGE-M3 余弦；0.55 为相关/不相关经验分界",
    ),

    # ---- L3 History（BGE-M3 余弦，但语义不同）----
    # 关键差异：L3 存的是「本系统过去生成的答案」，即使语义相似，
    # 也**不能等同于外部权威证据**（可能是过去的幻觉、可能已过时）。
    # 所以在同样的余弦分数下，L3 的概率应该系统性低于 L2：
    # 中点抬到 0.62（要求更相似才认可），斜率略缓（a=10）。
    "L3_history": PlattParams(
        a=10.0, b=0.62,
        note="同为 BGE-M3 余弦，但历史答案非权威证据，中点上调以降权",
    ),

    # ---- L4 Web（位次衰减，不是相似度）----
    # 原始分是 `1.0 - rank*0.05`，即 rank1=1.0, rank2=0.95, ...
    # 这只是位次的线性映射，携带的信息量远低于余弦。
    # 校准目标：让 top1 ≈ 0.70（搜索引擎首条通常靠谱但不保证），
    # 逐位缓慢衰减。中点 b=0.85（约等于 rank4），斜率 a=8。
    #   rank1 (1.00) → 0.77   rank3 (0.90) → 0.60
    #   rank5 (0.80) → 0.39   rank8 (0.65) → 0.17
    "L4_web": PlattParams(
        a=8.0, b=0.85,
        note="位次衰减而非相似度；top1≈0.77，逐位缓慢下降",
    ),

    # ---- L5 KG（linker 混合分 0.7*cos + 0.3*weight）----
    # 双重性质：
    #   - KG 三元组是**结构化事实**，一旦实体链接正确就非常可靠；
    #   - 但实体链接本身可能错（"苹果"→水果 vs 公司），
    #     且 weight 只是热度先验，热门实体分数天然偏高。
    # 因此中点设在 0.5、斜率给较大的 14：链接置信度高时快速上升到
    # 高概率，链接置信度低时快速掉到低概率——符合"要么很对要么很错"。
    # ⚠️ 另见下方 `KG_MULTI_HOP_DECAY`：多跳发现的实体必须额外降权。
    "L5_kg": PlattParams(
        a=14.0, b=0.50,
        note="linker 混合分；链接对则事实可靠，错则完全无关，故斜率大",
    ),
}

# 未知层的兜底参数（保守：中点 0.6、斜率平缓）
_FALLBACK_PARAMS = PlattParams(a=8.0, b=0.60, note="未知层兜底")


# ════════════════════════════════════════════════════════════════════════
#                          KG 多跳衰减
# ════════════════════════════════════════════════════════════════════════
# `traverse_multi_hop()` 发现的实体是"从种子实体沿关系边走出来的"，
# 每走一跳，与原始 query 的相关性都显著下降（经典的 KG reasoning 衰减）。
# 做法：hop-N 的实体分数 = 种子分数 * DECAY^(N-1)。
#   hop1（种子）    : ×1.00
#   hop2（1 跳邻居）: ×0.60
#   hop3（2 跳邻居）: ×0.36
KG_MULTI_HOP_DECAY: float = float(os.getenv("RAG_KG_HOP_DECAY", "0.6"))

# 当 linker 完全没给出分数时（例如未构建热门实体向量、退化成 weight-only
# 排序且 weight 也缺失），用这个保守的基准分。
# 关键：**必须显著低于 WEB_FALLBACK 对应的概率**，否则又会变成
# "KG 一命中就阻止 L4 兜底"。0.35 经校准后 ≈ 0.10，足够保守。
KG_UNSCORED_BASELINE: float = float(os.getenv("RAG_KG_UNSCORED_BASE", "0.35"))


# ════════════════════════════════════════════════════════════════════════
#                      参数加载（支持外部覆盖）
# ════════════════════════════════════════════════════════════════════════
def _load_overrides() -> dict[str, PlattParams]:
    """从环境变量指定的 JSON 文件加载校准参数覆盖。

    生产流程：
        1. 在自建标注集上跑 `fit_platt()` 得到每层的 (a, b)
        2. 存成 JSON：{"L2_wiki": {"a": 11.3, "b": 0.58}, ...}
        3. `export RAG_CALIBRATION_FILE=/path/to/calib.json`
        4. 重启服务即生效，**无需改代码**

    文件缺失 / 解析失败时静默使用默认参数（不影响启动）。
    """
    path = os.getenv("RAG_CALIBRATION_FILE", "")
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        out: dict[str, PlattParams] = {}
        for layer, cfg in (raw or {}).items():
            out[layer] = PlattParams(
                a=float(cfg["a"]), b=float(cfg["b"]),
                note=str(cfg.get("note", "loaded from " + os.path.basename(path))),
            )
        if out:
            print(f"[calibration] 已加载外部校准参数: {sorted(out)} ← {path}")
        return out
    except Exception as e:
        print(f"[calibration] 外部校准参数加载失败（回退默认）: {e}")
        return {}


_PARAMS: dict[str, PlattParams] = {**_DEFAULT_PARAMS, **_load_overrides()}


def get_params(layer: str) -> PlattParams:
    """取某层的校准参数（未注册的层返回保守兜底参数）。"""
    return _PARAMS.get(layer, _FALLBACK_PARAMS)


def set_params(layer: str, a: float, b: float, note: str = "") -> None:
    """运行期覆盖某层参数（单测 / 在线调参用）。"""
    _PARAMS[layer] = PlattParams(a=float(a), b=float(b), note=note or "runtime override")


# ════════════════════════════════════════════════════════════════════════
#                              核心：校准
# ════════════════════════════════════════════════════════════════════════
def _sigmoid(x: float) -> float:
    """数值稳定的 sigmoid（避免 exp 溢出）。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    e = math.exp(max(x, -60.0))
    return e / (1.0 + e)


def calibrate(layer: str, raw_score: float, *, hop: int = 1) -> float:
    """把某层的原始分映射为统一的相关概率 P(relevant) ∈ [0, 1]。

    Args:
        layer:     层名（"L1_qa" / "L2_wiki" / "L3_history" / "L4_web" / "L5_kg"）。
        raw_score: 该层给出的原始分数（各层量纲不同，本函数负责统一）。
        hop:       仅对 L5_kg 有意义——该实体是第几跳发现的。
                   hop=1 为种子实体（linker 直接链接到的），
                   hop>=2 为 `traverse_multi_hop` BFS 扩展出来的，
                   按 `KG_MULTI_HOP_DECAY^(hop-1)` 逐跳衰减。

    Returns:
        校准后的概率，跨层可比。

    示例：
        >>> round(calibrate("L2_wiki", 0.55), 3)      # 恰好在中点
        0.5
        >>> round(calibrate("L4_web", 1.0), 3)        # web 首条
        0.769
        >>> calibrate("L5_kg", 0.9, hop=3) < calibrate("L5_kg", 0.9, hop=1)
        True
    """
    s = float(raw_score or 0.0)

    # KG 多跳衰减：在进 sigmoid 之前先按跳数打折。
    # 放在校准之前而非之后，是为了让衰减作用在"原始分尺度"上，
    # 这样 sigmoid 的非线性能正确放大"跳数越多越不可信"的效应。
    if layer == "L5_kg" and hop > 1:
        s *= KG_MULTI_HOP_DECAY ** (hop - 1)

    p = get_params(layer)
    return _sigmoid(p.a * (s - p.b))


def calibrate_passages(passages: Iterable, *, overwrite: bool = True) -> list:
    """就地为一批 Passage 计算校准概率，写入 `metadata["calibrated"]`。

    为什么写 metadata 而不是覆盖 `Passage.score`：
      - `score` 是**层内排序**用的原始分，覆盖它会破坏层内顺序语义，
        也会让 debug 时看不到原始值；
      - RRF 融合用的是 rank 而非 score，不受影响；
      - 上层（retriever / agent / 前端）需要"可解释的置信度"时读
        `metadata["calibrated"]` 即可，两者并存、各司其职。

    Args:
        passages: 可迭代的 Passage 列表（需有 layer / score / metadata 属性）。
        overwrite: 是否覆盖已存在的 `calibrated` 值。

            ⚠️ **在 RRF 融合之后必须传 False**，原因是一个很容易踩的坑：

            `rrf_fuse()` 会新建 Passage，并把 `score` 换成 RRF 贡献值
            `Σ 1/(k+rank)`（k=60，所以典型值只有 0.016~0.03），
            它是**排序用的秩倒数，不是相似度**。若此时用 overwrite=True
            重新校准，就会拿 0.016 去过 sigmoid，所有 conf 都被压成 0.00
            —— 明明检索得很好，来源面板却全显示"置信度 0.00"。

            正确姿势：融合前在各层内校准（那时 score 还是层的原始分），
            融合后只用 overwrite=False 给漏掉的补算。
    Returns:
        原列表（便于链式调用）。
    """
    out = []
    for p in passages:
        try:
            meta = getattr(p, "metadata", None) or {}
            # 已有校准值且不允许覆盖 → 直接跳过（保住融合前算好的正确值）
            if not overwrite and meta.get("calibrated") is not None:
                out.append(p)
                continue
            # KG 的 hop 信息藏在 metadata["via"]（多跳时才有）与
            # metadata["source"]（形如 "hop2"）里，这里做兼容解析。
            hop = 1
            src = str(meta.get("source") or "")
            if src.startswith("hop"):
                try:
                    hop = int(src[3:]) or 1
                except ValueError:
                    hop = 2      # 解析不出具体跳数时，保守当 2 跳处理
            elif meta.get("via"):
                hop = 2          # 有 via 字段说明是从别的实体跳来的
            meta["calibrated"] = round(
                calibrate(getattr(p, "layer", ""), getattr(p, "score", 0.0), hop=hop),
                4,
            )
            if hasattr(p, "metadata"):
                p.metadata = meta
        except Exception as e:      # 校准失败不应影响检索主流程
            print(f"[calibration] passage 校准异常（跳过）: {e}")
        out.append(p)
    return out


# ════════════════════════════════════════════════════════════════════════
#                     聚合置信度（供 abstention 使用）
# ════════════════════════════════════════════════════════════════════════
# 单条证据的概率不足以刻画"整体证据是否充分"。两个额外因素：
#   1. **多层一致性**：同一事实被 L2 和 L4 同时命中，可信度应高于单层。
#   2. **收益递减**：第 5 条相关证据的边际价值远低于第 1 条。
#
# 所以用「噪声-OR」聚合（也叫 soft-OR / probabilistic OR）：
#       P(至少一条相关) = 1 - Π(1 - p_i)
# 它天然满足上述两点：多路互相增强、且单调有上界。
#
# 但纯噪声-OR 有个问题：很多条低质证据也能把概率堆到很高
# （10 条 p=0.2 的证据 → 1-0.8^10 = 0.89）。
# 所以只取 **top-N 条**参与聚合（默认 3），避免"用数量伪造置信度"。
AGG_TOP_N: int = int(os.getenv("RAG_AGG_TOP_N", "3"))


def aggregate_confidence(
    per_layer_passages: dict,
    *,
    top_n: Optional[int] = None,
) -> float:
    """把多层召回结果聚合成一个整体证据置信度 ∈ [0, 1]。

    Args:
        per_layer_passages: `{layer_name: [Passage, ...]}`，
                            即 `LayeredRetriever._parallel_search()` 的返回。
        top_n: 参与聚合的最高分证据条数（默认 `AGG_TOP_N`=3）。
               只取 top-N 是为了防止"大量低质证据堆出高置信度"。

    Returns:
        聚合置信度。语义："至少有一条真正相关的证据"的概率。

    用法（在 retriever 里）：
        conf = aggregate_confidence(per_layer)
        if conf < WEB_FALLBACK_CONFIDENCE:   →  补 L4
        if conf < ABSTAIN_CONFIDENCE:        →  提示"资料不足"（用）
    """
    n = top_n if top_n is not None else AGG_TOP_N

    # 收集所有 passage 的校准概率
    probs: list[float] = []
    for layer, plist in (per_layer_passages or {}).items():
        for p in plist or []:
            meta = getattr(p, "metadata", None) or {}
            cal = meta.get("calibrated")
            if cal is None:
                # 尚未校准（例如调用方跳过了 calibrate_passages）→ 现场算
                hop = 2 if meta.get("via") else 1
                cal = calibrate(layer, getattr(p, "score", 0.0), hop=hop)
            probs.append(float(cal))

    if not probs:
        return 0.0

    probs.sort(reverse=True)
    # 噪声-OR：1 - Π(1 - p_i)，只用 top-N
    inv = 1.0
    for p in probs[:max(n, 1)]:
        inv *= (1.0 - max(0.0, min(1.0, p)))
    return round(1.0 - inv, 4)


# ════════════════════════════════════════════════════════════════════════
#                     参数拟合（离线，供生产替换默认值）
# ════════════════════════════════════════════════════════════════════════
def fit_platt(
    samples: list[tuple[float, int]],
    *,
    lr: float = 0.1,
    epochs: int = 500,
    init_a: float = 10.0,
    init_b: float = 0.55,
) -> PlattParams:
    """在标注数据上拟合单层的 Platt scaling 参数 (a, b)。

    这是把「默认先验参数」换成「数据驱动参数」的入口。
    你需要为每一层准备一份标注：对若干 query 跑该层检索，
    人工/GPT 标注每条召回是否真正相关。

    Args:
        samples: `[(raw_score, label), ...]`，label ∈ {0, 1}。
                 建议每层 ≥ 300 条、正负比例不要过于失衡。
        lr:      梯度下降学习率。
        epochs:  迭代轮数。
        init_a / init_b: 初值（用该层的默认参数即可）。

    Returns:
        拟合出的 PlattParams。把它序列化进 JSON 并用
        `RAG_CALIBRATION_FILE` 指向该文件即可生效。

    实现说明：目标是最小化 log-loss
        L = -Σ [ y·log(p) + (1-y)·log(1-p) ],  p = sigmoid(a(s-b))
    对 a、b 的梯度非常简洁（d = p - y）：
        ∂L/∂a = Σ d·(s-b)
        ∂L/∂b = Σ d·(-a)
    这里用最朴素的全批量 GD——数据量小（几百条），无需上 sklearn，
    保持本模块零额外依赖。

    示例：
        >>> ps = fit_platt([(0.3, 0), (0.4, 0), (0.7, 1), (0.8, 1)] * 30)
        >>> 0.0 < ps.b < 1.0
        True
    """
    if not samples:
        return PlattParams(init_a, init_b, note="no samples, returned init")

    a, b = float(init_a), float(init_b)
    n = float(len(samples))
    for _ in range(epochs):
        ga = gb = 0.0
        for s, y in samples:
            p = _sigmoid(a * (s - b))
            d = p - float(y)          # log-loss 对 logit 的梯度
            ga += d * (s - b)
            gb += d * (-a)
        a -= lr * ga / n
        b -= lr * gb / n
        # 约束 a > 0：保证单调递增（分数越高越可信），否则校准会反转层内排序
        a = max(a, 0.1)
    return PlattParams(
        a=round(a, 4), b=round(b, 4),
        note=f"fitted on {int(n)} labeled samples",
    )


# ════════════════════════════════════════════════════════════════════════
#                              自检 / 演示
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":  # pragma: no cover
    print("=" * 76)
    print("各层原始分 → 校准概率 对照表（这就是「跨层可比」的含义）")
    print("=" * 76)
    grid = {
        "L2_wiki":    [0.35, 0.45, 0.55, 0.65, 0.75],
        "L3_history": [0.35, 0.45, 0.55, 0.65, 0.75],
        "L4_web":     [1.00, 0.95, 0.90, 0.85, 0.80],
        "L5_kg":      [0.35, 0.45, 0.55, 0.70, 0.90],
    }
    for layer, scores in grid.items():
        cells = "  ".join(f"{s:.2f}→{calibrate(layer, s):.3f}" for s in scores)
        print(f"  {layer:<12} {cells}")

    print()
    print("=" * 76)
    print('Bug ① 修复验证：KG 多跳实体不再「凭空」高于种子')
    print("=" * 76)
    print("  用 `or` 兜底：traverse_multi_hop score=0.0 被 `or 0.9` 抬成 0.9")
    print(f"          → 校准后 = {calibrate('L5_kg', 0.9, hop=1):.3f}（比种子还高，荒谬）")
    print("  用 `is None` 判定：种子分数按跳数衰减")
    for h in (1, 2, 3):
        print(f"          hop={h}: raw 0.70 → {calibrate('L5_kg', 0.70, hop=h):.3f}")
    print(f"  未打分实体基准 {KG_UNSCORED_BASELINE} → "
          f"{calibrate('L5_kg', KG_UNSCORED_BASELINE):.3f}（足够低，不会阻止 L4 兜底）")

    print()
    print("=" * 76)
    print("聚合置信度（噪声-OR，只取 top-3）")
    print("=" * 76)

    class _P:  # 轻量假 Passage
        def __init__(self, layer, score):
            self.layer, self.score, self.metadata = layer, score, {}

    cases = {
        "只有弱 KG 命中（应触发 L4 兜底）": {"L5_kg": [_P("L5_kg", 0.35)]},
        "L2 强命中":                      {"L2_wiki": [_P("L2_wiki", 0.72)]},
        "L2+L4 双路一致":                 {"L2_wiki": [_P("L2_wiki", 0.62)],
                                          "L4_web": [_P("L4_web", 1.0)]},
        "10 条低质证据（不应堆出高分）":    {"L2_wiki": [_P("L2_wiki", 0.42) for _ in range(10)]},
    }
    for name, pl in cases.items():
        calibrate_passages([p for ps in pl.values() for p in ps])
        print(f"  {name:<34} conf = {aggregate_confidence(pl):.4f}")

    print()
    print("=" * 76)
    print("fit_platt 拟合演示（合成数据）")
    print("=" * 76)
    syn = [(0.30, 0), (0.40, 0), (0.50, 0), (0.60, 1), (0.70, 1), (0.80, 1)] * 60
    fitted = fit_platt(syn, init_a=10.0, init_b=0.55)
    print(f"  拟合结果: a={fitted.a} b={fitted.b}  ({fitted.note})")
    print(f"  0.45 → {_sigmoid(fitted.a * (0.45 - fitted.b)):.3f}   "
          f"0.65 → {_sigmoid(fitted.a * (0.65 - fitted.b)):.3f}")
