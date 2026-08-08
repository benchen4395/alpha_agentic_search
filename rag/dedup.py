# rag/dedup.py
"""近重去重（near-duplicate）+ MMR 多样性重排 —— P2-2c。

════════════════════════════════════════════════════════════════════════
要解决的问题：证据席位被「同一份信息」占满
════════════════════════════════════════════════════════════════════════
`FUSION_TOP_K = 6` 是喂给 LLM 的证据预算，它很小、很贵。而现在的去重
（`fusion._passage_key`）只做**精确**匹配：

    url::<完整url>                 # url 完全一致
    txt::<title[:40]>|<text[:80]>  # 标题前40字 + 正文前80字都一致

这挡不住真实世界最常见的一种重复 —— **同一条新闻被 N 家转载**：

    [1] 新华网   《我国上半年GDP同比增长5.3%》  https://news.cn/...
    [2] 人民网   《我国上半年GDP同比增长5.3%》  https://people.cn/...
    [3] 央视网   《上半年GDP同比增长5.3%》      https://cctv.com/...
    [4] 腾讯新闻 《我国上半年GDP同比增长5.3%》  https://qq.com/...

url 各不相同、正文前 80 字也常因导语改写而不同，于是 4 条全部通过去重，
**吃掉 6 个席位里的 4 个**。模型看到的是"4 个独立信源都这么说"，
但其实只有 1 份信息 —— 这带来三个后果：

  ① **信息量坍缩**：6 段证据的实际信息量只有 2~3 段，本该进来的
     补充视角（如"分季度拆解""与去年同期对比"）被挤掉；
  ② **虚假共识**：模型会因"多个来源一致"而提高确信度，
     但这些来源同源，独立性是假的 —— 这是幻觉的一种隐性来源；
  ③ **引用面板冗余**：用户看到 4 条几乎一样的来源，体验很差。

════════════════════════════════════════════════════════════════════════
为什么不能只靠「域名去重」
════════════════════════════════════════════════════════════════════════
最省事的做法是「同一域名只留 1 条」。但这个规则**误伤严重**：

    维基百科同一实体的不同小节        → 都在 wikipedia.org，但内容互补
    政府站的「政策原文」+「答记者问」 → 同域名，但一个是条款一个是解读
    大站的专题页（新华网的系列报道）  → 同域名，各篇讲不同侧面

域名相同 ≠ 内容相同。**真正要判定的是"文本讲的是不是同一件事"**，
这是语义问题，只能用语义方法解。所以本模块的判据是**向量余弦**，
域名只作为一个**辅助**信号（见 `_domain_of` 的用法说明）。

════════════════════════════════════════════════════════════════════════
两个阶段，职责不同（这是本模块最重要的设计）
════════════════════════════════════════════════════════════════════════
很多实现把"去重"和"多样性"混成一件事，结果两个目标互相干扰。
这里刻意拆成两步，**顺序不能颠倒**：

  【阶段 A】`drop_near_duplicates()` —— 硬去重，阈值很高（0.95）
      判定「这两段**实质上是同一份文本**」。这是**事实判断**，
      结果应该没有争议：余弦 0.95 以上的两段中文短文本，
      基本就是转载/改写关系。删掉是无损的。

  【阶段 B】`mmr_rerank()` —— 软重排，λ 权衡（0.7）
      在"已经不重复"的候选里，进一步偏好**互补**的段落。
      这是**偏好判断**，没有绝对对错 —— λ 就是"相关性 vs 多样性"
      这个 trade-off 的旋钮。

为什么必须先 A 再 B：MMR 的惩罚是**连续**的，对余弦 0.98 的转载
只会施加"较大惩罚"，但如果它的相关性分也很高，仍可能被选进来。
硬去重先把这类"确定该删的"清掉，MMR 才能专注于做它擅长的
"在有差异的候选之间平衡"。反过来（先 MMR）则会让转载稿参与
多样性计算，污染整个选择过程。

════════════════════════════════════════════════════════════════════════
阈值取值依据
════════════════════════════════════════════════════════════════════════
BGE-M3 在中文短文本上的经验分布（与 cache_policy.py 里实测的
「苹果CEO vs 苹果CFO = 0.8781」「2024GDP vs 2025GDP = 0.8531」同一量纲）：

    0.99+      逐字相同 / 仅差空白标点
    0.95~0.99  同一篇文章的转载、改写导语、截取不同长度   ← 该删
    0.88~0.95  同一事件的不同报道，含不同细节             ← 不该删，交给 MMR
    0.80~0.88  同一主题的不同子问题（CEO vs CFO）         ← 绝对不能删
    <0.80      不同话题

所以 `NEAR_DUP_THRESHOLD = 0.95` 落在「转载」与「不同报道」之间。
**这个阈值刻意取得保守（偏高）**：漏删一条转载的代价是浪费 1 个席位，
而误删一条独立证据的代价是可能丢掉答案本身 —— 两者不对称，
所以宁可漏删。这与 L1 缓存槽位门禁"宁可漏命中不可错命中"是同一取向。

════════════════════════════════════════════════════════════════════════
成本与降级（这里是全模块唯一的性能敏感点）
════════════════════════════════════════════════════════════════════════
需要 N 个候选的向量。实测（本机 BGE-M3 / MPS，中位数）：

    3 段  491 ms |  5 段  675 ms |  8 段  767 ms
    13 段 1241 ms | 16 段 1501 ms | 18 段 1691 ms
    → 约 80 ms/段 + 约 250 ms 固定开销

而算法本身几乎免费：近重去重 1.25 ms、MMR 0.80 ms。
**99%+ 的耗时都在编码** —— 要降延迟只有一条路：少编码几段。

【为什么不能"复用各层已有向量"】这条路我查过，**走不通**：
  * L4 web  —— 完全没有索引，snippet 是纯文本，**从来没被编码过**
  * L2/L3/L5 —— 检索时只编码 **query**；passage 向量在 FAISS 内部，
    `search()` 只返回 (metadata, score)，行号是局部变量直接丢弃
  也就是说 **passage 侧向量从来没被算过**，这不是"重复劳动"而是
  新增的必要成本。（`_resolve_vectors` 仍保留复用分支：未来若某层
  愿意把向量写进 metadata，就能白拿；但目前没有层这么做。）

【实际采用的优化：只对 L4 做】见 `diversify` 的 `target_layers`。
转载重复几乎只发生在 web 检索，所以把编码限定在 L4：
    纯离线命中（L4 未触发）→ 零编码
    L4 兜底触发           → 只编码 5 段（约 675ms 而非 1691ms）

任何一步失败（无 embedder、编码异常、numpy 缺失）都**原样返回输入**，
绝不让"优化项"变成"故障点"。这与 `_safe_search` 的取向一致。
"""
from __future__ import annotations

import os
from typing import Optional, Sequence
from urllib.parse import urlparse

from . import config as rag_config
from .types import Passage

try:
    import numpy as _np
except Exception:                        # pragma: no cover - numpy 是硬依赖，兜底而已
    _np = None                           # type: ignore


# ============================================================
#                        参数
# ============================================================
# 近重判定阈值：余弦 >= 此值视为「同一份文本」。见模块 docstring 的分布表。
# 偏高是刻意的：漏删浪费 1 个席位，误删可能丢掉答案，代价不对称。
NEAR_DUP_THRESHOLD: float = float(os.getenv("RAG_NEAR_DUP_THRESHOLD", "0.95"))

# MMR 的 λ：最终得分 = λ * 相关性 - (1-λ) * 与已选集合的最大相似度
#
# λ=1.0 → 完全退化为原排序（等于关闭 MMR）
# λ=0.0 → 只要多样性，会把最不相关的段落选进来（显然不行）
#
# 取 0.7 的依据：相关性仍是主导（0.7 vs 0.3）。检索场景里"相关"是
# 及格线、"多样"是加分项 —— 一段不相关但很独特的文本对回答毫无价值。
# 业界（MMR 原论文 Carbonell & Goldstein 1998，以及后续 IR 实践）
# 在"文档检索"任务上的常用区间是 0.6~0.8，0.7 是中位取值。
MMR_LAMBDA: float = float(os.getenv("RAG_MMR_LAMBDA", "0.7"))

# 向量在 Passage.metadata 里的键名。
#
# ⚠️ 目前**没有任何层**会写这个键（已核实 rag/layers.py 里
# "embedding" 出现 0 次）—— 各层检索时只编码 query，passage 向量在
# FAISS 内部且 `VectorStore.search()` 不返回行号，取不出来。
# 保留这条复用分支是为了将来：若某层愿意把向量带出来（需要改
# VectorStore 签名），这里就能零成本复用，不必再改 dedup。
EMBEDDING_META_KEY: str = "embedding"


# ============================================================
#                      内部工具
# ============================================================
def _domain_of(url: str) -> str:
    """取 url 的注册域名（用于日志与「同域加权」的辅助判断）。

    ⚠️ 注意这个函数**不用于**直接去重。域名相同不代表内容相同
    （维基同一实体的不同小节、政府站的原文+解读都是同域名但互补），
    详见模块 docstring 的「为什么不能只靠域名去重」。

    它只有两个用途：
      ① 打日志时让人看懂"删掉的是哪几家转载"；
      ② `drop_near_duplicates` 里对**同域名**的段落把阈值放宽一点点
         —— 同一家站点内部的近重（同一文章的 amp 版 / 分页版）
         比跨站转载更可能是真重复。
    """
    if not url:
        return ""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    # 去掉 www. 前缀，让 www.news.cn 与 news.cn 视为同一家
    return host[4:] if host.startswith("www.") else host


def _texts_for_embedding(passages: Sequence[Passage]) -> list[str]:
    """构造用于相似度计算的文本。

    用 `title + text` 而非只用 text，原因：
      * L5 KG 的 passage 正文可能极短（"张三 —— 职业：演员"），
        标题携带了关键的实体信息；
      * 转载稿的标题往往比正文更一致（正文导语常被改写），
        带上标题能提高近重的召回。

    截断到 512 字符：BGE-M3 的有效上下文远大于此，但近重判定
    只需要"开头够不够像"—— 转载稿如果前 512 字都一样，
    后面基本不会突然变得不一样。截断能显著降低编码耗时。
    """
    out = []
    for p in passages:
        t = f"{p.title} {p.text}".strip() if p.title else p.text
        out.append(t[:512])
    return out


def _resolve_vectors(
    passages: Sequence[Passage],
    embedder=None,
) -> Optional["_np.ndarray"]:
    """拿到所有 passage 的 L2 归一化向量矩阵 (N, D)；失败返回 None。

    两条路径，优先零成本的那条：
      ① **复用**：若某个 passage 的 metadata 里已有 `embedding`
         （L2/L3 检索时顺手写入的），直接取用；
      ② **补算**：仍缺的那些统一走一次 `embedder.embed_batch()`
         批量编码（BGE-M3 批量比逐条快很多）。

    注意只对"仍缺的"补算，而不是全量重算 —— 混合场景（L2 带了向量、
    L4 web 没带）下能省掉大部分编码量。

    返回 None 的情形（全部按"不做去重"降级处理）：
      * numpy 不可用
      * 没有 embedder 且 metadata 里也没有现成向量
      * 编码抛异常
      * 向量维度不一致（不同来源的向量空间不可比，强行算余弦无意义）
    """
    if _np is None or not passages:
        return None

    n = len(passages)
    vecs: list[Optional[list[float]]] = [None] * n

    # ---- ① 复用 metadata 里已有的向量 ----
    for i, p in enumerate(passages):
        v = p.metadata.get(EMBEDDING_META_KEY)
        if v:
            try:
                vecs[i] = [float(x) for x in v]
            except Exception:
                vecs[i] = None

    # ---- ② 给仍缺的批量补算 ----
    missing = [i for i, v in enumerate(vecs) if v is None]
    if missing:
        if embedder is None:
            return None
        texts = _texts_for_embedding([passages[i] for i in missing])
        try:
            got = embedder.embed_batch(texts)
        except Exception as e:
            print(f"[dedup] 向量编码失败，跳过去重: {e}")
            return None
        for slot, v in zip(missing, got):
            if not v:
                return None                  # 有一条编不出来就整体降级，避免矩阵缺行
            vecs[slot] = [float(x) for x in v]

    # ---- 维度一致性校验 ----
    # 不同来源的向量可能来自不同模型（理论上不该发生，但 metadata 是
    # 开放字段，谁都能往里写）。维度不齐会让 np.asarray 抛
    # inhomogeneous shape；即使维度凑巧一致但模型不同，余弦也没有意义。
    dims = {len(v) for v in vecs if v}
    if len(dims) != 1:
        print(f"[dedup] 向量维度不一致 {dims}，跳过去重")
        return None

    mat = _np.asarray(vecs, dtype=_np.float32)
    # 保险再归一化一次（Embedder.embed 已归一，这里幂等）：
    # 只有单位向量才满足 cos = dot，下面全程用矩阵乘算余弦。
    norms = _np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ============================================================
#            阶段 A：近重去重（硬删，事实判断）
# ============================================================
def drop_near_duplicates(
    passages: list[Passage],
    embedder=None,
    threshold: Optional[float] = None,
    verbose: bool = True,
) -> list[Passage]:
    """删掉与「已保留段落」语义近重的段落，保序。

    算法：贪心扫描（O(N·K)，K = 已保留数，N 很小所以完全够用）
        按输入顺序遍历 → 与**已保留**的每一段算余弦 →
        任一超过阈值就丢弃，否则保留。

    为什么按输入顺序扫、而不是先聚类再选代表：
        输入顺序就是 RRF 相关性降序，**先来的那条天然是更好的代表**
        （排名更高 = 多路一致性更强）。聚类还要额外定义"选谁当代表"，
        引入不必要的复杂度和不确定性。

    为什么保留者是"先来的"而不是"文本最长的"：
        看起来长文本信息更多，但实践中长的往往是**带导航栏/推荐位的
        脏页面**，而不是正文更全。相关性排序比长度更可信。

    Args:
        passages:  候选段落（应已按相关性降序）。
        embedder:  用于补算向量；None 时只能复用 metadata 里现成的向量。
        threshold: 覆盖默认的 `NEAR_DUP_THRESHOLD`（测试/调参用）。
        verbose:   是否打印被删清单（上线后据此判断阈值是否合适）。

    Returns:
        去重后的列表（保持原相对顺序）。任何异常/降级情形都**原样返回输入**。
    """
    if len(passages) < 2:
        return passages

    mat = _resolve_vectors(passages, embedder)
    if mat is None:
        return passages                      # 降级：拿不到向量就不做去重

    thr = float(NEAR_DUP_THRESHOLD if threshold is None else threshold)

    kept_idx: list[int] = []
    # 记录"谁被谁挤掉"，只为日志可读；上线后看这个能判断阈值合不合适。
    dropped: list[tuple[int, int, float]] = []      # (被删, 保留者, 余弦)

    for i in range(len(passages)):
        dup_of = -1
        dup_sim = 0.0
        for j in kept_idx:
            sim = float(mat[i] @ mat[j])
            # 同域名时略微放宽阈值（更容易判为重复）。
            # 依据：同一站点内部的 amp 版 / 分页版 / 移动版是真重复的
            # 概率更高；跨站转载则可能有独立的编辑加工。
            # 只放宽 0.02，是很轻的先验 —— 不足以让"同域名不同小节"
            # （典型余弦 0.7~0.9）被误删。
            eff = thr - 0.02 if _domain_of(passages[i].url) and \
                _domain_of(passages[i].url) == _domain_of(passages[j].url) else thr
            if sim >= eff:
                dup_of, dup_sim = j, sim
                break
        if dup_of >= 0:
            dropped.append((i, dup_of, dup_sim))
        else:
            kept_idx.append(i)

    if dropped and verbose:
        # 打印格式刻意带上域名和余弦：上线后如果发现误删，
        # 从日志就能直接看出是"哪两家、相似到什么程度"被判重了，
        # 不需要复现就能决定阈值该往哪调。
        detail = ", ".join(
            f"#{i+1}({_domain_of(passages[i].url) or 'n/a'})"
            f"≈#{j+1}({_domain_of(passages[j].url) or 'n/a'}):{s:.3f}"
            for i, j, s in dropped[:6]
        )
        print(
            f"[dedup] 🧹 近重去重：{len(passages)} → {len(kept_idx)} 段"
            f"（阈值 {thr}）[{detail}]"
        )

    return [passages[i] for i in kept_idx]


# ============================================================
#          阶段 B：MMR 多样性重排（软偏好，权衡判断）
# ============================================================
def mmr_rerank(
    passages: list[Passage],
    embedder=None,
    top_k: Optional[int] = None,
    lambda_: Optional[float] = None,
) -> list[Passage]:
    """MMR（Maximal Marginal Relevance）重排：在相关性与多样性之间权衡。

    公式（Carbonell & Goldstein, SIGIR'98）：
        MMR = argmax_{d ∈ R\\S} [ λ·Rel(d) − (1−λ)·max_{d'∈S} Sim(d, d') ]

    其中 R 是候选集、S 是已选集。逐个贪心选择：每一步选"相关性高
    **且**与已选内容差异大"的那一段。

    ════════════════════════════════════════════════════════════════
    与阶段 A 的分工（务必先跑 A）
    ════════════════════════════════════════════════════════════════
    A 做的是**事实判断**（这两段是不是同一份文本），阈值 0.95，硬删。
    B 做的是**偏好判断**（在有差异的候选里，更想要互补的），λ 权衡，软排。

    如果跳过 A 直接跑 B：余弦 0.98 的转载稿只会被"惩罚"，
    若它相关性也高，仍可能挤进结果。硬删必须在前面。

    ════════════════════════════════════════════════════════════════
    相关性分数的口径问题（这里有个真实的坑）
    ════════════════════════════════════════════════════════════════
    `Rel(d)` 不能直接用 `p.score`。因为到这一步 `p.score` 已经被
    `rrf_fuse` 换成了 **RRF 贡献值** Σ1/(60+rank)，典型 0.016~0.03
    —— 那是**秩倒数，不是相似度**（这个坑在 retriever.py 里
    `calibrate_passages(..., overwrite=False)` 那处已经踩过一次）。

    直接代进公式会出事：Rel 的量级 ~0.02，而 Sim 的量级 ~0.9，
    于是 `λ·0.02 - 0.3·0.9` 里**多样性项完全压倒相关性项**，
    MMR 退化成"只挑最不像的"，等于把相关性排序整个丢掉。

    【解法】把 Rel 归一化到 [0,1] 的同一量纲：
        Rel_i = (score_i − min) / (max − min)
    这样两项可比，λ 才真正是"权重"而不是"随量纲漂移的巧合"。
    用 min-max 而非 softmax：min-max 保持线性、可解释，
    且 rank 序完全不变（我们只想改量纲，不想改相对关系）。

    Args:
        passages: 候选（应已按相关性降序，且已过阶段 A）。
        embedder: 用于补算向量。
        top_k:    最终返回段数；None = 全部重排。
        lambda_:  覆盖默认 `MMR_LAMBDA`。1.0 等价于不做 MMR。

    Returns:
        重排后的列表。降级情形（拿不到向量等）**原样返回输入**。
    """
    k = top_k or len(passages)
    if len(passages) <= 1 or k <= 1:
        return passages[:k]

    lam = float(MMR_LAMBDA if lambda_ is None else lambda_)
    if lam >= 1.0:
        # λ=1 时多样性项系数为 0，MMR 完全退化为原排序。
        # 直接短路，省掉一次编码 —— 这也是"关闭 MMR"的正确方式。
        return passages[:k]

    mat = _resolve_vectors(passages, embedder)
    if mat is None:
        return passages[:k]                  # 降级：拿不到向量就保持原序

    # ---- 相关性归一化到 [0,1]（见 docstring 的量纲说明）----
    raw = _np.asarray([float(p.score) for p in passages], dtype=_np.float32)
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-9:
        # 所有分数相同（例如全部来自同一层同一 rank）→ 无相关性信息可用，
        # 此时 Rel 项是常数，MMR 变成纯多样性选择。给全 1 而不是全 0：
        # 全 0 会让公式只剩惩罚项，语义上等价，但全 1 更直观。
        rel = _np.ones_like(raw)
    else:
        rel = (raw - lo) / (hi - lo)

    n = len(passages)
    selected: list[int] = []
    remaining = set(range(n))

    # 第一段：直接取相关性最高的。
    # 此时已选集为空，max_{d'∈S} Sim = 0，公式退化为 argmax Rel。
    first = int(_np.argmax(rel))
    selected.append(first)
    remaining.discard(first)

    while remaining and len(selected) < k:
        best_i, best_score = -1, -1e18
        for i in remaining:
            # 与**已选集合**的最大相似度（不是平均）。
            # 用 max 而非 mean 的理由：只要与任意一条已选段落高度重复，
            # 这一段就该被压制 —— 用 mean 会被大量不相关的已选段落稀释，
            # 让重复项蒙混过关。
            sim_max = max(float(mat[i] @ mat[j]) for j in selected)
            score = lam * float(rel[i]) - (1.0 - lam) * sim_max
            if score > best_score:
                best_i, best_score = i, score
        if best_i < 0:
            break
        selected.append(best_i)
        remaining.discard(best_i)

    return [passages[i] for i in selected]


# ============================================================
#                     对外统一入口
# ============================================================
def _run_stages(
    passages: list[Passage],
    embedder,
    top_k: Optional[int],
    use_dup: bool,
    use_mmr: bool,
    verbose: bool,
) -> list[Passage]:
    """执行「近重去重 → MMR」两阶段。被 `diversify` 的两种模式共用。

    ⚠️ 顺序固定，不可配置。原因见 `mmr_rerank` 的 docstring：
    反过来会让转载稿参与多样性计算、污染整个选择过程。
    这属于**正确性**而非偏好，所以不给开关。

    ════════════════════════════════════════════════════════════════════
    向量只编码一次（避免两阶段各编一遍）
    ════════════════════════════════════════════════════════════════════
    `drop_near_duplicates` 和 `mmr_rerank` 都会调 `_resolve_vectors`。
    如果不做任何处理，两阶段串联时会**编码两次**：

        阶段 A 编码 N 段  →  删掉 k 段  →  阶段 B 又编码 N−k 段

    实测（回归测试抓到的）：5 段 L4 候选实际编码了 8 段
    （A 编 5 段 + B 编剩下的 3 段）。按 80ms/段算，白白多花约 240ms ——
    这恰好抵消掉"只对 L4 去重"省下来的一部分，很不划算。

    【解法】先统一编码一次，把向量塞进 `metadata[EMBEDDING_META_KEY]`。
    两个阶段内部的 `_resolve_vectors` 有**复用分支**（优先读 metadata），
    于是它们都直接取用，不再触发编码。

    【为什么最后要清理掉】metadata 会随 Passage 一路带到
    `evidence.build_evidence_block` → `Source` → `AnswerResult.to_dict()`
    → JSON 输出。1024 个 float 序列化出来是几十 KB，会让
    `scripts/search.py` 的输出膨胀几十倍，也会进 L3 归档。
    所以只在本函数内借用，出去之前**恢复原状** ——
    只删"我们自己塞进去的"，不碰调用方本来就写好的向量。
    """
    borrowed: list[Passage] = []
    if (use_dup or use_mmr) and len(passages) > 1:
        mat = _resolve_vectors(passages, embedder)
        if mat is not None:
            for p, row in zip(passages, mat):
                if EMBEDDING_META_KEY not in p.metadata:
                    p.metadata[EMBEDDING_META_KEY] = [float(x) for x in row]
                    borrowed.append(p)

    try:
        out = passages
        if use_dup:
            out = drop_near_duplicates(out, embedder=embedder, verbose=verbose)
        if use_mmr:
            out = mmr_rerank(out, embedder=embedder, top_k=top_k)
        elif top_k:
            out = out[:top_k]
        return out
    finally:
        # 只清理本函数塞进去的，调用方原有的向量保持不动。
        # 放在 finally：即使某阶段抛异常，也不能把大向量留在 metadata 里。
        for p in borrowed:
            p.metadata.pop(EMBEDDING_META_KEY, None)


def diversify(
    passages: list[Passage],
    embedder=None,
    top_k: Optional[int] = None,
    enable_near_dup: Optional[bool] = None,
    enable_mmr: Optional[bool] = None,
    target_layers: Optional[frozenset[str]] = None,
    verbose: bool = True,
) -> list[Passage]:
    """去重 + 多样性的统一入口：先硬去重，再 MMR 重排。

    这是 `rag/retriever.py` 唯一需要调用的函数。两个阶段各有独立开关，
    便于 A/B 与故障隔离：怀疑哪一步有问题就单独关掉它，
    不需要改代码（走 `RAG_ENABLE_NEAR_DUP` / `RAG_ENABLE_MMR` 环境变量）。

    ════════════════════════════════════════════════════════════════════
    `target_layers`：只对指定层做语义去重（延迟优化，见 config 的长注释）
    ════════════════════════════════════════════════════════════════════
    【为什么需要它】唯一的真实开销是 **BGE-M3 编码**。实测（本机）：

        3 段  491 ms | 5 段  675 ms | 8 段  767 ms
        13 段 1241 ms | 16 段 1501 ms | 18 段 1691 ms

    近线性（约 80 ms/段 + 约 250 ms 固定开销）。而去重算法本身只要
    1.25 ms、MMR 只要 0.80 ms —— 也就是说 **99%+ 的耗时都在编码**。
    要降延迟，唯一有效的手段就是**少编码几段**。

    【为什么可以只挑 L4】转载重复**几乎只发生在 web 检索**：
    同一条新闻被新华网/人民网/央视网转载是 L4 特有的现象；
    维基的不同小节、KG 三元组、历史问答都不存在这种形态的重复
    （它们各自是结构化去重过的语料）。

    所以把语义去重限定在 L4，能用约 1/3 的编码量拿到绝大部分收益。

    【代价（这是个妥协，必须说清楚）】
      * **跨层近重挡不住**：若 L2 wiki 与 L4 web 命中同一主题的不同页面，
        且语义高度重合，这种重复不会被删。
        缓解：完全相同 URL 的情况 `fusion._passage_key` 的精确去重已覆盖；
        真正漏掉的是"不同 URL 但内容近重"的跨层组合，实测远比 L4 内部
        的转载罕见。
      * **MMR 只在目标层内生效**：非目标层的段落不参与多样性计算，
        它们按原 RRF 顺序保留。

    想要全量模式（更彻底但更慢）：传 `target_layers=frozenset()` 或
    把 `RAG_DEDUP_LAYERS` 设为空串。

    Args:
        passages:        候选（应已按相关性降序）。
        embedder:        BGE-M3 Embedder（`rag.embedder.Embedder`）。
        top_k:           最终段数。
        enable_near_dup: None → 读 config.ENABLE_NEAR_DUP。
        enable_mmr:      None → 读 config.ENABLE_MMR。
        target_layers:   只对这些 layer 的段落做语义去重/重排。
                         None  → 读 config.DEDUP_LAYERS（默认 {"L4_web"}）
                         空集合 → 全量模式（对所有段落做，行为同改造初版）
        verbose:         是否打印去重日志。

    Returns:
        处理后的段落列表，长度 ≤ top_k。
    """
    if not passages:
        return passages

    use_dup = (
        rag_config.ENABLE_NEAR_DUP if enable_near_dup is None else enable_near_dup
    )
    use_mmr = rag_config.ENABLE_MMR if enable_mmr is None else enable_mmr
    if not (use_dup or use_mmr):
        # 两个开关都关 → 逐段等价于原输入（零回归保证），且**不触发编码**
        return passages[:top_k] if top_k else passages

    layers = (
        rag_config.DEDUP_LAYERS if target_layers is None else target_layers
    )

    # ---- 全量模式：对所有候选做（更彻底，但编码量最大）----
    if not layers:
        return _run_stages(passages, embedder, top_k, use_dup, use_mmr, verbose)

    # ══════════════════════════════════════════════════════════════════
    # 分层模式：只把目标层的段落送去编码
    # ══════════════════════════════════════════════════════════════════
    # 拆开处理再合并。合并方式是这里唯一有技术含量的地方，先说清楚
    # 两个**都不能用**的直觉做法：
    #
    #   ✗ 合并后按 score 重排
    #       会把 MMR 的成果整个抹掉。MMR 的产出是**一个顺序**
    #       （"先看这条，再看这条最互补的"），它刻意不按 score 排 ——
    #       按 score 重排等于白跑一遍 MMR。
    #
    #   ✗ kept + others 直接拼接
    #       等于把所有 L4 段落无条件排在离线段落之前（或之后），
    #       破坏了 quota_fuse 建立的跨层相关性排序。
    #
    # ✓ 正确做法：**保留原始的层交错位置，只在目标层的槽位里填新顺序**。
    #   输入 passages 已是 quota_fuse 的相关性降序，其中目标层段落占据
    #   若干"槽位"。去重删掉了一部分，于是：
    #       原始:  [L4a, W1, L4b, L4c, W2]      # W=离线
    #       槽位:  [ ①,      ②,   ③     ]      # 3 个 L4 槽位
    #       MMR 后 kept = [L4c, L4a]（顺序变了、数量少了）
    #       填充:  [L4c, W1, L4a, W2]           # 前 2 个槽位按 MMR 顺序填，
    #                                          # 第 3 个槽位空缺自然消失
    #   这样两个排序语义同时保住：跨层的相关性交错来自原始顺序，
    #   目标层内部的多样性顺序来自 MMR。
    #
    # 为什么顺序这么重要：下游 `evidence.build_evidence_block` 按顺序
    # 编号 <doc id=1..n>，而 LLM 对靠前的证据权重更高 —— 顺序**有语义**，
    # 不是"反正都在里面就行"。
    target: list[Passage] = []
    slots: list[int] = []          # 目标层段落在原列表中的下标
    for i, p in enumerate(passages):
        if p.layer in layers:
            target.append(p)
            slots.append(i)

    if not target:
        # 本轮没有目标层的段落（如纯离线命中，L4 未触发）→ 零编码。
        # 这是最常见的省时场景：离线层之间本来就没有转载问题。
        return passages[:top_k] if top_k else passages

    # top_k=None：先不截断，留给合并后统一截断。
    # 否则会先把目标层裁到 top_k，再和其余段落合并后又裁一次，双重截断
    # 会让目标层被过度压缩（例如 top_k=6 时 L4 只剩 6 段里的一部分）。
    kept = _run_stages(target, embedder, None, use_dup, use_mmr, verbose)

    # 按"槽位"回填：第 n 个保留下来的目标段落放进第 n 个槽位。
    # kept 通常比 slots 短（去重删了几条），多余的槽位直接留空。
    fill = dict(zip(slots[:len(kept)], kept))
    dropped_slots = set(slots[len(kept):])

    merged: list[Passage] = []
    for i, p in enumerate(passages):
        if i in dropped_slots:
            continue               # 该槽位对应的段落已被去重删掉
        merged.append(fill.get(i, p))

    return merged[:top_k] if top_k else merged
