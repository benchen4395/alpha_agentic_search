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


# =============================================================================
# 上下文词面匹配（description ↔ query）
# =============================================================================
#
# 【要解决的问题：query_context 传了但完全没起作用】
# ---------------------------------------------------------------------------
# 实测同一个 mention、语义完全相反的两个上下文，返回结果与打分**一模一样**：
#     "华盛顿是美国第一任总统"  → 華盛頓哥倫比亞特區(0.18)  ✗ 应为 喬治·華盛頓
#     "美国首都华盛顿的人口"    → 華盛頓哥倫比亞特區(0.18)  ✓
#     "特斯拉发明了交流电"      → 特斯拉公司(0.07)          ✗ 应为 尼古拉·特斯拉
#     "特斯拉发布了新款电动车"  → 特斯拉公司(0.07)          ✓
# 即消歧退化成了"按热度返回一个固定答案"，碰巧对一半。
#
# 【根因】`link()` 里的 cosine 只在候选**命中热门实体向量库**时才计算，
# 而那个库只有 2871 条，对比 KG 的 1038 万实体 —— 覆盖率约 0.03%。
# 绝大多数候选走 `cold_pos` 分支，score = 0.5 * prior，**与上下文无关**。
#
# 【为什么用词面匹配，而不是给 cold 候选现算 embedding】
# 实测对比过（9 个消歧 case，正确答案均在候选内）：
#     baseline(现状)              5/9    0ms
#     现算 description embedding  4/9    673ms/题   ← 更差，而且慢
#     description 词面重叠        8/9    ~0ms       ← 采用
# embedding 方案失败的原因很具体：description 极短（"磁感應單位強度"7 字、
# "车型"2 字），BGE-M3 在这种长度上向量不稳定，反而把「特斯拉工廠」
# 「苹果产品发布会」这类**字面接近但语义错误**的候选拉了上来，还弄坏了
# baseline 本来正确的 3 个 case。
# 这是个反直觉但可复现的结论：**短文本上词面匹配优于语义向量**。
# description 本身是人工撰写的高信息密度短语（"第1任美国总统"、
# "美籍塞尔维亚族发明家"、"美利堅合眾國的首都"），词面命中already足够判别。

def _char_bigrams(text: str) -> set[str]:
    """取字符级 bigram 集合。

    用 bigram 而非单字：单字在中文里太容易偶然命中（"国""的""人"），
    会让所有候选都拿到分从而失去判别力。bigram 兼顾了中文无空格分词的
    现实与判别性，且不依赖任何分词器。
    """
    s = "".join(ch for ch in text if ch.strip())
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def context_lexical_score(query: str, description: str) -> float:
    """description 与 query 的词面重叠度，取值 0~1。

    用 description 侧作分母（而非并集/query 侧）：我们要问的是
    "这条 description 有多少内容被 query 提到了"，属于**精确率**语义。
    若用 query 侧作分母，长 query 会把所有候选的分数一起压低，
    候选之间的相对差异被抹平，判别力下降。

    Args:
        query:       用户 query 全句（提供消歧上下文）
        description: 候选实体的 Wikidata 描述

    Returns:
        0~1 的重叠比例；description 为空时返回 0.0（退化为不提供信号）。

    示例（实测）：
        >>> context_lexical_score("华盛顿是美国第一任总统", "第1任美国总统")
        0.5     # 命中 "任美"/"美国"/"总统" 等 bigram
        >>> context_lexical_score("华盛顿是美国第一任总统", "美利堅合眾國的首都")
        0.0     # 无重叠 → 不加分
    """
    if not description:
        return 0.0
    dg = _char_bigrams(description)
    if not dg:
        return 0.0
    return len(dg & _char_bigrams(query)) / len(dg)


# 上下文词面分在最终打分中的权重。
#
# 取 0.5 的依据（实测 9 个消歧 case，正确答案均在候选内）：
#     权重 0.0（即现状，纯先验）        5/9
#     权重 0.5                        8/9   ← 采用
# 之所以敢给到与先验分等权，是因为词面分**只在真的有重叠时才非零**：
# description 与 query 无关时它是 0.0，不会干扰先验排序；只有 query
# 明确提到了 description 里的内容（"第1任美国总统" ↔ "…第一任总统"）
# 才会介入。这种"有信号才发声"的特性使它不需要保守加权。
_CTX_W: float = 0.5


# =============================================================================
# 类型-意图匹配（query 问的是人还是物）
# =============================================================================
#
# 【要解决的问题：description 零重叠时消歧退化成拼热度】
# ---------------------------------------------------------------------------
#     "牛顿提出了万有引力定律" → 牛顿(Q49196, desc="美国城市")  ✗
#                                应为 艾薩克·牛頓(Q935)
# 词面分在这里救不了：Q49196 的 "美国城市" 与 query 零重叠，Q935 的
# "英格兰物理学家、数学家…" 也零重叠（query 说的是"万有引力定律"而非
# "物理学家"）—— 两个候选 ctx 都是 0.0，只能比先验热度，
# 而城市热度(135) 高于 牛顿本人(60)，于是必错。
#
# 【突破口：实体侧的类型信号极其干净】（实测）
#     艾薩克·牛頓  P31=[人類(Q5)]         P106=[数学家, 哲學家]
#     牛顿(城市)   P31=[美国城市, 城市…]  P106=[]
#     喬治·華盛頓  P31=[人類(Q5)]         P106=[政治人物, 农民…]
#     華盛頓特區   P31=[美国城市…]        P106=[]
#     尼古拉·特斯拉 P31=[人類(Q5)]        P106=[发明家, 电气工程师…]
#     特斯拉公司   P31=[法人團體, 企業…]  P106=[]
# `P31 → Q5(人類)` 是**二值且无歧义**的人物判定，比 description 可靠得多。
#
# 【query 侧：用谓词线索判断问的是人还是物】
# 中文里主谓搭配有强约束："提出/发明/创立/担任/毕业于" 的主语必须是人；
# "位于/成立于/人口/面积/单位/等于" 的主语必须不是人。
# 实测这套线索在 11 个消歧 case 上把意图判得很干净：
#     需要人物的 case  意图分全部 > 0
#     需要非人的 case  意图分全部 ≤ 0
#
# 【关键设计：无信号时完全不介入】
# 意图分为 0（query 里没有任何线索）时类型分恒为 0，排序与修复前一致。
# 实测 11 个 case 里有 5 个意图分为 0（"水星是太阳系最内侧的行星" 等），
# 它们本来就排对了，不介入正是我们想要的行为 —— 这保证了
# **只可能修好、不可能弄坏**。实测：10/11 → 11/11，弄坏 0 个。

# 人类行为谓词：这些动作/关系的主语只能是人
_PERSON_CUES: tuple[str, ...] = (
    "提出", "发明", "创立", "创建", "出生", "逝世", "去世", "担任",
    "领导", "写了", "撰写", "发现", "证明", "当选", "第一任",
    "任总统", "毕业于", "师从", "曾在",
)

# 非人谓词：指向地点 / 组织 / 计量单位 / 物种等
_NONPERSON_CUES: tuple[str, ...] = (
    "位于", "成立于", "总部", "人口", "面积", "海拔", "单位",
    "等于", "换算", "首都", "发布了", "股价", "市值", "型号",
    "这种", "营养",
)

# 类型分权重。0.30 的依据：它必须能翻盘先验分的差距，但不能碾压词面分。
# 实测 "牛顿" 那组先验分差约 0.18（城市 0.35 vs 人 0.17，各乘 1-_CTX_W），
# 类型分给出 ±0.30 的差值（一致 +0.30、冲突 -0.30，相对差 0.60）足以翻盘；
# 同时 0.30 < _CTX_W(0.5)，保证 description 明确命中时词面分仍占主导。
_TYPE_W: float = 0.30

# Q5 = human / 人類。Wikidata 里所有人物都有 P31→Q5。
_QID_HUMAN: str = "Q5"


def query_person_intent(query: str) -> int:
    """判断 query 问的是人还是物。

    Returns:
        > 0 倾向人物；< 0 倾向非人物；**== 0 表示无线索**（调用方应不介入）。

    示例（实测）：
        >>> query_person_intent("牛顿提出了万有引力定律")   #  +1 → 人
        >>> query_person_intent("力的单位牛顿等于多少")      #  -2 → 物
        >>> query_person_intent("水星是太阳系最内侧的行星")  #   0 → 不介入
    """
    if not query:
        return 0
    return (sum(1 for k in _PERSON_CUES if k in query)
            - sum(1 for k in _NONPERSON_CUES if k in query))


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
        # P31→Q5 人物判定的进程内缓存（见 _is_human）
        self._human_cache: Dict[str, bool] = {}

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
    def _is_human(self, qid: str) -> bool:
        """该实体是否为人物（P31 → Q5）。

        带进程内缓存：同一批候选里常出现重复 qid，且热门实体跨 query 复现率高。
        任何异常都返回 False —— L5 的异常会被上层 `_safe_search` 吞掉导致
        整层静默返回空（见 kg_store.py 里记录的同类故障），所以这里必须容错。
        """
        cache = self.__dict__.setdefault("_human_cache", {})
        hit = cache.get(qid)
        if hit is not None:
            return hit
        try:
            row = self.kg.conn.execute(
                "SELECT 1 FROM triples WHERE subject_qid=? "
                "AND predicate_pid='P31' AND object_qid=? LIMIT 1",
                (qid, _QID_HUMAN),
            ).fetchone()
            result = row is not None
        except Exception:
            result = False
        cache[qid] = result
        return result

    def _type_score(self, qid: str, intent: int) -> float:
        """类型-意图一致性分：+1 一致 / -1 冲突 / 0 无线索。"""
        if intent == 0:
            return 0.0
        return 1.0 if (self._is_human(qid) == (intent > 0)) else -1.0

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

        # ---- 上下文词面分：所有候选都算，与热门向量库是否命中无关 ----
        # 这是修复"query_context 传了却没起作用"的关键：热门向量库只覆盖
        # 2871 条实体（KG 有 1038 万），绝大多数候选拿不到 cosine，
        # 此前只能吃纯先验分 → 消歧退化成"按热度返回固定答案"。
        # 词面分不依赖任何索引，对每个候选都可用，成本≈0。
        ctx_text = query_context or ""
        for c in cands:
            c["ctx"] = (context_lexical_score(ctx_text, c.get("description") or "")
                        if ctx_text else 0.0)

        # ---- 类型-意图分：query 问人还是问物 ----
        # 解决 description 零重叠时只能拼热度的问题（"牛顿提出万有引力" 会
        # 选到 desc="美国城市" 的 Q49196，因为城市热度更高）。
        # intent==0 时 _type_score 恒返回 0，排序与不启用本特性完全一致。
        intent = query_person_intent(ctx_text)
        for c in cands:
            c["type_s"] = self._type_score(c["qid"], intent)

        # ---- 没有热门 embedding 时：先验分 + 上下文词面分 + 类型分 ----
        # ⚠️ 不能退化成纯 weight：weight 只有 1.0/0.8/0.6/0.5 四档，
        # 大量同名候选全是 1.0，排不出顺序来，等于把 KGStore 里刚算好的
        # 入度信息丢掉 —— “中国”会回到指向 1972 年的意大利电影。
        if self.emb is None or len(cands) <= 1:
            for c in cands:
                c["score"] = (_CTX_W * c["ctx"]
                              + (1.0 - _CTX_W) * c["prior"]
                              + _TYPE_W * c["type_s"])
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
                # 融合分：cosine + 先验分 + 上下文词面分
                # 先验分 = weight × log1p(入度) 归一化，既包含 label/alias
                # 的证据强度，也包含实体重要度。
                # ⚠️ 原实现是 `0.7*cosine + 0.3*weight`，直接用裸 weight，
                # 把 KGStore 里算好的入度信号**整个覆盖掉了** —— 这是
                # 为什么修了 kg_store 后，link("中国") 已经能排对，
                # 但走完整管线的 retrieve() 仍然返回「1972 年的意大利电影」。
                cands[pos]["score"] = (
                    (1.0 - _CTX_W) * (0.6 * float(sim)
                                      + 0.4 * cands[pos]["prior"])
                    + _CTX_W * cands[pos]["ctx"]
                    + _TYPE_W * cands[pos]["type_s"]
                )
        # 冷门候选：没有 embedding，用先验分 + 上下文词面分 + 类型分。
        # 0.5 系数保留：避免冷门候选与热门候选的分数尺度差异过大。
        # ⚠️ 类型分**不乘 0.5**：它是二值判定（是人/不是人），与候选冷热无关，
        # 若跟着缩放会让冷门候选的类型信号弱一半，而人物实体恰恰大量是冷门的
        # （艾薩克·牛頓 pop=60，远低于同名城市 135）—— 那正是要修的场景。
        for p in cold_pos:
            cands[p]["score"] = 0.5 * (
                (1.0 - _CTX_W) * cands[p]["prior"] + _CTX_W * cands[p]["ctx"]
            ) + _TYPE_W * cands[p]["type_s"]

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
