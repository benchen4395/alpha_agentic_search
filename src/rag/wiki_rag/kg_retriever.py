"""KGRetriever：KG 端到端检索组件。

将"query → mention 抽取 → 实体链接消歧 → (可选) 多跳图遍历 → 拼装 context"
的整条 pipeline 封装成检索器，接口与 :class:`wiki_rag.retriever.WikiRetriever`
对齐，可以直接接入 :func:`wiki_rag.hybrid.hybrid_retrieve` 或作为独立组件被
Agent / RAG 系统调用。

设计说明
--------
- **对外接口**：``retrieve(query) -> List[Dict]``，字段 (source/title/text/score/qid/mention)
  与 hybrid 融合的其他召回源保持一致。
- **端到端流水线**：由 :func:`query_kg_end_to_end` 承担；内部逐步计时便于观测。
- **Mention 抽取**：三种策略 ("ngram" / "jieba" / "hybrid")，见 `_extract_mentions_*`。
- **多跳**：默认关闭（每跳增加 10-数百 ms 延迟），由 ``multi_hop=True`` 开启。

TODO: 后续可以包装成 Agent Tool（Function Calling / MCP）让 LLM 决定何时调用。
      推荐做法：
        1) 在 Agent 系统里注册 `kg_search_tool(query, multi_hop=False) -> str`
        2) tool schema 描述："在中文百科知识图谱里查找实体的属性、关系和事实"
        3) 让 LLM 判断简单事实 → multi_hop=False；组合关系 → multi_hop=True
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set

from .kg_store import KGStore
from .linker import Linker


# =============================================================================
# Step A: Mention 抽取
# =============================================================================
#
# 方案对比（选型参考）
# ---------------------------------------------------------------------------
# 方案 1（ngram）：从长到短枚举 query 的所有子串，逐个去 mentions 表精确查
#   优点：零依赖、快（1-5 ms）、召回高
#   缺点：会误抽（如"是谁"若在 mention 表里就会命中）
#
# 方案 2（jieba）：jieba.posseg 切词 → 只保留名词类词性 → mentions 表验证
#   优点：语义边界清晰、噪声低
#   缺点：需装 jieba（词典 ~60 MB）；对不在词典里的新词/专名可能切散
#
# 方案 3（hybrid）：先 ngram 广召回，再用 jieba 词性给"打个分"做去噪
#   - 命中 ngram 且是 jieba 判定为名词的 → 保留（高置信）
#   - 命中 ngram 但被 jieba 切碎 / 判为非名词的 → 降权或过滤
#   - jieba 不可用时自动降级为纯 ngram
#   优点：兼顾召回和精度，Agent 场景推荐
#
# TODO: 未来可以再加"方案 4：LLM-based NER"降级方案：
#       - 前三种召回都为空 or query 特别复杂时才调 LLM，避免高延迟
# ---------------------------------------------------------------------------

def _extract_mentions_ngram(query: str,
                            kg: KGStore,
                            max_span: int = 8) -> List[str]:
    """方案 1：N-gram 暴力匹配 mentions 表。

    从长 span 到短 span 遍历，命中后把该字符范围标记 covered，
    避免"苹果公司"命中后又被"苹果"重复抽出。
    """
    mentions: List[str] = []
    covered: Set[int] = set()
    n = len(query)
    for span_len in range(min(max_span, n), 1, -1):
        for start in range(n - span_len + 1):
            end = start + span_len
            # 已被更长 span 覆盖，跳过
            if any(i in covered for i in range(start, end)):
                continue
            span = query[start:end]
            row = kg.conn.execute(
                "SELECT 1 FROM mentions WHERE mention=? LIMIT 1", (span,)
            ).fetchone()
            if row:
                mentions.append(span)
                covered.update(range(start, end))
    return mentions


def _extract_mentions_jieba(query: str, kg: KGStore) -> List[str]:
    """方案 2：jieba 分词 + 词性过滤 + mentions 表验证。"""
    try:
        import jieba.posseg as pseg  # type: ignore
    except ImportError:
        print("[kg-retriever] jieba not installed, fallback to n-gram.")
        return _extract_mentions_ngram(query, kg)

    keep_flags = {"n", "nr", "ns", "nt", "nz", "eng"}
    words = [w.word for w in pseg.cut(query)
             if (w.flag[:2] in keep_flags or w.flag in keep_flags)
             and len(w.word) >= 2]

    seen: Set[str] = set()
    result: List[str] = []
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        row = kg.conn.execute(
            "SELECT 1 FROM mentions WHERE mention=? LIMIT 1", (w,)
        ).fetchone()
        if row:
            result.append(w)
    return result


def _extract_mentions_hybrid(query: str, kg: KGStore) -> List[str]:
    """方案 3：ngram 广召回 + jieba 词性做去噪。

    策略：
      1) 用 ngram 抽出所有可能 mention（高召回、可能有噪声）
      2) 用 jieba 切出所有"名词类词性"的词，作为"可信集合"
      3) 若 mention 在可信集合里出现（或作为其中某个词的前后缀）→ 保留
         若 mention 完全不在可信集合中，但长度 >= 3 → 保留（长 span 通常可信）
         其它 → 过滤
      4) jieba 不可用时降级为纯 ngram（原样返回）
    """
    ngram_hits = _extract_mentions_ngram(query, kg)
    if not ngram_hits:
        return []

    try:
        import jieba.posseg as pseg  # type: ignore
    except ImportError:
        # 没装 jieba，直接返回 ngram 结果（等价于方案 1）
        return ngram_hits

    keep_flags = {"n", "nr", "ns", "nt", "nz", "eng"}
    trusted: Set[str] = set()
    for w in pseg.cut(query):
        if (w.flag[:2] in keep_flags or w.flag in keep_flags) and len(w.word) >= 1:
            trusted.add(w.word)

    filtered: List[str] = []
    for m in ngram_hits:
        # 命中 1：mention 本身就是 jieba 切出来的名词
        if m in trusted:
            filtered.append(m)
            continue
        # 命中 2：mention 是可信词的超串（如 mention="苹果公司"，可信=["苹果","公司"]）
        if any(t in m for t in trusted if len(t) >= 2):
            filtered.append(m)
            continue
        # 命中 3：长 span (>=3 字) 通常是真实实体（如 "量子纠缠"），保留
        if len(m) >= 3:
            filtered.append(m)
            continue
        # 其它：短 mention 且 jieba 不认，很可能是误抽（如"是谁""哪里"），过滤
    return filtered or ngram_hits   # 完全被过滤空 → 回退到原始 ngram，避免误伤


# =============================================================================
# Step A-2: 泛实体（类实体）识别与重排
# =============================================================================
#
# 【问题】混淆式多跳 query 上 L5 贡献为 0（实测 0/60）
# ---------------------------------------------------------------------------
# 实测 BrowseComp-ZH 那道题：
#   "哪个地方拥有AAAAA级景区、被称为成熟周期很长的水果之乡，并且有一位科学家
#    曾在欧洲知名大学学习后回国奠定了一个学科基础？"
#   hybrid 抽出 7 个 mention：
#       ['AAA', '科学家', '地方', '景区', '水果', '欧洲', '学科']
#   其中只有 '欧洲' 是真实体，其余 6 个都是**类/泛实体**。
#
# 【真正的杀手：预算挤占，而不是排序噪声】
# `query_kg_end_to_end` 按 mention 顺序消耗 `max_entities`(默认5) 额度：
#       AAA(+2) → 科学家(+2) → 地方(+1) → 预算耗尽，硬 break
#   于是 ['景区','水果','欧洲','学科'] **完全没有机会被链接** ——
#   唯一的真实体 '欧洲' 被泛实体挤掉了。这解释了为什么"瓶颈在 mention 抽取
#   而非排序"：不是排序把真实体排后了，是它根本没进候选。
#
# 【判别信号选型（全部实测过）】
#   ✗ 候选实体数（歧义度）：爱因斯坦=4 vs 城市=10，真实体并不更少 → 无效
#   ✗ jieba 词性：城市→ns(专名)、苹果公司→n(普通词) → 不可靠
#   ✗ 自身是否有 P279：量子纠缠/相对论 自身都有 P279（概念天然是某类子类）
#                      → 会误杀真实体，这条一定不能用
#   ✓ **P31/P279 入度**：有多少实体声明"我是它的实例/子类"。
#       泛实体 城市=5853、公司=2391、地方=1030、工作=1384
#       真实体 爱因斯坦/北京/苹果公司/相对论/量子纠缠 全部 = 0
#     这是数据驱动、语言无关的信号，不需要人工维护停用词表。
#
# 【延迟】朴素 COUNT(*) 在 城市 上要 349ms（扫 5853 行），在线不可接受。
#   改用**封顶计数**（LIMIT 子查询）：只需知道"是否超过阈值"，不需要精确值。
#   实测 349ms → 0.16ms（**2209x**），整体 3.9ms/词。
#
# 【策略：重排优先于过滤】
#   默认只做**稳定重排**（真实体提前、泛实体后置），不丢弃任何 mention。
#   这样即使判别器误判，最坏结果也只是顺序变化，信息不丢失 ——
#   而顺序恰好就是预算分配顺序，所以重排已经能解决挤占问题。
#   `drop_generic=True` 时才真正过滤，且**全被判为泛实体时自动回退**，
#   避免"一个 mention 都不剩"的退化。
# ---------------------------------------------------------------------------

# 判为"被当作类使用"的入度阈值。取值依据（实测分布）：
#   真实体这两个入度几乎恒为 0，泛实体动辄上千 —— 阈值落在 4~8 这个区间
#   有很大的安全边际，不是需要精调的敏感参数。
_CLASS_IN31_TH: int = 8      # 多少实体声明 "instance of 它"
_CLASS_IN279_TH: int = 4     # 多少实体声明 "subclass of 它"

# 封顶计数上限：只要 >= 阈值就够，不必数到 5853。
# 取 max(阈值) 即可 —— 判定只需知道"是否达到阈值"，多数一行都是浪费。
_INDEG_CAP: int = max(_CLASS_IN31_TH, _CLASS_IN279_TH)

# 每个 mention 最多检查几个候选实体。
# 为什么不能只看 top-1（按 popularity）：实测 "大学" 的 top-1 是
# Q1069886「著作」（一本书），入度=0，于是漏判。多看几个候选才稳。
# 但也不能太多：每个候选要 2 次索引查询，实测 5 个候选时单 query 达 660ms。
_MAX_CAND_PROBE: int = 3

# ── 进程内判定缓存 ──────────────────────────────────────────────────────
# 泛实体判定是**纯函数**（同一 mention + 同一份 KG 永远同结果），且
# 泛实体天然高频复现（"城市/国家/公司"几乎每条 query 都出现）。
# 实测未加缓存时单条混淆式 query 要 660ms（7 个 mention × 3 候选 × 2 PID
# = 42 次索引查询），加缓存后重复 query 降到 ~0ms。
# 用 dict 而非 lru_cache：KGStore 不可 hash，且这里要按 mention 而非
# (mention, kg) 缓存 —— 单进程内 KG 实例固定，不存在串库风险。
_GENERIC_CACHE: dict[str, bool] = {}


def _indegree_capped(kg: KGStore, qid: str, pid: str,
                     cap: int = _INDEG_CAP) -> int:
    """统计"有多少实体通过 pid 指向 qid"，命中 cap 行即停。

    用 `LIMIT` 子查询而非裸 COUNT(*)：我们只关心"是否超过阈值"，
    数到 cap 就足够判定，不必扫完全部 5853 行（349ms → 0.16ms）。
    """
    try:
        row = kg.conn.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM triples "
            "WHERE predicate_pid=? AND object_qid=? LIMIT ?)",
            (pid, qid, cap),
        ).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        # KG 结构异常/表缺失时不能让整条检索挂掉：返回 0 视为"非类实体"，
        # 等价于退化到未启用本特性的行为。
        return 0


def is_generic_mention(mention: str, kg: KGStore) -> bool:
    """判断 mention 是否指向一个**类/泛实体**（而非具体实例）。

    原理：Wikidata 里"类"会被大量实体通过 P31(instance of) /
    P279(subclass of) 指向；具体实例的这两个入度几乎恒为 0。

    Args:
        mention: 字面串，如 "城市" / "爱因斯坦"
        kg:      KGStore

    Returns:
        True 表示是泛实体（建议后置或过滤）。

    示例（实测）：
        >>> is_generic_mention("城市", kg)      # True  （P31 入度 5853）
        >>> is_generic_mention("爱因斯坦", kg)   # False （入度 0）
        >>> is_generic_mention("量子纠缠", kg)   # False （概念但非类）
    """
    cached = _GENERIC_CACHE.get(mention)
    if cached is not None:
        return cached

    try:
        qids = [r[0] for r in kg.conn.execute(
            "SELECT m.qid FROM mentions m JOIN entities e ON e.qid=m.qid "
            "WHERE m.mention=? ORDER BY e.popularity DESC LIMIT ?",
            (mention, _MAX_CAND_PROBE),
        )]
    except Exception:
        return False

    result = False
    for q in qids:
        if _indegree_capped(kg, q, "P31") >= _CLASS_IN31_TH:
            result = True
            break
        if _indegree_capped(kg, q, "P279") >= _CLASS_IN279_TH:
            result = True
            break

    _GENERIC_CACHE[mention] = result
    return result


def rerank_mentions_by_specificity(
    mentions: List[str],
    kg: KGStore,
    *,
    drop_generic: bool = False,
) -> List[str]:
    """把具体实体排到泛实体之前（可选丢弃泛实体）。

    为什么重排就能解决问题：`query_kg_end_to_end` 是**按 mention 顺序**
    消耗 `max_entities` 预算的，顺序即优先级。把真实体提前，它们就能在
    预算耗尽前被链接。

    Args:
        mentions:     `extract_mentions` 的输出（保序）
        kg:           KGStore
        drop_generic: True 则真正丢弃泛实体；默认 False 只重排（更安全）

    Returns:
        重排后的 mention 列表。组内保持原有相对顺序（稳定排序）。
    """
    if not mentions:
        return mentions

    specific: List[str] = []
    generic: List[str] = []
    for m in mentions:
        (generic if is_generic_mention(m, kg) else specific).append(m)

    if drop_generic:
        # 全被判为泛实体时回退到原列表：宁可带噪声，也不能一个都不剩
        # （典型场景："城市人口最多的国家" 这类整句都是类词的 query）
        return specific or mentions

    return specific + generic


def extract_mentions(query: str,
                     kg: KGStore,
                     method: str = "hybrid",
                     *,
                     rerank_specificity: bool = True,
                     drop_generic: bool = False) -> List[str]:
    """统一的 mention 抽取入口。

    Args:
        query:  自然语言 query
        kg:     KGStore
        method: "ngram" | "jieba" | "hybrid"
        rerank_specificity: 是否把具体实体排到泛实体之前（默认 True）。
            解决泛实体挤占 `max_entities` 预算的问题，实测 +3.9ms/词。
        drop_generic: 是否直接丢弃泛实体（默认 False，只重排不丢）。

    Returns:
        mention 列表（保序、已去重、已用 mentions 表校验存在）
    """
    if method == "jieba":
        mentions = _extract_mentions_jieba(query, kg)
    elif method == "hybrid":
        mentions = _extract_mentions_hybrid(query, kg)
    else:
        mentions = _extract_mentions_ngram(query, kg)

    if rerank_specificity and mentions:
        mentions = rerank_mentions_by_specificity(
            mentions, kg, drop_generic=drop_generic)
    return mentions


# =============================================================================
# 多跳图遍历
# =============================================================================
#
# 成本说明：
#   - 1 跳：< 5 ms
#   - 2 跳（种子 S=5, 每种子扩 E=3）：10-50 ms
#   - 3 跳（不限扩展）：100-500 ms（指数膨胀）
#
# 因此默认关闭 (multi_hop=False)，只在组合关系问题（"库克领导的公司在哪个城市"）
# 时按需打开。
# =============================================================================

def traverse_multi_hop(seed_qids: List[str],
                       kg: KGStore,
                       *,
                       max_hops: int = 2,
                       max_expand_per_seed: int = 3,
                       visited: Optional[Set[str]] = None) -> List[Dict]:
    """从种子实体出发做 BFS 图遍历。

    Args:
        seed_qids:              起始实体 QID 列表（一般来自 linker.link）
        kg:                     KGStore
        max_hops:               最大跳数（1 = 只种子；2 = 种子 + 1 跳邻居…）
        max_expand_per_seed:    每个种子最多扩展多少个 object 实体（防指数膨胀）
        visited:                外部传入的已访问集合（避免和主链路重复）

    Returns:
        新发现实体列表，元素结构和 linker.link_and_expand 对齐 + 溯源字段。
    """
    if visited is None:
        visited = set()

    queue = deque((q, 1) for q in seed_qids)
    for q in seed_qids:
        visited.add(q)

    new_entities: List[Dict] = []

    while queue:
        qid, hop = queue.popleft()
        if hop >= max_hops:
            continue
        triples = kg.triples_of(qid, limit=max_expand_per_seed * 3)
        expanded = 0
        for t in triples:
            obj_qid = t.get("object_qid")
            if not obj_qid or t.get("object_type") != "entity":
                continue
            if obj_qid in visited:
                continue
            visited.add(obj_qid)
            row = kg.conn.execute(
                "SELECT qid, label_zh, description FROM entities WHERE qid=?",
                (obj_qid,)
            ).fetchone()
            if row is None:
                continue
            new_entities.append({
                "qid":         row["qid"],
                "label_zh":    row["label_zh"],
                "description": row["description"],
                "score":       0.0,       # 多跳发现的实体不参与打分
                "source":      f"hop{hop+1}",
                "context":     kg.to_context(row["qid"]),
                "via":         qid,       # 从哪个种子跳来
                "predicate":   t.get("predicate_label") or t.get("predicate_pid"),
            })
            expanded += 1
            queue.append((obj_qid, hop + 1))
            if expanded >= max_expand_per_seed:
                break
    return new_entities


# =============================================================================
# 端到端 pipeline
# =============================================================================

def query_kg_end_to_end(query: str,
                        linker: Linker,
                        kg: KGStore,
                        *,
                        method: str = "hybrid",
                        top_k_per_mention: int = 2,
                        max_entities: int = 5,
                        multi_hop: bool = False,
                        max_hops: int = 2,
                        max_hop_expand: int = 3) -> Dict:
    """完整 KG 检索流水线。

    query → mention 抽取 → 实体链接消歧 → (可选) 多跳 → 拼 LLM-ready context

    Returns:
        {
            "query":        原始 query,
            "mentions":     抽出的 mention,
            "entities":     [{qid, label_zh, description, score, source,
                             context, mention, via?, predicate?}, ...],
            "context_text": 拼好的整段 context 文本,
            "timing":       每阶段耗时（秒），便于观测优化,
        }
    """
    timing: Dict[str, float] = {}

    # ---- Step A: mention 抽取 ----
    t0 = time.perf_counter()
    mentions = extract_mentions(query, kg, method=method)
    timing["extract_mentions"] = time.perf_counter() - t0
    if not mentions:
        return {"query": query, "mentions": [], "entities": [],
                "context_text": "", "timing": timing}

    # ---- Step B: 每个 mention 做实体链接 + 消歧 ----
    t0 = time.perf_counter()
    all_entities: List[Dict] = []
    seen_qids: Set[str] = set()
    for m in mentions:
        try:
            results = linker.link_and_expand(m, query_context=query,
                                             top_k=top_k_per_mention)
        except Exception as e:
            print(f"[kg-pipeline] linker failed on mention={m!r}: {e}")
            continue
        for r in results:
            if r["qid"] in seen_qids:
                continue
            seen_qids.add(r["qid"])
            all_entities.append({**r, "mention": m})
            if len(all_entities) >= max_entities:
                break
        if len(all_entities) >= max_entities:
            break
    timing["link_and_expand"] = time.perf_counter() - t0

    # ---- Step C: 多跳（可选）----
    if multi_hop and max_hops > 1 and all_entities:
        t0 = time.perf_counter()
        seed_qids = [e["qid"] for e in all_entities]
        hop_entities = traverse_multi_hop(
            seed_qids, kg,
            max_hops=max_hops,
            max_expand_per_seed=max_hop_expand,
            visited=seen_qids,
        )
        for h in hop_entities:
            all_entities.append(h)
        timing["multi_hop"] = time.perf_counter() - t0

    # ---- Step D: 拼装 context ----
    t0 = time.perf_counter()
    parts: List[str] = []
    for ent in all_entities:
        header = f"【{ent.get('label_zh') or ent['qid']}（{ent['qid']}）】"
        if ent.get("description"):
            header += f" — {ent['description']}"
        body = ent.get("context") or ""
        parts.append(header + ("\n" + body if body else ""))
    context_text = "\n\n---\n\n".join(parts)
    timing["assemble_context"] = time.perf_counter() - t0

    return {
        "query": query,
        "mentions": mentions,
        "entities": all_entities,
        "context_text": context_text,
        "timing": timing,
    }


# =============================================================================
# KGRetriever：仿 WikiRetriever 的检索器组件
# =============================================================================

class KGRetriever:
    """将 KG 端到端流水线包装成"检索器"，接口和 :class:`WikiRetriever` 对齐。

    典型用法::

        # 独立使用
        kg_retr = KGRetriever(config_path="configs/default.yaml")
        docs = kg_retr.retrieve("苹果公司的 CEO 是谁", top_k=3)
        # docs: [{source:'kg', title, text, score, qid, mention, ...}, ...]

        # 或者通过 hybrid_retrieve 接入（推荐）
        from wiki_rag.hybrid import hybrid_retrieve
        results = hybrid_retrieve(query, wiki_retriever=..., kg_retriever=kg_retr,
                                  multi_hop=True)

    TODO: 后续可以包装成 Agent Tool（Function Calling / MCP）让 LLM 决定何时调用。
    """

    def __init__(self,
                 config_path: str | Path | None = None,
                 *,
                 kg: Optional[KGStore] = None,
                 linker: Optional[Linker] = None,
                 mention_method: str = "hybrid"):
        """允许外部注入 KGStore / Linker，避免重复打开数据库。

        Args:
            config_path:      配置文件路径；传 None 使用 vendored 默认 YAML
                              （rag/configs/default.yaml，路径已锚定项目根）。
            kg:               外部已有的 KGStore（可选）
            linker:           外部已有的 Linker（可选）
            mention_method:   "ngram" | "jieba" | "hybrid"（默认 hybrid）
        """
        # 注意：不要对 None 做 str() —— 会变成字符串 "None" 导致找不到配置。
        self.config_path = str(config_path) if config_path is not None else None
        self.kg = kg or KGStore(self.config_path)
        self.linker = linker or Linker(kg_store=self.kg,
                                       config_path=self.config_path)
        self.mention_method = mention_method        # hybird

    def retrieve(self,
                 query: str,
                 *,
                 top_k: int = 3,
                 multi_hop: bool = False,
                 max_hops: int = 2,
                 max_hop_expand: int = 3) -> List[Dict]:
        """标准检索接口。

        Returns:
            List[Dict]，元素格式与 WikiRetriever.search() 对齐：
              {source='kg', title, text, score, qid, mention,
               via?, predicate?}
        """
        result = query_kg_end_to_end(
            query,
            linker=self.linker,
            kg=self.kg,
            method=self.mention_method,
            top_k_per_mention=2,
            max_entities=top_k,
            multi_hop=multi_hop,
            max_hops=max_hops,
            max_hop_expand=max_hop_expand,
        )
        docs: List[Dict] = []
        for ent in result["entities"]:
            docs.append({
                "source":    "kg",
                "title":     ent.get("label_zh") or ent["qid"],
                "text":      ent.get("context") or "",
                "score":     float(ent.get("score", 0.0)),
                "qid":       ent["qid"],
                "mention":   ent.get("mention"),
                "via":       ent.get("via"),
                "predicate": ent.get("predicate"),
            })
        return docs

    def retrieve_as_text(self, query: str, **kwargs) -> str:
        """一站式返回拼好的整段 context 文本，可直接塞进 LLM prompt。"""
        result = query_kg_end_to_end(
            query, linker=self.linker, kg=self.kg,
            method=self.mention_method, **kwargs
        )
        return result["context_text"]
