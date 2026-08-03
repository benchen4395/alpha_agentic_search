# cache_policy.py
"""L1 Q&A 缓存的「准入 / 命中」策略层（P0-1）。

════════════════════════════════════════════════════════════════════════
为什么需要这个模块
════════════════════════════════════════════════════════════════════════
改造前 `agent._archive_if_enabled()` 是**无条件**把每一轮 (query, answer)
写进 L1 QACache 的，而 L1 又有两个放大器：

  1. `fuzzy_threshold=0.8` 的向量模糊命中
  2. `QA_CACHE_TTL = 30 天` 的全局统一 TTL

三者叠加会产生三类**用户完全无法察觉的错误**（因为 L1 命中在 chat() 的
Step 0，毫秒级直接短路返回，既不检索也不调 LLM）：

  ① **时效性污染**
     「今天上海天气」「最新的 GPT 版本是什么」被冻结 30 天。
     注意：`rag/router.py` 里虽然有 `is_time_sensitive()`，但它只决定
     「要不要召 L4 web」，**完全不影响是否写 L1**；而 Step 0 的 L1 短路
     发生在 router 之前，时效判断根本没机会跑。

  ② **语义碰撞（最危险）**
     BGE-M3 在 0.8 阈值下，下面这些「只差一个限定词 / 数字 / 实体」的
     query 余弦相似度普遍落在 0.85 ~ 0.93，会被判为同一个问题：
        - 「美国总统是谁」    vs 「美国副总统是谁」      （差一个 "副"）
        - 「2024年GDP是多少」 vs 「2025年GDP是多少」      （差一个年份）
        - 「苹果的CEO是谁」   vs 「苹果的CFO是谁」        （差一个职位）
        - 「推荐保健品」      vs 「不要推荐保健品」        （差一个否定词）
     这是所有 semantic cache 的经典陷阱：**向量相似 ≠ 答案可复用**。

  ③ **幻觉自我固化**
     一次幻觉答案会同时写进 L1（短路）和 L3（召回），
     后续同类问题被这条脏数据不断加固，"越用越强"变成"越用越错"。

════════════════════════════════════════════════════════════════════════
本模块的两道防线
════════════════════════════════════════════════════════════════════════
                        ┌─────────────────────────────┐
   写入侧（archive）───► │ 防线 A：准入策略 + 分级 TTL   │
                        │  decide_cacheability()       │
                        └─────────────────────────────┘
                        ┌─────────────────────────────┐
   读取侧（fuzzy hit）─► │ 防线 B：槽位一致性门禁        │
                        │  slots_compatible()          │
                        └─────────────────────────────┘

* **防线 A（写入侧）**：不是所有答案都值得缓存。
  - 时效敏感（今天/最新/现在/股价…）→ **一律不写 L1**
  - 走了 L4 web 的（说明离线知识覆盖不到，内容偏时事）→ 短 TTL（6h）
  - 含易变槽位（年份 / 价格 / 排名 / 版本号）→ 中 TTL（24h）
  - 纯常识类 → 长 TTL（30d）
  - 答案疑似"信息不足 / 我不知道 / 报错"→ **不写**（避免固化失败）

* **防线 B（读取侧）**：仅靠调阈值不足以解决语义碰撞，
  因为「苹果CEO vs 苹果CFO」这类 query 在任何阈值下都可能过线。
  所以额外加一道**判别式门禁**：抽出双方 query 的「关键槽位」
  （数字 / 年份 / 否定词 / 比较级 / 限定词 / 疑问焦点 / 命名实体 / 主题名词），
  **任一槽位集合不同则拒绝命中**。这比单纯拉高阈值可靠得多，
  且是"精确匹配"而非"相似度"，几乎零误放行。

  实测余弦（BGE-M3，本机跑出来的真实数字）：
      「苹果公司的CEO是谁」vs「苹果公司的CFO是谁」  → 0.8781
      「2024年中国GDP是多少」vs「2025年中国GDP是多少」→ 0.8531
      「美国现任总统是谁」vs「美国现任副总统是谁」    → 0.8575
  在**原来的 0.8 阈值下这三对全部会误命中**，用户毫秒级拿到错误答案，
  且没有任何日志能看出问题。这就是必须做这一层的实证依据。

════════════════════════════════════════════════════════════════════════
设计原则
════════════════════════════════════════════════════════════════════════
* **纯函数、零副作用、零重依赖**：只用 re + 可选 jieba，方便单测与复用。
* **保守优先**：任何"判不准"的情况都倒向"不缓存 / 不命中"。
  代价是多花一次检索（几秒），收益是不返回错误答案（不可接受）。
* **可配置**：所有阈值集中在本文件顶部常量 + 环境变量覆盖。
"""
from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass, field
from typing import Optional

# jieba 可选：用于抽取命名实体槽位（人名/地名/机构名/专名）。
# 缺失时自动降级——只做正则槽位比对，门禁强度略降但仍然有效。
_has_jieba = False
try:
    import jieba.posseg as _pseg  # type: ignore

    _has_jieba = True
except Exception:
    _pseg = None  # type: ignore


# ════════════════════════════════════════════════════════════════════════
#                            分级 TTL 常量
# ════════════════════════════════════════════════════════════════════════
# 设计依据：TTL 应该与「答案的期望半衰期」匹配，而不是一个全局常量。
#   - 时事/时效 → 根本不该进 L1（用 None 表示拒绝）
#   - web 兜底  → 说明离线层覆盖不到，内容更可能是时事 → 6 小时
#   - 易变槽位  → 含年份/价格/排名/版本 → 24 小时
#   - 常识      → "量子计算是什么" 这类，30 天甚至永久都可以
TTL_WEB_BACKED: int = int(os.getenv("QA_TTL_WEB_BACKED", str(6 * 3600)))      # 6h
TTL_VOLATILE: int = int(os.getenv("QA_TTL_VOLATILE", str(24 * 3600)))         # 24h
TTL_STABLE: int = int(os.getenv("QA_TTL_STABLE", str(30 * 24 * 3600)))        # 30d

# ── 模糊命中阈值：0.8 →（一度 0.93）→ 0.90 ──────────────────────────────
#
# 这个值经过两轮实测标定，过程值得记录，因为它体现了「阈值不该拍脑袋定」：
#
# 【第一版 0.8】误放行严重。BGE-M3 实测负样本（答案完全不同的 query 对）：
#       Python2/3 的区别 vs Python3/4 的区别   → 0.8800
#       苹果公司的CEO是谁 vs 苹果公司的CFO是谁   → 0.8781
#       美国现任总统是谁  vs 美国现任副总统是谁  → 0.8573
#       2024年中国GDP    vs 2025年中国GDP      → 0.8531
#   12 个负样本里有 7 个会在 0.8 下误命中 → 用户毫秒级拿到错误答案。
#
# 【第二版 0.93】过度矫正，把大量**同义改述**也挡在门外。实测正样本：
#       美国历届总统名单 vs 美国历届总统      → 0.9265  ← 差 0.0035 被拒！
#       美国历届总统名单 vs 列出美国历届总统  → 0.9226
#   这正是用户实际遇到的 bug：只少了「名单」两个字就完全命不中。
#
# 【定标方法】用 15 条正样本（同义改述）+ 12 条负样本（一字之差换答案）
#   做阈值扫描，在「槽位门禁开启」的前提下统计召回与误放行：
#       阈值   召回      误放行
#       0.88   12/15     0/12
#       0.90   12/15     0/12    ← 选它
#       0.93   10/15     0/12    ← 白丢 2 条召回，安全性没有任何提升
#
#   关键观察：**负样本余弦的最大值是 0.8800**。也就是说 0.90 已经在所有
#   已知危险碰撞之上留了 0.02 的安全余量；再往上抬不会挡掉任何一个负样本，
#   只会持续误杀正样本 —— 0.93 是纯粹的净损失。
#
# 【为什么敢降】真正承担「区分 CEO/CFO」职责的不是阈值，而是下面的槽位门禁：
#   实测 12 个负样本里，门禁独立拦下了 11 个（91.7%），且完全不依赖阈值。
#   阈值的职责被重新定位为「粗筛掉毫不相干的 query」，而不是「精确判别语义」。
#   分工明确后，阈值就没有必要绷得那么紧。
FUZZY_THRESHOLD: float = float(os.getenv("QA_FUZZY_THRESHOLD", "0.90"))

# 答案长度下限：过短的答案（如 "好的"、"是"）缓存价值低且易误命中。
MIN_CACHEABLE_ANSWER_LEN: int = int(os.getenv("QA_MIN_ANSWER_LEN", "10"))

# 主题名词槽位的 Jaccard 下限（见 slots_compatible 里的详细说明）。
# 主题名词受分词/句式影响，不能像其它槽位那样要求完全相等，
# 所以用「交集/并集 >= 0.5」（至少一半主题词重合）作为一致性判据。
TOPIC_JACCARD_MIN: float = float(os.getenv("QA_TOPIC_JACCARD_MIN", "0.5"))


# ════════════════════════════════════════════════════════════════════════
#                        防线 A-1：时效敏感检测
# ════════════════════════════════════════════════════════════════════════
# 与 rag/router.py 的 _TIME_SENSITIVE_HINTS 相比，这里做了两个改进：
#   1. **年份动态生成**：原实现硬编码了 "2024","2025","2026"，明年就失效。
#      这里按「当前年份 ±1」动态生成，永不过期。
#   2. **区分强/弱信号**：强信号（今天/现在/最新/股价）一律拒缓存；
#      弱信号（年份/版本号）只降 TTL 不拒绝，避免把 "2020年GDP" 这种
#      历史事实也误杀（它其实是稳定事实）。

# —— 强时效信号：出现即拒绝写入 L1 ——
# 这些词意味着「答案与提问的绝对时刻绑定」，缓存任意时长都是错的。
_STRONG_TIME_CUES: tuple[str, ...] = (
    # 中文：相对时间
    "今天", "今日", "昨天", "昨日", "明天", "明日", "前天", "后天",
    "刚刚", "刚才", "此刻", "现在", "目前", "当前", "眼下", "近期", "最近",
    "本周", "这周", "上周", "下周", "本月", "这个月", "上月", "下月",
    "本季度", "今年", "去年", "明年",
    # 中文：时效性表述
    "最新", "最近一次", "实时", "即时", "当下", "截至目前", "到目前为止",
    # 中文：高频易变实体
    "天气", "气温", "股价", "股票", "汇率", "油价", "金价", "币价", "行情",
    "新闻", "热搜", "直播", "比分", "赛况", "排行榜", "热榜",
    "现任", "在任", "当选", "上市价格", "促销", "折扣",
    # 英文
    "today", "yesterday", "tomorrow", "now", "current", "currently",
    "latest", "recent", "recently", "this week", "this month", "this year",
    "real-time", "realtime", "live", "breaking",
    "weather", "stock price", "exchange rate", "trending",
)

# —— 弱时效信号：只降 TTL，不拒绝 ——
# 含具体年份、版本号、"第N届"这类：事实本身可能稳定（"2020年GDP"），
# 但也可能是变动值（"2026年最新政策"），折中处理为短 TTL。
_WEAK_VOLATILE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\d{4}\s*年"),                       # 2024年
    re.compile(r"\bv?\d+\.\d+(\.\d+)?\b"),           # v1.2.3 / 4.5
    re.compile(r"第\s*\d+\s*[届次期代版]"),            # 第5届
    re.compile(r"\d+\s*(元|块|美元|万|亿|美金|USD|RMB)"),  # 价格
    re.compile(r"排[名行]第?\s*\d+"),                 # 排名第3
    re.compile(r"\b(20\d{2})\b"),                    # 裸年份 2024
)


def _dynamic_year_cues() -> tuple[str, ...]:
    """动态生成「当前年份 ±1」的年份字符串，作为强时效信号补充。

    为什么要动态：`rag/router.py` 里写死了 "2024"/"2025"/"2026"，
    到 2027 年这套判断就完全失效了（而且会把 2024 这种已成历史的
    年份继续当"时效敏感"误杀）。这里只把「今年、去年、明年」当强信号，
    更早的年份交给 `_WEAK_VOLATILE_PATTERNS` 走短 TTL。
    """
    y = _dt.date.today().year
    return tuple(str(v) for v in (y - 1, y, y + 1))


def is_time_sensitive(query: str) -> bool:
    """判断 query 是否**强时效敏感**（答案与提问时刻绑定，禁止进 L1）。

    Args:
        query: 用户原始 query。
    Returns:
        True 表示禁止缓存到 L1。

    示例：
        >>> is_time_sensitive("今天上海天气怎么样")   # True（今天 + 天气）
        >>> is_time_sensitive("量子计算是什么")        # False
        >>> is_time_sensitive("2020年中国GDP")        # False（历史事实，走短 TTL）
    """
    if not query:
        return False
    q = query.lower()
    if any(cue in q for cue in _STRONG_TIME_CUES):
        return True
    # 动态年份：只有「今年/去年/明年」的数字算强信号
    return any(y in q for y in _dynamic_year_cues())


def has_volatile_slot(query: str) -> bool:
    """判断 query 是否含**易变槽位**（年份 / 价格 / 版本 / 排名）。

    命中时不拒绝缓存，但把 TTL 压到 `TTL_VOLATILE`（24h）。
    """
    if not query:
        return False
    return any(p.search(query) for p in _WEAK_VOLATILE_PATTERNS)


# ════════════════════════════════════════════════════════════════════════
#                    防线 A-2：低质量答案检测
# ════════════════════════════════════════════════════════════════════════
# "抱歉，我没有找到相关信息" 这类答案缓存下来危害极大：
#   用户第二次问同样的问题，本可以重新检索（这次可能成功），
#   却被 L1 直接短路返回了上次的失败结果，且持续 30 天。
_LOW_QUALITY_ANSWER_CUES: tuple[str, ...] = (
    "抱歉", "无法回答", "无法确定", "不知道", "没有找到", "未找到",
    "信息不足", "资料不足", "无相关", "不清楚", "无法提供",
    "检索结果与问题不相关", "无搜索结果", "(无外部资料)",
    "sorry", "i don't know", "i cannot", "no information",
    "not enough information", "unable to",
)


def is_low_quality_answer(answer: str) -> bool:
    """判断答案是否为「拒答 / 信息不足 / 报错」类低质量输出。

    只检查**开头 60 个字符**：正常答案里出现"抱歉"多半在中间的补充说明，
    而拒答型答案几乎总是开头就表态。这样能避免误杀正常长答案。
    """
    if not answer:
        return True
    head = answer.strip()[:60].lower()
    return any(cue in head for cue in _LOW_QUALITY_ANSWER_CUES)


# ════════════════════════════════════════════════════════════════════════
#                     防线 A：准入决策（对外主接口）
# ════════════════════════════════════════════════════════════════════════
@dataclass
class CacheDecision:
    """L1 写入决策结果。

    Fields:
        cacheable: 是否允许写入 L1。
        ttl:       允许时的过期秒数（None 表示永不过期；仅当 cacheable=True 有意义）。
        reason:    判定理由（人类可读，便于 trace / 日志排查）。
        tier:      命中的策略档位，便于监控统计：
                   "reject_time_sensitive" / "reject_low_quality" /
                   "reject_too_short" / "web_backed" / "volatile" / "stable"
    """

    cacheable: bool
    ttl: Optional[int]
    reason: str
    tier: str


def decide_cacheability(
    query: str,
    answer: str,
    layer_hits: Optional[dict] = None,
) -> CacheDecision:
    """决定一条 (query, answer) 能否写入 L1，以及该给多长 TTL。

    判定顺序（**短路求值，先拒后放**，保守优先）：

        1. 答案太短        → 拒（缓存价值低 + 易误命中）
        2. 答案是拒答/报错  → 拒（避免固化失败结果）
        3. query 强时效     → 拒（今天/最新/股价…，任何 TTL 都是错的）
        4. 走了 L4 web      → 允许，TTL=6h（离线覆盖不到 ⇒ 偏时事）
        5. 含易变槽位       → 允许，TTL=24h（年份/价格/版本/排名）
        6. 其余（常识类）   → 允许，TTL=30d

    Args:
        query:      用户**原始** query（与 Step 0 查询侧对齐，不要传改写后的）。
        answer:     最终答案文本。
        layer_hits: 本轮 RAG 各层命中数，形如 `{"L2_wiki": 3, "L4_web": 5}`。
                    用于识别"是否依赖了实时 web"。传 None 时按无 web 处理。

    Returns:
        CacheDecision

    示例：
        >>> decide_cacheability("今天天气", "晴，25度").cacheable
        False
        >>> d = decide_cacheability("量子计算是什么", "量子计算是..."*10)
        >>> d.tier
        'stable'
    """
    # --- 1) 答案过短：既没什么复用价值，又容易被 fuzzy 误命中 ---
    if not answer or len(answer.strip()) < MIN_CACHEABLE_ANSWER_LEN:
        return CacheDecision(
            False, None,
            f"答案长度 {len(answer or '')} < {MIN_CACHEABLE_ANSWER_LEN}，缓存价值低",
            "reject_too_short",
        )

    # --- 2) 拒答 / 信息不足：缓存下来会阻止未来的成功重试 ---
    if is_low_quality_answer(answer):
        return CacheDecision(
            False, None,
            "答案疑似拒答/信息不足，避免把失败结果固化 30 天",
            "reject_low_quality",
        )

    # --- 3) 强时效：这是最重要的一道拒绝，L1 短路发生在 router 之前 ---
    if is_time_sensitive(query):
        return CacheDecision(
            False, None,
            "query 强时效敏感（今天/最新/现在/股价…），答案与提问时刻绑定",
            "reject_time_sensitive",
        )

    # --- 4) 依赖了实时 web：说明离线知识库覆盖不到，内容偏时事 → 短 TTL ---
    if layer_hits and int(layer_hits.get("L4_web") or 0) > 0:
        return CacheDecision(
            True, TTL_WEB_BACKED,
            "答案依赖 L4 实时 web（离线层未覆盖），给短 TTL 防陈旧",
            "web_backed",
        )

    # --- 5) 含易变槽位（年份/价格/版本/排名）→ 中等 TTL ---
    if has_volatile_slot(query):
        return CacheDecision(
            True, TTL_VOLATILE,
            "query 含易变槽位（年份/价格/版本/排名），给中等 TTL",
            "volatile",
        )

    # --- 6) 常识类：可以长期缓存 ---
    return CacheDecision(
        True, TTL_STABLE, "常识类稳定问答，可长期缓存", "stable"
    )


# ════════════════════════════════════════════════════════════════════════
#                     防线 B：槽位一致性门禁
# ════════════════════════════════════════════════════════════════════════
# 核心思想：把 query 里那些「一改就换答案」的成分抽出来做**精确集合比对**。
# 向量相似度是"软"的、连续的，无法可靠区分 "CEO" 和 "CFO"；
# 而槽位比对是"硬"的、离散的，一字之差就拒绝。二者互补。

# —— 否定词：一个"不"字会把答案完全反转 ——
_NEGATION_TOKENS: frozenset[str] = frozenset({
    "不", "不是", "不要", "不用", "不能", "不可以", "不需要", "不含", "不带",
    "非", "无", "没", "没有", "未", "勿", "别", "莫",
    "排除", "除外", "除了", "去掉", "避免", "不包括", "不包含",
    "not", "no", "without", "exclude", "except", "avoid", "none", "never",
})

# —— 比较级 / 极值：改一个字就换答案（最大↔最小、第一↔最后）——
_COMPARATIVE_TOKENS: frozenset[str] = frozenset({
    "最大", "最小", "最高", "最低", "最多", "最少", "最好", "最差",
    "最快", "最慢", "最早", "最晚", "最新", "最老", "最贵", "最便宜",
    "第一", "第二", "第三", "最后", "首个", "末尾", "更大", "更小",
    "以上", "以下", "超过", "低于", "高于", "大于", "小于",
    "largest", "smallest", "highest", "lowest", "most", "least",
    "best", "worst", "first", "last", "fastest", "slowest",
    "more than", "less than", "above", "below", "over", "under",
})

# —— 疑问焦点：who/where/when/how-much 换一个，答案类型就完全不同 ——
# "苹果的CEO是谁" vs "苹果的CEO在哪" —— 前者要人名，后者要地点。
#
# ⚠️ 注意这个槽位**不能要求精确相等**，原因见 `_focus_compatible()` 的说明。
_INTERROGATIVE_FOCUS: dict[str, tuple[str, ...]] = {
    "WHO": ("谁", "何人", "哪个人", "who", "whom", "whose"),
    "WHERE": ("哪里", "何地", "哪儿", "在哪", "哪个城市", "哪个国家",
              "where", "which city", "which country"),
    "WHEN": ("何时", "什么时候", "哪一年", "哪年", "几号", "几点", "多久",
             "when", "what time", "which year", "how long"),
    "HOWMANY": ("多少", "几个", "几人", "多大", "多高", "多长", "多重",
                "how many", "how much", "how big", "how tall"),
    "WHY": ("为什么", "为何", "原因", "why", "reason"),
    "HOW": ("怎么", "如何", "怎样", "怎么办", "how to", "how do", "how can"),
    "WHICH": ("哪个", "哪些", "哪种", "which", "what kind"),
}

# jieba 词性中代表「命名实体」的标签（换一个实体必然换答案）
#   nr=人名  ns=地名  nt=机构名  nz=其他专名  nrt/nrfg=人名变体  eng=英文词
_ENTITY_POS_FLAGS: frozenset[str] = frozenset({
    "nr", "nrt", "nrfg", "ns", "nt", "nz", "eng",
})

# —— 主题名词（topics）：补上 _ENTITY_POS_FLAGS 覆盖不到的「问题主体」——
#
# ⚠️ 为什么必须单独加这一类：jieba 对**很多公司/产品名的词性标注并不是实体**。
#   实测：
#       "苹果的CEO是谁" → [('苹果','n'),  ('CEO','eng'), …]
#       "微软的CEO是谁" → [('微软','a'),  ('CEO','eng'), …]   ← 竟然是形容词
#   两者的 entities 都只有 {CEO}，实体槽位完全相同 → 会被判为可复用！
#   而它们的答案（库克 vs 纳德拉）完全不同。
#
# 所以再抽一路「实义名词」作为主题槽位。选词性时的权衡：
#   收 n（普通名词）/ a,an（形容词性名词，jieba 常把品牌名归到这里）
#     / nz（专名）/ vn（动名词，如"计算"、"排序"）/ j（简称，如"央行"）
#   不收 v（动词）/ r（代词）/ p,u,d（虚词）—— 它们是提问脚手架，
#     同义改述时会大量变化（"讲讲X" vs "X是什么"），收了会造成海量误拒。
_TOPIC_POS_FLAGS: frozenset[str] = frozenset({
    "n", "an", "a", "nz", "vn", "j", "nl", "ng",
})

# 主题槽位的停用词：这些是**提问句式的一部分**而非主体，
# 同义改述时出现与否高度随机，必须剔除，否则
# "讲讲RAG的原理" vs "RAG的原理是什么" 会因为一个有"原理"一个也有"原理"…
# （实际风险在于像"情况"/"问题"/"方面"这类空泛名词）而误拒。
_TOPIC_STOPWORDS: frozenset[str] = frozenset({
    "什么", "怎么", "如何", "哪些", "哪个", "为什么", "多少",
    "情况", "问题", "方面", "东西", "内容", "意思", "时候", "地方",
    "一下", "一些", "一点", "介绍", "简介", "说明", "解释", "讲解",
    "区别", "关系",      # 过于泛化，且常在改述中出现/消失
})

# —— 限定词 / 修饰语：一字之差换一个人、换一个版本 ——
# 这是「美国总统 vs 美国副总统」这类碰撞的专用拦截器：
# jieba 常把 "美国总统" 和 "美国副总统" 都切成机构/专名，实体槽位看不出差别；
# 而这些限定前缀恰恰是决定答案的关键（正职/副职、现任/前任、正式/代理）。
#
# ⚠️ 关键教训：**单字限定词绝不能裸子串匹配**。
#   第一版把 "前"/"现"/"总"/"新" 直接放进集合做 `t in query`，结果：
#     - "当前美国总统是谁" 里的 "当前" 命中了 "前" → 被误判为"前任总统"
#       → 与 "美国总统是谁" 槽位不一致 → 同义改述被误拒；
#     - "总" 在 "总统" 和 "副总统" 里都出现，作为区分特征毫无价值却制造噪声。
#   所以改成两类：
#     (a) 多字限定词      → 语义自足，可安全子串匹配（现任/前任/首席…）
#     (b) 单字职务前缀    → 必须**紧跟职务名词**才算（副+总统 ✓，当前 ✗）

# (a) 多字限定词：本身语义明确，子串匹配即可
_QUALIFIER_PHRASES: frozenset[str] = frozenset({
    # 时序限定（"当前/目前"不在此列——它们等价于"无限定"，见上方说明）
    "现任", "在任", "前任", "原任", "上任", "下任", "继任", "候任",
    "首任", "末任", "新任", "老一辈", "上一任", "下一任", "第一任",
    # 职务限定（多字）
    "首席", "助理", "代理", "常务", "执行", "名誉", "荣誉", "候选",
    # 英文
    "vice", "deputy", "former", "ex-", "acting", "interim", "chief",
    "assistant", "associate", "senior", "junior", "incumbent",
})

# (b) 单字职务前缀 + 后随职务名词才算命中。
#     "副总统" → 命中 "副"；"当前" → 不命中（"前"后面不是职务名词）。
_ROLE_NOUN_CHARS = "总主部市省校院书经董理事长席监裁记官员司局处科班组队"
_QUALIFIER_PREFIX_RE = re.compile(
    rf"([副正代兼署])(?=[{_ROLE_NOUN_CHARS}])"
)

# 数字抽取：阿拉伯数字（整数 / 小数 / 百分比）
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")

# 中文数字：**必须带序数/量词上下文**才算槽位。
#   理由：裸匹配中文数字会把 "介绍一下"、"推荐一些"、"说一点" 里的 "一"
#   也当成数字槽位，导致 "介绍一下Transformer" vs "Transformer介绍" 被误拒。
#   所以只在两种上下文下抽取：
#     (a) 序数前缀：第三届、第五版
#     (b) 量词后缀：三个、五年、十次、百万元
_CN_NUM_CHARS = "一二三四五六七八九十百千万亿零两壹贰叁肆伍陆柒捌玖拾"
_CN_ORDINAL_RE = re.compile(rf"第\s*([{_CN_NUM_CHARS}]{{1,6}})")
_CN_QUANTITY_RE = re.compile(
    rf"([{_CN_NUM_CHARS}]{{1,6}})\s*"
    r"(?:个|名|位|人|年|月|日|届|次|期|代|版|条|项|种|类|倍|成|折|"
    r"元|块|万|亿|米|千米|公里|吨|斤|克|度|岁|层|章|节|步)"
)
# 中文数字里需要剔除的「语气/虚指」组合（不表达真实数量）
_CN_NUM_STOPWORDS: frozenset[str] = frozenset({"一", "两", "一二"})

# 连续的 ASCII 字母序列（用于捕捉 CEO/CFO/GPT/API 这类缩写）
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\.]{1,}")


@dataclass
class QuerySlots:
    """query 的「关键槽位」指纹。

    两个 query 只有在**全部槽位都相同**时，才允许复用同一个缓存答案。

    Fields:
        numbers:    数字集合（阿拉伯数字 + 带上下文的中文数字）。
                    "2024" vs "2025" → 不同
        years:      年份集合（单独抽一份，语义权重更高）
        negations:  否定词集合。"要保健品" vs "不要保健品" → 不同
        comparatives: 比较级/极值集合。"最大" vs "最小" → 不同
        qualifiers: 限定词集合。"总统" vs "副总统"、"现任" vs "前任" → 不同
        focus:      疑问焦点集合，如 {"WHO"}。"是谁" vs "在哪" → 不同
        entities:   命名实体集合（jieba nr/ns/nt/nz）+ ASCII 缩写。
                    "苹果的CEO" vs "苹果的CFO" → {CEO} vs {CFO} → 不同
        topics:     主题名词集合（jieba n/a/vn/j 等实义名词，已去停用词）。

                    ⚠️ 为什么要有它：jieba 对很多公司/产品名**不标实体词性**
                    —— "苹果"被标成 n、"微软"被标成 a，两者的 entities 都只有
                    {CEO}，实体槽位完全相同，于是"苹果的CEO是谁"与
                    "微软的CEO是谁"会被判为可复用同一答案。topics 补上这个洞。
    """

    numbers: frozenset[str] = field(default_factory=frozenset)
    years: frozenset[str] = field(default_factory=frozenset)
    negations: frozenset[str] = field(default_factory=frozenset)
    comparatives: frozenset[str] = field(default_factory=frozenset)
    qualifiers: frozenset[str] = field(default_factory=frozenset)
    focus: frozenset[str] = field(default_factory=frozenset)
    entities: frozenset[str] = field(default_factory=frozenset)
    topics: frozenset[str] = field(default_factory=frozenset)


def _longest_only(tokens: set[str]) -> set[str]:
    """去掉被更长同类词包含的短词。

    例：命中集合 {"不", "不要"} → 只保留 {"不要"}。
    否则 "不要X" 与 "不X" 会因为一个多了 "不要"、一个只有 "不" 而被判不同，
    产生大量误拒。
    """
    return {t for t in tokens if not any(t != o and t in o for o in tokens)}


def _extract_cn_numbers(query: str) -> set[str]:
    """抽取**有真实数量语义**的中文数字。

    只在两种上下文下抽取（见 `_CN_ORDINAL_RE` / `_CN_QUANTITY_RE` 注释）：
      - 序数：第三届 → {"三"}
      - 量词：五个人 → {"五"}
    并剔除 "一/两" 这类常出现在 "介绍一下"、"说两句" 里的虚指用法。
    """
    found: set[str] = set()
    found |= set(_CN_ORDINAL_RE.findall(query))
    found |= set(_CN_QUANTITY_RE.findall(query))
    return {n for n in found if n not in _CN_NUM_STOPWORDS}


def extract_slots(query: str) -> QuerySlots:
    """抽取 query 的关键槽位指纹。

    这是**纯规则**实现（正则 + 可选 jieba 词性），特点：
      - 快：微秒级，对 L1 毫秒级短路的延迟预算完全无感
      - 确定性：同一 query 永远得到同一指纹，可复现、可单测
      - 无外部依赖：jieba 缺失时自动降级（只丢中文实体槽位，其余仍生效）

    Args:
        query: 原始 query 文本。
    Returns:
        QuerySlots

    示例：
        >>> s = extract_slots("苹果公司2024年的CEO是谁")
        >>> sorted(s.years), sorted(s.focus)
        (['2024'], ['WHO'])
    """
    if not query:
        return QuerySlots()

    q_lower = query.lower()

    # --- 数字 / 年份 ---
    numbers = set(_NUMBER_RE.findall(query))
    years = set(_YEAR_RE.findall(query))
    # 中文数字：只收「序数 / 量词」上下文里的，避免 "介绍一下" 的 "一" 造成误拒
    numbers |= _extract_cn_numbers(query)

    # --- 否定词：整串子串匹配（中文无空格，不能按 token 切）---
    negations = _longest_only({t for t in _NEGATION_TOKENS if t in q_lower})

    # --- 比较级 / 极值 ---
    comparatives = _longest_only(
        {t for t in _COMPARATIVE_TOKENS if t in q_lower}
    )

    # --- 限定词：拦截 "总统 vs 副总统"、"现任 vs 前任" ---
    # 两路合并（见 _QUALIFIER_PHRASES / _QUALIFIER_PREFIX_RE 的设计说明）：
    #   (a) 多字限定词直接子串匹配；
    #   (b) 单字职务前缀必须紧跟职务名词，避免 "当前" 被误当成 "前任"。
    qualifiers = _longest_only({t for t in _QUALIFIER_PHRASES if t in q_lower})
    qualifiers |= set(_QUALIFIER_PREFIX_RE.findall(query))

    # --- 疑问焦点：归一化为 WHO/WHERE/WHEN/... 标签 ---
    focus: set[str] = set()
    for label, cues in _INTERROGATIVE_FOCUS.items():
        if any(c in q_lower for c in cues):
            focus.add(label)

    # --- 命名实体 ---
    entities: set[str] = set()
    topics: set[str] = set()
    # (a) ASCII 缩写/英文词：CEO、CFO、GPT-4、API…（大写归一化便于比较）
    #     这一路不依赖 jieba，是"苹果CEO vs 苹果CFO"的主要拦截手段。
    for w in _ASCII_WORD_RE.findall(query):
        if len(w) >= 2:
            entities.add(w.upper())
    # (b) jieba 词性标注抽中文实体（人名/地名/机构/专名）+ 主题名词
    if _has_jieba and _pseg is not None:
        try:
            for tok in _pseg.cut(query):
                flag, word = tok.flag, tok.word
                if len(word) < 2:
                    continue
                if flag in _ENTITY_POS_FLAGS or flag[:2] in _ENTITY_POS_FLAGS:
                    entities.add(word)
                elif flag in _TOPIC_POS_FLAGS and word not in _TOPIC_STOPWORDS:
                    # 主题名词：拦截 "苹果的CEO" vs "微软的CEO"
                    #（jieba 把苹果标成 n、微软标成 a，两者都不是实体标签）
                    topics.add(word)
        except Exception:
            # jieba 异常不应影响主流程；降级为只用 ASCII 槽位
            pass

    return QuerySlots(
        numbers=frozenset(numbers),
        years=frozenset(years),
        negations=frozenset(negations),
        comparatives=frozenset(comparatives),
        qualifiers=frozenset(qualifiers),
        focus=frozenset(focus),
        entities=frozenset(entities),
        topics=frozenset(topics),
    )


# ── 疑问焦点的「等价类」：这些焦点期望的答案形态相同，可互相复用 ──────────
#
# 为什么需要等价类：`_INTERROGATIVE_FOCUS` 的标签是按**疑问词字面**归一化的，
# 但不同疑问词可能指向**同一种答案**。实测的两个误拒：
#     "美国历届总统名单"  focus = ∅         （陈述式名词短语，没有疑问词）
#     "美国历届总统有哪些" focus = {WHICH}
#     "美国总统历届都是谁" focus = {WHO}
# 三者要的都是「一串人名」，答案完全可复用，却因为标签不等而全被拒。
#
# WHO 与 WHICH 都属于「枚举/指认实体」，归为一类。
# 刻意**不**把 HOWMANY 放进来 —— "有哪些总统"（要名单）与 "有多少总统"（要数字）
# 答案形态不同，必须保持互斥。
_FOCUS_EQUIV_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"WHO", "WHICH"}),
)

# ── ∅（未指定焦点）允许与哪些焦点配对 ────────────────────────────────────
#
# ⚠️ 这里踩过一个坑，值得记下来：第一版把 ∅ 当成**万能通配**（"任一方为空即
#   兼容"），结果放过了一个真实的负样本：
#       缓存「美国历届总统名单」 focus = ∅         → 答案是一串人名
#       查询「美国有多少位总统」 focus = {HOWMANY} → 答案是一个数字
#   两者其余槽位全同（实体 {美国}、主题 {总统}），于是被判为可复用 —— 用户
#   问"多少位"却拿到一份名单。
#
# 修正后的认识：**∅ 不是"任意焦点"，而是"隐含的枚举/指认"**。
#   像「美国历届总统名单」「量子计算」「Transformer介绍」这类陈述式名词短语，
#   语义上等价于"请列出/说明它"，正好落在 WHO/WHICH 这一类。
#   而 HOWMANY（要数字）/ WHERE（要地点）/ WHEN（要时间）/ WHY（要原因）
#   是在追问**另一个属性**，答案形态完全不同，不能与 ∅ 互通。
#
# 为什么不把 HOW 也放进来：「Nginx反向代理配置」(∅) vs「如何配置Nginx反向
#   代理」({HOW}) 看起来是可以复用的，但按本模块"保守优先"的原则——误拒只
#   损失一次检索（几秒），误放行则返回错误答案（不可接受）——先不放宽。
#   这两种问法的余弦通常很高，靠分数门槛+其余槽位已能正常命中大部分场景。
_FOCUS_OPEN_COMPATIBLE: frozenset[str] = frozenset({"WHO", "WHICH"})


def _focus_compatible(
    fa: frozenset[str], fb: frozenset[str]
) -> tuple[bool, str]:
    """判断两个疑问焦点集合是否兼容。

    ⚠️ 这个槽位是**唯一不能用精确相等**判定的离散槽位（topics 用 Jaccard 是
    另一种情况）。原因有两个，都是实测踩出来的：

    **原因一：空集表示「未指定」，而不是「一种焦点」。**
        陈述式提问根本不含疑问词：
            "美国历届总统名单"      → ∅
            "介绍一下量子计算"       → ∅
        它们与对应的疑问句式（"…有哪些"/"…都是谁"）问的是同一件事。
        如果拿 ∅ 去和 {WHICH} 做相等比较，**所有「陈述式 ↔ 疑问式」的改述
        都会被拒**，而这恰恰是用户最常见的两种问法。
        这正是用户报的 bug 之一：缓存里存的是陈述式，用户用疑问式问 → 全miss。

        但 ∅ **不是万能通配**（这一点第一版搞错了，见
        `_FOCUS_OPEN_COMPATIBLE` 的注释）：∅ 的隐含语义是"请列出/说明"，
        只与 WHO/WHICH 相容。「美国历届总统名单」(∅) 与
        「美国有多少位总统」({HOWMANY}) 要的是完全不同的答案形态，必须拒绕。

    **原因二：不同疑问词可能要同一种答案。**
        "…有哪些"{WHICH} 与 "…都是谁"{WHO} 都要一串人名 → 见 _FOCUS_EQUIV_GROUPS。

    判定规则（从宽到严）：
        1. 双方都为空 → 兼容
        2. 一方为空   → 仅当另一方 ⊆ `_FOCUS_OPEN_COMPATIBLE`（WHO/WHICH）才兼容
        3. 有交集     → 兼容
        4. 落在同一等价类 → 兼容
        5. 其余（完全不同的焦点，如 WHO vs WHERE）→ 拒绝

    Args:
        fa: 候选缓存条目的焦点集合。
        fb: 当前 query 的焦点集合。
    Returns:
        (是否兼容, 拒绝理由)。

    示例：
        >>> _focus_compatible(frozenset(), frozenset({"WHICH"}))[0]
        True
        >>> _focus_compatible(frozenset(), frozenset({"HOWMANY"}))[0]
        False
        >>> _focus_compatible(frozenset({"WHO"}), frozenset({"WHICH"}))[0]
        True
        >>> _focus_compatible(frozenset({"WHO"}), frozenset({"WHERE"}))[0]
        False
    """
    # 1) 双方都未指定焦点（两个陈述式短语）
    if not fa and not fb:
        return True, ""
    # 2) 一方未指定：∅ 隐含"枚举/指认"，只与 WHO/WHICH 相容。
    #    绝不能当万能通配 —— 否则 "…名单"(∅) 会命中 "…有多少"({HOWMANY})。
    if not fa or not fb:
        other = fb or fa
        if other <= _FOCUS_OPEN_COMPATIBLE:
            return True, ""
        return False, (
            f"疑问焦点冲突: 陈述式(∅) vs {sorted(other)}"
            f"（陈述式隐含「枚举/说明」，与该焦点期望的答案形态不同）"
        )
    # 3) 有共同焦点
    if fa & fb:
        return True, ""
    # 4) 同一等价类内（如 WHO ~ WHICH，都要"枚举实体"）
    for group in _FOCUS_EQUIV_GROUPS:
        if fa <= group and fb <= group:
            return True, ""
    # 5) 真正冲突：如 "是谁"(WHO) vs "在哪"(WHERE)，答案类型都不同
    return False, (
        f"疑问焦点冲突: {sorted(fa)} vs {sorted(fb)}（期望的答案类型不同）"
    )


def slots_compatible(
    query_a: str,
    query_b: str,
    *,
    strict_entities: bool = True,
) -> tuple[bool, str]:
    """判断两个 query 的槽位是否兼容（即能否复用同一个缓存答案）。

    这是 **fuzzy 命中的最后一道门禁**：向量说"像"，槽位说"是不是同一个问题"。

    比对规则（任一不通过即拒绝）：
        1. 数字集合必须完全相同     —— "2024" vs "2025"
        2. 年份集合必须完全相同     —— 单独再校验一次，权重更高
        3. 否定词集合必须完全相同   —— "要" vs "不要"（答案完全反转）
        4. 比较级集合必须完全相同   —— "最大" vs "最小"
        5. 限定词集合必须完全相同   —— "总统" vs "副总统"、"现任" vs "前任"
        6. 疑问焦点需**兼容**（非相等） —— ∅（陈述式）与 WHO/WHICH 兼容，
                                       WHO~WHICH 等价；而 ∅ vs HOWMANY
                                       、WHO vs WHERE 这类仍拒。
                                       详见 `_focus_compatible()`。
        7. 命名实体集合必须完全相同 —— "CEO" vs "CFO"、"苹果" vs "微软"
        8. 主题名词用 Jaccard 相似度  —— 允许少量分词差异

    Args:
        query_a:         候选缓存条目的原始 query。
        query_b:         当前用户 query。
        strict_entities: True 时要求实体集合**完全相等**（推荐，最安全）；
                         False 时只要求"没有对称差集"（等价，保留参数以便
                         未来放宽为单向包含）。默认 True。

    Returns:
        (是否兼容, 拒绝理由)。兼容时理由为空串。

    示例：
        >>> slots_compatible("苹果的CEO是谁", "苹果的CFO是谁")[0]
        False
        >>> slots_compatible("量子计算是什么", "什么是量子计算")[0]
        True
    """
    sa = extract_slots(query_a)
    sb = extract_slots(query_b)

    # 逐槽位比对；每条都给出可读理由，便于 verbose 排查误拒
    # 注意：疑问焦点（focus）**不在这个精确相等列表里**，它走下面的
    # `_focus_compatible()` 做兼容判定——因为∅（陈述式提问）必须能和
    # 「枚举类」焦点共存，否则「陈述式 ↔ 疑问式」的改述会全部被误拒。
    checks: tuple[tuple[str, frozenset, frozenset], ...] = (
        ("数字", sa.numbers, sb.numbers),
        ("年份", sa.years, sb.years),
        ("否定词", sa.negations, sb.negations),
        ("比较级", sa.comparatives, sb.comparatives),
        ("限定词", sa.qualifiers, sb.qualifiers),
    )
    for name, va, vb in checks:
        if va != vb:
            return False, (
                f"{name}槽位不一致: {sorted(va) or '∅'} != {sorted(vb) or '∅'}"
            )

    # 疑问焦点：兼容式判定（∅ 与 WHO/WHICH 兼容，WHO~WHICH 等价）
    _ok_focus, _why_focus = _focus_compatible(sa.focus, sb.focus)
    if not _ok_focus:
        return False, _why_focus

    # 实体槽位：单独处理，因为可以选严格/宽松
    if strict_entities:
        if sa.entities != sb.entities:
            return False, (
                f"实体槽位不一致: {sorted(sa.entities) or '∅'} "
                f"!= {sorted(sb.entities) or '∅'}"
            )
    else:
        diff = sa.entities.symmetric_difference(sb.entities)
        if diff:
            return False, f"实体槽位存在差异: {sorted(diff)}"

    # ══════════════════════════════════════════════════════════════════
    # 主题名词槽位：**用 Jaccard 相似度而非精确相等**
    # ══════════════════════════════════════════════════════════════════
    # 为什么这一槽位不能像其它槽位那样要求"完全相等"：
    #   主题名词受分词与句式影响大，同义改述时会有小幅出入。比如
    #       "讲讲RAG的原理"     → topics 可能含 {原理}
    #       "RAG 的原理是什么"  → topics 可能含 {原理}
    #   一般能对上，但换成
    #       "量子计算是什么"    → {量子}
    #       "什么是量子计算"    → {量子}
    #   也能对上；然而
    #       "介绍一下Transformer" → topics = ∅（Transformer 走 entities）
    #   与 "Transformer介绍"（"介绍"已在停用词里）→ ∅，同样能对上。
    #   但边缘情况下（分词把"量子计算"切成"量子"+"计算"，而另一句没切）
    #   要求完全相等会造成误拒。
    #
    # 所以改用 Jaccard：交集/并集 >= 阈值即认为主题一致。
    #   "苹果的CEO" {苹果} vs "微软的CEO" {微软} → Jaccard = 0.0 → 拒绝 ✓
    #   小幅分词差异 {量子,计算} vs {量子}        → Jaccard = 0.5 → 允许 ✓
    # 阈值 0.5：要求至少一半主题词重合。
    if sa.topics or sb.topics:
        inter = len(sa.topics & sb.topics)
        union = len(sa.topics | sb.topics)
        jac = inter / union if union else 1.0
        if jac < TOPIC_JACCARD_MIN:
            return False, (
                f"主题槽位差异过大 (Jaccard={jac:.2f} < {TOPIC_JACCARD_MIN}): "
                f"{sorted(sa.topics) or '∅'} != {sorted(sb.topics) or '∅'}"
            )

    return True, ""


# ════════════════════════════════════════════════════════════════════════
#                              自检 / 演示
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":  # pragma: no cover
    print("=" * 72)
    print("防线 A：准入策略 decide_cacheability()")
    print("=" * 72)
    _samples = [
        ("今天上海天气怎么样", "上海今天晴，气温 25 度，东南风 3 级。", None),
        ("量子计算是什么", "量子计算是利用量子力学原理进行信息处理的计算范式……", None),
        ("2020年中国GDP是多少", "2020 年中国 GDP 约为 101.6 万亿元人民币。", None),
        ("特朗普最新表态", "抱歉，我没有找到相关信息。", {"L4_web": 5}),
        ("Python装饰器怎么用", "装饰器是一种语法糖，用于在不修改原函数……", {"L4_web": 3}),
        ("你好", "你好！", None),
    ]
    for _q, _a, _lh in _samples:
        _d = decide_cacheability(_q, _a, _lh)
        _ttl = "—" if _d.ttl is None else f"{_d.ttl // 3600}h"
        print(f"  [{'✓' if _d.cacheable else '✗'}] {_q!r:<28} "
              f"tier={_d.tier:<22} ttl={_ttl:<6} {_d.reason}")

    print()
    print("=" * 72)
    print("防线 B：槽位门禁 slots_compatible()")
    print("=" * 72)
    _pairs = [
        ("美国总统是谁", "美国副总统是谁"),
        ("苹果的CEO是谁", "苹果的CFO是谁"),
        ("苹果的CEO是谁", "微软的CEO是谁"),      # 主题名词槽位拦截
        ("2024年GDP是多少", "2025年GDP是多少"),
        ("推荐一些保健品", "不要推荐保健品"),
        ("世界最高的山是哪座", "世界最低的地方是哪里"),
        ("现任美国总统是谁", "前任美国总统是谁"),
        ("第三届世界杯在哪举办", "第五届世界杯在哪举办"),
        # —— 下面这些应当「允许」命中（同义改述）——
        ("量子计算是什么", "什么是量子计算"),
        ("介绍一下Transformer", "Transformer介绍"),
        ("讲讲RAG的原理", "RAG 的原理是什么"),
        ("美国总统是谁？", "当前美国总统是谁"),   # "当前"≠"前任"
    ]
    for _qa, _qb in _pairs:
        _ok, _why = slots_compatible(_qa, _qb)
        print(f"  [{'✓ 允许' if _ok else '✗ 拒绝'}] {_qa!r} vs {_qb!r}")
        if _why:
            print(f"           └─ {_why}")
