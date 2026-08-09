# query_rewriter.py
"""Query 改写器：把用户原始 query 转成更适合搜索引擎检索的形式。

包含三个对外函数：
1. shorten_query()        : 非 LLM 规则方式（jieba 分词 + 否定意图识别）
2. rewrite_query()        : LLM 方式（本地 Ollama qwen3:4b）
3. query_rewrite_route()  : 路由器，按 rewrite_type 选择实现策略

输出格式统一：
  - 命中：返回搜索友好的 query 字符串
  - 不需要联网：返回 config.NO_SEARCH_SENTINEL（"NO_SEARCH"）
  - 异常/空结果：回退到原始 query
"""
from __future__ import annotations

import re

from configs import config
from context_provider import build_context_block
from llm_client import complete as llm_complete
from configs.prompts import render as render_prompt

# jieba 可选：用于中文分词 + 词性标注（识别实体），缺失时自动降级
_has_jieba: bool = False
try:
    import jieba  # noqa: F401
    import jieba.posseg as pseg
    _has_jieba = True
except Exception:
    pseg = None  # type: ignore


# ============================================================
#                   常量 / 词典（规则方式使用）
# ============================================================

# jieba 中识别"实体/核心信息"的词性标签（优先保留）
#   n=普通名词  nr=人名  ns=地名  nt=机构  nz=其他专名  nl=名词性惯用语
#   j=简称略语  vn=名动词  an=名形词  t=时间词  m=数词  eng=英文  x=非语素(包括字母数字)
_ENTITY_FLAGS: set[str] = {
    "n", "nr", "nrt", "nrfg", "ns", "nt", "nz", "nl",
    "j", "vn", "an", "t", "m", "mq", "eng", "x",
}

# 常用停用词 / 语气助词 / 虚词（中英双语）
STOPWORDS: set[str] = {
    # —— 中文：助词 / 语气词 / 代词 / 介词 / 连词 ——
    "的", "地", "得", "了", "着", "过", "吗", "呢", "吧", "啊", "呀", "哦", "哈",
    "是", "在", "和", "与", "及", "或", "并", "也", "都", "就", "还", "又", "再",
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
    "这", "那", "这个", "那个", "这些", "那些", "这样", "那样", "这里", "那里",
    "什么", "怎么", "怎样", "如何", "哪个", "哪些", "为什么", "为何", "多少", "几",
    "有", "没有", "没",
    "把", "被", "给", "让", "从", "向", "往", "到", "对", "对于", "关于", "由于",
    "因为", "所以", "但是", "而且", "如果", "虽然", "然而", "因此", "并且",
    "请问", "麻烦", "帮我", "帮忙", "一下", "想问", "想要", "想知道", "告诉我",
    "请", "谢谢", "麻烦了", "可以", "能否", "能不能", "可不可以",
    "当前", "现在", "目前", "最近", "刚才",
    # —— 英文：常见 stopwords ——
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "of", "to", "in", "on", "at", "by", "for",
    "with", "about", "from", "as", "into", "through", "during", "before", "after",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom", "whose",
    "how", "why", "when", "where", "do", "does", "did", "have", "has", "had",
    "can", "could", "would", "should", "may", "might", "will", "shall", "must",
    "please", "tell", "want", "would", "like",
}

# 标点 / 符号（不作为检索关键词）
_PUNCT: set[str] = set("，。！？；：、,.?!;:'\"“”‘’（）()【】[]<>《》「」…—–·-_/\\|`~@#$%^&*+=")

# —— 否定线索（子句级触发，整子句中的核心词改为"排除项"） ——
_NEGATION_CUES: set[str] = {
    # 中文
    "不", "不要", "不想", "不需要", "不希望", "不喜欢", "别要", "别买", "别选",
    "无需", "勿", "莫", "未", "非", "无", "没", "没有",
    "排除", "去掉", "除外", "除了", "除", "不含", "不带", "不包括", "避免", "拒绝",
    "讨厌", "反感", "不感兴趣",
    # 英文
    "no", "not", "don't", "dont", "doesn't", "doesnt", "won't", "wont",
    "without", "exclude", "except", "avoid", "skip", "minus", "no-",
}

# —— 转折/对比连接词 ——
_CONTRAST_MARKERS: set[str] = {
    "但", "但是", "可是", "不过", "然而", "却", "只是", "而",
    "but", "however", "yet", "though",
}

# —— 子句切分：句号/问号/感叹号/逗号/分号/顿号/换行/中文逗号 ——
_CLAUSE_SPLIT_RE = re.compile(r"[，。！？；,.;!?\n、]+")


# ============================================================
#                LLM 方式 (rewrite_query) 相关常量
# ============================================================
# Prompt 模板已经迁移到 prompts.py（key = "rewriter"）；
# 模型/参数已经迁移到 models_config.STAGES["rewriter"]。
# 本文件只关心"规则 + 路由"，不再硬编码 prompt 字面量。


# ============================================================
#                  非 LLM 方式：内部工具函数
# ============================================================

def _tokenize(query: str) -> list[tuple[str, str]]:
    """对 query 做分词 + 词性标注，返回 [(word, flag), ...]。

    - 含空格 → 视为已分词（英文 / 已分词中文）
    - 否则用 jieba.posseg 切（中文）；jieba 缺失则降级为单 token
    """
    if " " in query.strip():
        toks: list[tuple[str, str]] = []
        for w in query.split():
            if not w:
                continue
            if w.isascii():
                toks.append((w, "eng"))
            elif _has_jieba and pseg is not None:
                toks.extend((tw.word, tw.flag) for tw in pseg.cut(w))
            else:
                toks.append((w, "x"))
        return toks

    if _has_jieba and pseg is not None:
        return [(t.word, t.flag) for t in pseg.cut(query)]

    return [(query, "x")]


def _is_negative_clause(clause: str, tokens: list[tuple[str, str]]) -> bool:
    """判断子句是否为否定意图。"""
    low = clause.lower()
    for cue in _NEGATION_CUES:
        if cue in low:
            return True
    for w, _ in tokens:
        if w.lower() in _NEGATION_CUES:
            return True
    return False


def _extract_keywords(tokens: list[tuple[str, str]]) -> list[str]:
    """从已分词子句中抽取关键词（去停用/标点/虚词/否定线索/转折词）。"""
    out: list[str] = []
    for w, flag in tokens:
        w = w.strip()
        if not w:
            continue
        if w in _PUNCT or all(c in _PUNCT for c in w):
            continue
        if w.lower() in STOPWORDS:
            continue
        if w.lower() in _NEGATION_CUES:
            continue
        if w.lower() in _CONTRAST_MARKERS:
            continue
        if len(w) == 1 and flag in {"u", "p", "c", "d", "y", "e", "o", "r", "uj"}:
            continue
        out.append(w)
    return out


def _join_tokens(ws: list[str]) -> str:
    """中文 token 直接拼接，英文 token 之间留空格。"""
    out = ""
    for w in ws:
        if not w:
            continue
        if out and (out[-1].isascii() and w[0].isascii()):
            out += " " + w
        else:
            out += w
    return out


def _dedup(seq: list[str]) -> list[str]:
    """去重保序（按小写比较）。"""
    seen: set[str] = set()
    out: list[str] = []
    for w in seq:
        key = w.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(w)
    return out


# ============================================================
#                ① 非 LLM 方式：shorten_query
# ============================================================

def shorten_query(query: str) -> str:
    """非 LLM 的 query 改写（规则方式 + jieba）。

    流程：
      1) 标点 + 转折词切分子句
      2) 子句级判断是否「否定意图」
      3) 否定子句的核心词加 `-` 前缀（搜索引擎通用排除语法）
      4) 正向子句保留实体优先的关键词
      5) 截断到 config.SEARCH_LONG_QUERY_THRESHOLD 个正向词；否定项不计入截断

    示例:
      "我比较累,想买东西缓解疲劳,但不想要保健品"
        → "比较 累 想 买 东西 缓解 疲劳 -保健品"
    """
    query = (query or "").strip()
    if not query:
        return ""

    n = config.SEARCH_LONG_QUERY_THRESHOLD

    # 1) 子句切分（标点 + 转折词）
    raw_clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(query) if c.strip()]
    clauses: list[str] = []
    for c in raw_clauses:
        toks = _tokenize(c)
        cur: list[str] = []
        for w, _f in toks:
            if w.lower() in _CONTRAST_MARKERS and cur:
                clauses.append(_join_tokens(cur))
                cur = []
            else:
                cur.append(w)
        if cur:
            clauses.append(_join_tokens(cur))
    if not clauses:
        clauses = [query]

    # 2) 逐子句处理：分类为正向 / 负向
    pos_keywords: list[str] = []
    neg_keywords: list[str] = []
    for c in clauses:
        toks = _tokenize(c)
        kws = _extract_keywords(toks)
        if not kws:
            continue
        if _is_negative_clause(c, toks):
            neg_keywords.extend(kws)
        else:
            pos_keywords.extend(kws)

    # 3) 去重
    pos_keywords = _dedup(pos_keywords)
    neg_keywords = _dedup(neg_keywords)
    # 同一关键词正/负冲突时，否定优先
    pos_keywords = [
        w for w in pos_keywords
        if w.lower() not in {x.lower() for x in neg_keywords}
    ]

    # 4) 全部为空 → 回退原 query
    if not pos_keywords and not neg_keywords:
        return query

    # 5) 截断
    pos_keywords = pos_keywords[:n]
    parts = pos_keywords + [f"-{w}" for w in neg_keywords]
    return " ".join(parts)


# ============================================================
#                ② LLM 方式：rewrite_query
# ============================================================

def rewrite_query(
    user_query: str,
    history: str = "",
    model: str | None = None,
    temperature: float | None = None,
    enable_thinking: bool | None = None,
) -> str:
    """调用 LLM 完成 query 改写。
      - 模型/provider 走 models_config.STAGES["rewriter"]
      - prompt 模板走 prompts.PROMPTS["rewriter"]
      - 参数 (model / temperature / enable_thinking) 仅作为"临时覆盖"使用，
        缺省时全部从配置文件读取，便于一处修改、处处生效。

    返回：
      - 一个搜索友好的 query 字符串
      - "NO_SEARCH"（config.NO_SEARCH_SENTINEL）表示无需联网检索
      - 失败时返回原始 query（与 shorten_query 输出格式对齐）
    """
    user_query = (user_query or "").strip()
    if not user_query:
        return ""

    try:
        prompt = render_prompt(
            "rewriter",
            context=build_context_block(),
            history=history if history else "(无)",
            query=user_query,
        )

        # 运行期覆盖：调用方传入 model / temperature 时优先使用
        overrides: dict = {}
        if model is not None:
            overrides["model"] = model
        if temperature is not None:
            overrides["temperature"] = temperature
        # qwen3 thinking 开关：通过 extra 透传给 ollama provider
        if enable_thinking is not None:
            overrides["extra"] = {"think": bool(enable_thinking)}
            if not enable_thinking:
                prompt += "\n/no_think"

        text = llm_complete("rewriter", prompt, **overrides).strip()

        # 清洗 thinking 模式残留
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        # 只取首行，避免模型啰嗦
        first_line = text.splitlines()[0].strip() if text else ""
        return first_line or user_query
    except Exception as e:
        print(f"[query_rewriter] rewrite_query 调用失败: {e}")
        return user_query


# ============================================================
#            对话历史污染检测（history contamination）
# ============================================================
# 背景：mode 1/2 会把「对话历史」塞进 rewriter prompt，用途是补全代词
# （"它涨了多少" → "英伟达股价涨了多少"）。代价是历史里的**实体**可能
# 被误当成本轮意图，改写出跨题的 query。曾观测到一次：
#     本轮提问：日本目前为止今年发生了几次地震
#     实际改写：日本 今年 地震 次数 USGS 数据 震中位置 五位数邮政编码
#                                    └────── 全部来自上一题（小丑鱼/USGS）
# 「五位数邮政编码」会把检索带偏，直接稀释召回质量。
#
# ⚠️ 关于复现：我用 20 次重复（占位答案 / 完整长拒答 / 真实 agent 链路
#    三种历史形态）都**没能复现**，说明它是低频抖动（rewriter
#    temperature=0.2，非 0）。因此这里**不改 prompt、不删历史**——
#    在无法复现的情况下调 prompt 等于盲改，可能反而破坏代词补全。
#
# 采取的策略是「可观测 + 兜底」，而不是「猜一个修法」：
#   · 正常情况零影响（纯字符串集合运算，无额外 LLM 调用）；
#   · 真的漂移时打印告警，把低频抖动变成**可排查**的显式信号；
#   · 只在最严重的情形（改写结果与本轮 query 毫无交集）才回退规则方式。
# 这样即使今后偶发，也能从日志里直接定位，而不是又一次"查不出来"。

def _core_terms(text: str) -> set[str]:
    """抽取用于比对的实词集合（复用规则方式的分词与停用词表）。"""
    toks = _tokenize(text or "")
    return {w.lower() for w in _extract_keywords(toks) if len(w) > 1}


def detect_history_contamination(
    user_query: str, rewritten: str, history: str = "",
) -> tuple[bool, set[str]]:
    """检测改写结果是否被对话历史带偏。

    返回 (是否严重漂移, 疑似来自历史的词集)。

    判定逻辑分两级，**故意保持保守**——改写器加词是它的本职工作
    （补专业术语、地名以提高召回），所以"加了历史里的词"本身不算错，
    只有下面两种才算问题：

      ① 疑似污染（仅告警，不拦截）：
         改写结果里出现了「历史里有、但本轮 query 里没有」的实词。
         这只是**嫌疑**：也可能是合理的代词补全（"它" → "英伟达"），
         所以绝不能据此拦截，只打日志供人工核查。

      ② 严重漂移（触发兜底）：
         改写结果与本轮 query 的实词**交集为空**。
         此时改写结果已经和用户这次问的东西完全无关，无论原因是什么，
         都不该拿去检索 —— 回退规则方式一定比它更贴题。
    """
    q_terms = _core_terms(user_query)
    r_terms = _core_terms(rewritten)
    if not q_terms or not r_terms:
        return False, set()

    h_terms = _core_terms(history) if history else set()
    # 出现在历史与改写结果里，却不在本轮 query 里 → 疑似来自历史
    suspicious = (r_terms & h_terms) - q_terms
    # 严重漂移：改写结果与本轮 query 完全不沾边
    severe = not (r_terms & q_terms)
    return severe, suspicious


# ============================================================
#                ③ 路由：query_rewrite_route
# ============================================================

def _is_empty_or_invalid(result: str) -> bool:
    """判断 LLM 改写结果是否"无效"（当前口径：**仅判空**）。

    若被判为无效，路由器会回退到规则方式 `shorten_query` 兜底。

    为什么只判空、不再判「与原句相同」：
        「改写结果 == 原句」曾被当作失败信号，但实测它恰恰是**正确行为** ——
        当原句本身已经是干净、适合检索的短句时（「量子计算是什么」），
        最优改写就是原样返回。把它判为失败会触发 `shorten_query` 二次裁剪，
        反而可能切掉必要的限定词，让检索质量变差。

        真正的失败只有一种：模型什么都没输出（空串 / 纯空白）。

    NO_SEARCH 同样不算"无效" —— 它是有效信号，由调用方在此之前拦截。
    """
    return not result


def _guard_history_drift(user_query: str, rewritten: str, history: str) -> str:
    """对改写结果做一次历史污染体检，必要时回退规则方式。

    只在 mode 1/2（会吃 history 的路径）调用。mode 0 是纯规则改写，
    根本抽不到历史，不需要体检。

    两类处理（为何如此分级见 detect_history_contamination 的注释）：
      · 严重漂移 → 回退 shorten_query（它只看本轮 query，不可能跑题）
      · 仅可疑 → **只告警不拦截**，因为它也可能是合理的代词补全
    """
    if not history:
        return rewritten
    if rewritten == config.NO_SEARCH_SENTINEL:
        return rewritten

    severe, suspicious = detect_history_contamination(
        user_query, rewritten, history)

    if severe:
        fallback = shorten_query(user_query)
        print(f"[query_rewriter] ⚠️ 改写结果与本轮提问无任何实词交集，"
              f"判为被历史带偏，回退规则改写："
              f"{rewritten!r} → {fallback!r}")
        return fallback

    if suspicious:
        # 仅提示。这里不能拦截 —— “它涨了多少” → “英伟达股价…”
        # 里的“英伟达”同样来自历史，却正是我们想要的行为。
        print(f"[query_rewriter] ℹ️ 改写结果含来自对话历史的词 "
              f"{sorted(suspicious)[:6]}（可能是代词补全，也可能是跑题；"
              f"若检索质量异常请优先排查这里）")
    return rewritten


def query_rewrite_route(
    user_query: str,
    history: str = "",
    rewrite_type: int = 0,
    model: str | None = None,
    temperature: float | None = None,
    enable_thinking: bool | None = None,
) -> str:
    """统一的 query 改写入口。

    rewrite_type:
      0 → 仅规则方式：shorten_query()
      1 → 仅 LLM 方式：rewrite_query()
      2 → 混合方式：先 rewrite_query()；若结果为空/异常 → 用 shorten_query() 兜底

    其它参数会原样透传给 rewrite_query()，未指定时均使用 config 默认值。

    返回：
      - 改写后的 query 字符串（统一格式）
      - "NO_SEARCH" 表示无需联网检索
      - 输入为空 → 返回空字符串
    """
    user_query = (user_query or "").strip()
    if not user_query:
        return ""

    # ---------- mode 0: 纯规则 ----------
    if rewrite_type == 0:
        return shorten_query(user_query)

    # ---------- mode 1: 纯 LLM ----------
    if rewrite_type == 1:
        out = rewrite_query(
            user_query=user_query,
            history=history,
            model=model,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        return _guard_history_drift(user_query, out, history)

    # ---------- mode 2: 混合（LLM 优先，规则兜底）----------
    if rewrite_type == 2:
        llm_out = rewrite_query(
            user_query=user_query,
            history=history,
            model=model,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        # NO_SEARCH 是有效信号，直接返回
        if llm_out == config.NO_SEARCH_SENTINEL:
            return llm_out
        # 空结果 → 视为 LLM 改写失败，用规则兜底
        # （注意：「改写结果 == 原句」不算失败，见 `_is_empty_or_invalid`）
        if _is_empty_or_invalid(llm_out):
            print("[query_rewriter] LLM 改写未生效，回退至 shorten_query")
            return shorten_query(user_query)
        return _guard_history_drift(user_query, llm_out, history)

    raise ValueError(
        f"Invalid rewrite_type={rewrite_type}, must be one of 0/1/2"
    )
