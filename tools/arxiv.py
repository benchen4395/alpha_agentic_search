# tools/arxiv.py
"""arXiv 论文检索（官方 Atom API，无需 key）。

API 文档: https://info.arxiv.org/help/api/index.html

════════════════════════════════════════════════════════════════════════
检索质量：为什么不能直接 `search_query=all:<用户输入>`
════════════════════════════════════════════════════════════════════════
arXiv 的 `all:` 字段对**未加引号的多词输入按 OR 语义**处理，再叠加
`sortBy=submittedDate`（按时间倒序、完全不看相关性），结果就是
"最近提交的论文里，凡沾一个词的都算命中"。

实测（2026-08，全库）：

    all:on-policy distillation             → 命中 74890 篇
      前 5 条：humanoid loco-manipulation / VLA / world model …（全不相关）
    all:"on-policy distillation"           → 命中   389 篇
      前 5 条：On-Policy Delta Distillation / STEP-OPD / SPOT …（全部相关）

74890 vs 389 —— 相差 190 倍，且前者返回的内容与用户意图**完全无关**。
配合"只保留最近 N 天"的过滤，用户得到的是"最近几天提交的随机论文"，
看起来像在正常工作（有标题、有作者、有链接），实则是噪声。这种
**静默的错误**比直接报错危险得多。

因此采用「短语优先 + 逐级放宽」的检索阶梯（见 `_build_search_query`）。

════════════════════════════════════════════════════════════════════════
时间过滤：用 API 原生 submittedDate 区间，而不是取回来再本地筛
════════════════════════════════════════════════════════════════════════
原先的做法是「取 max_results 条 → 本地按 published 时间过滤」。
问题在于 `max_results` 限制的是**过滤前**的条数：想要"最近 5 天的 10 篇"，
API 先返回全库最新的 10 篇，再筛掉不在窗口内的 —— 结果几乎必然少于 10 条，
冷门主题下直接返回空列表，而用户以为"最近没有相关论文"。

arXiv 支持在 `search_query` 里内联 `submittedDate:[YYYYMMDDHHMM TO ...]`，
把过滤下推到服务端，`max_results` 就成了真正的"返回条数"。
实测 `all:"quantum error correction" AND all:"surface code"
AND submittedDate:[202608040000 TO 202608100000]` → 命中 3 篇，精确可用。
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from ._http import ToolHTTPError, get_text


# 用 HTTPS：明文 HTTP 会被部分网络中间层拦截/改写，且 arXiv 已全面支持 TLS。
_API = "https://export.arxiv.org/api/query"

_NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
}

# arXiv 的布尔运算符与分组字符。用户输入里若混进这些，会破坏我们拼出的
# 查询语法（甚至造成 API 400），统一剥掉后再按短语处理。
_RESERVED_RE = re.compile(r'["()\[\]]|(?<!\w)(?:AND|OR|ANDNOT|NOT)(?!\w)', re.IGNORECASE)

# 停用词：放进 AND 阶梯只会无谓收窄（几乎每篇论文都含 "of"/"the"），
# 但**不影响**短语检索——短语里的停用词是有意义的（"attention is all you need"）。
_STOPWORDS = {
    "a", "an", "the", "of", "for", "on", "in", "to", "and", "or", "with",
    "via", "using", "based", "from", "by", "at", "is", "are", "be",
}

_MAX_RESULTS_CAP = 50      # arXiv 单次建议不超过这个量级，也够工具场景用了


def _normalize(query: str) -> str:
    """剥掉保留字符/运算符，压缩空白。"""
    cleaned = _RESERVED_RE.sub(" ", query or "")
    return " ".join(cleaned.split())


def _date_clause(days: int) -> str:
    """构造 arXiv 原生的 submittedDate 区间子句。

    格式为 `submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]`（UTC）。
    上界取"明天"而不是"现在"：arXiv 的 announce 流程存在时区与批次延迟，
    卡在当前时刻会漏掉刚提交的当天论文。
    """
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=days)).strftime("%Y%m%d%H%M")
    end = (now + timedelta(days=1)).strftime("%Y%m%d%H%M")
    return f"submittedDate:[{start} TO {end}]"


def _build_search_query(query: str, days: int) -> list[str]:
    """生成由严到宽的候选查询列表（检索阶梯）。

    为什么是"阶梯"而不是单条查询
    --------------------------
    精确短语的准确率最高，但召回可能为 0 —— 用户输入的往往是**主题描述**
    而不是论文里的原句。实测 `all:"quantum error correction surface code"`
    在 6 天窗口内命中 0 篇，但拆成两个短语 AND 起来就有 3 篇，且全部相关。

    所以按顺序尝试，**第一个有结果的即采用**：

      ① 整体短语     `all:"on policy distillation"`
                     —— 最精确，命中即最相关
      ② 逐词 AND     `all:"on" AND ...` → 实为 `all:on AND all:policy ...`
                     —— 要求全部词都出现（但不要求相邻），召回更宽、
                        准确率仍远高于 OR
      ③ 原样         `all:on policy distillation`
                     —— 等价于旧行为（OR 语义）。保留它只是为了
                        "宁可给点弱相关结果，也不要空手而归"，
                        并且**只在前两级都为空时**才会走到。

    单词输入时 ①②③ 会退化成同一条，`dict.fromkeys` 负责去重，
    避免对同一查询白跑三次网络请求。
    """
    norm = _normalize(query)
    if not norm:
        return []

    date_clause = _date_clause(days) if days and days > 0 else ""

    def _wrap(core: str) -> str:
        return f"({core}) AND {date_clause}" if date_clause else core

    words = norm.split()
    candidates = [f'all:"{norm}"']                      # ① 整体短语

    if len(words) > 1:
        # ② 逐词 AND，去掉停用词（保留至少一个词以免全被滤空）
        kept = [w for w in words if w.lower() not in _STOPWORDS] or words
        if len(kept) > 1:
            candidates.append(" AND ".join(f"all:{w}" for w in kept))
        candidates.append(f"all:{norm}")                # ③ 原样（OR 兜底）

    return list(dict.fromkeys(_wrap(c) for c in candidates))


def _parse_entries(xml_text: str) -> list[dict]:
    """把 Atom 响应解析成结构化论文列表。"""
    root = ET.fromstring(xml_text)
    out: list[dict] = []

    for e in root.findall("a:entry", _NS):
        title = (e.findtext("a:title", default="", namespaces=_NS) or "").strip()
        if not title:
            continue

        # arXiv 的 title/summary 里带有为固定宽度排版插入的换行与多余空格，
        # 原样落进 prompt 会浪费 token 且影响可读性，压平成单行。
        title = " ".join(title.split())
        summary = " ".join(
            (e.findtext("a:summary", default="", namespaces=_NS) or "").split()
        )

        authors = [
            " ".join((a.findtext("a:name", default="", namespaces=_NS) or "").split())
            for a in e.findall("a:author", _NS)
        ]
        authors = [a for a in authors if a]

        pc = e.find("arxiv:primary_category", _NS)
        primary_category = pc.get("term", "") if pc is not None else ""
        categories = [
            c.get("term", "") for c in e.findall("a:category", _NS) if c.get("term")
        ]

        abs_url = (e.findtext("a:id", default="", namespaces=_NS) or "").strip()
        # PDF 直链在 <link title="pdf">，比让用户自己从 abs 页面点进去更有用
        pdf_url = ""
        for link in e.findall("a:link", _NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        out.append({
            "title":            title,
            # 截断到 500 字：摘要要进 LLM prompt，而 evidence.py 有整体
            # 字数预算；给 10 篇论文各留 500 字是一个平衡点。
            "summary":          summary[:500],
            "url":              abs_url,
            "pdf_url":          pdf_url,
            "authors":          authors[:10],   # 高能物理论文常有上千作者
            "author_count":     len(authors),
            "primary_category": primary_category,
            "categories":       categories,
            "published":        (e.findtext("a:published", default="", namespaces=_NS) or "").strip(),
            "updated":          (e.findtext("a:updated", default="", namespaces=_NS) or "").strip(),
            # v2/v3 说明论文被修订过，对判断"是否还在活跃迭代"有参考价值
            "version":          abs_url.rsplit("v", 1)[-1] if "v" in abs_url.rsplit("/", 1)[-1] else "",
            "comment":          " ".join(
                (e.findtext("arxiv:comment", default="", namespaces=_NS) or "").split()
            )[:200],
            "source":           "arXiv",
        })
    return out


def search_arxiv(
    query: str,
    days: int = 5,
    max_results: int = 10,
    sort_by: str = "submittedDate",
) -> list[dict]:
    """检索 arXiv 论文。

    Args:
        query:       检索主题，如 'on-policy distillation'。
                     内部会优先当作**精确短语**检索，无结果时逐级放宽。
        days:        仅保留最近 N 天提交的论文（下推到 API 的
                     `submittedDate` 区间）。传 0 / None 表示不限时间。
        max_results: 返回条数上限（1~50）。因为时间过滤已下推到服务端，
                     这里就是**实际返回**的条数上限，不再被过滤削减。
        sort_by:     'submittedDate'（默认，最新优先）或 'relevance'
                     （最相关优先）。问"最近有什么新论文"用前者；
                     问"某个主题有哪些工作"用后者更合适。

    Returns:
        [{title, summary, url, pdf_url, authors, author_count,
          primary_category, categories, published, updated, version,
          comment, source}, ...]
        无匹配时返回 `[]`。

    Raises:
        ValueError:    query 为空。
        ToolHTTPError: 网络/服务端故障（重试后仍失败），或响应无法解析。

    ⚠️ 失败时**抛异常而不是返回 `[{"error": ...}]`**。
       后者会被 `call_tool()` 判成成功（非空列表且首元素不是它检查的
       那种 dict），导致错误文本被当作"外部资料"喂给 LLM，
       而不是让 agent 降级到通用检索。
    """
    if not (query or "").strip():
        raise ValueError("query 不能为空")

    # 参数兜底：LLM 路由经常把数字当字符串传（"5"），或给出越界值
    try:
        days = int(days) if days else 0
    except (TypeError, ValueError):
        days = 5
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = 10
    max_results = max(1, min(max_results, _MAX_RESULTS_CAP))

    if sort_by not in ("submittedDate", "relevance", "lastUpdatedDate"):
        sort_by = "submittedDate"

    candidates = _build_search_query(query, days)
    if not candidates:
        raise ValueError("query 清洗后为空（仅含保留字符）")

    last_error: ToolHTTPError | None = None

    for search_query in candidates:
        try:
            xml_text = get_text(
                _API,
                params={
                    "search_query": search_query,
                    "sortBy": sort_by,
                    "sortOrder": "descending",
                    "max_results": max_results,
                },
                timeout=20,
                label="arXiv",
            )
        except ToolHTTPError as e:
            # 记下来继续试下一级：某一级可能因语法/长度被拒，
            # 但更宽松的那级往往能成功。全部失败时才上抛。
            last_error = e
            continue

        try:
            entries = _parse_entries(xml_text)
        except ET.ParseError as e:
            last_error = ToolHTTPError(f"arXiv 返回的 XML 无法解析: {e}")
            continue

        if entries:
            return entries[:max_results]

    if last_error is not None:
        raise last_error
    # 所有阶梯都成功请求但都无结果 —— 这是合法的"确实没有"，不是错误。
    # 返回空列表让 call_tool 归类为 kind="empty" → agent 降级去通用检索。
    return []
