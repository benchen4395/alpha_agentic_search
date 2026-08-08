# answer_types.py
"""结构化答案契约：AnswerResult / Source / Citation（P0.5）。

════════════════════════════════════════════════════════════════════════
问题：来源信息在 chat() 返回时被整个丢弃
════════════════════════════════════════════════════════════════════════
Perplexity 的核心产品价值就是「每句话都有可点击的来源」。而改造前的实现：

  1. `configs/prompts.py` 的 SUMMARY_SYSTEM 只是**祈使句式**地要求
     「用 [n] 标注来源序号」——没有任何机制保证 `[3]` 真的对应第 3 条资料，
     更没有校验被标注的句子是否真的被该资料支持。模型完全可以编一个 `[7]`
     出来（而资料只有 5 条）。

  2. **更致命的是 `chat()` 返回 `str`**。
     `rag_result.passages`（含 title / url / layer / score）在函数结束时
     就随栈销毁了。调用方（`main.py` CLI、`main_web.py` Web、
     `scripts/search.py`）**根本拿不到来源列表**，前端无从渲染 sources 面板。
     所以 Web UI 里确实一条来源都没有。

  结论：即使 LLM 老老实实标了 `[1][2]`，用户也点不动、验不了 ——
  Perplexity 最核心的那个体验在当前实现里完全不存在。

════════════════════════════════════════════════════════════════════════
本模块提供的契约
════════════════════════════════════════════════════════════════════════
    ┌─────────────────────────────────────────────────────────┐
    │ AnswerResult                                            │
    │   .text        最终答案文本                              │
    │   .sources     list[Source]   —— 编号↔URL 的权威映射     │
    │   .citations   list[Citation] —— 答案里每个 [n] 的位置    │
    │   .confidence  float          —— 来自 P0-2 校准聚合       │
    │   .low_evidence bool          —— abstention 信号          │
    │   .trace       list[dict]     —— 各步骤耗时（on_event 流） │
    │   .cache_hit / .used_tool / .layer_hits  —— 可观测元信息   │
    └─────────────────────────────────────────────────────────┘

关键设计决策
------------
1. **向后兼容优先**
   `AnswerResult.__str__` 返回 `.text`，且 `agent.chat()` 默认仍返回 `str`。
   只有显式传 `return_result=True` 才拿到 AnswerResult。
   这样 `scripts/search.py`、既有集成、单测全部无需修改。

2. **引用校验而非引用信任**
   `parse_citations()` 会解析答案里所有 `[n]`，并**逐个校验 n 是否落在
   有效 source id 范围内**。越界的编号被标为 `valid=False`，
   统计进 `invalid_citation_count`。这是从"提示模型要引用"迈向
   "系统保证引用可用"的第一步（完整的 span 级蕴含校验属 P2 的
   Citation Binder，本次先把数据通路和基础校验打通）。

3. **来源去重与排序**
   同一 URL 可能被多层同时命中（比如 L2 wiki 和 L4 web 都返回某页面）。
   `dedup_sources()` 按 URL 归并并保留最高置信度，避免来源面板出现重复卡片，
   也避免"5 家门户转载同一条新闻"制造虚假共识。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Optional


# ════════════════════════════════════════════════════════════════════════
#                              Source
# ════════════════════════════════════════════════════════════════════════
@dataclass
class Source:
    """一条可被引用、可被点击的来源。

    与 `evidence.build_evidence_block()` 产出的 `<doc id="n">` 严格一一对应，
    因此模型输出的 `[n]` 可以被系统精确解析并映射到 URL。

    Fields:
        id:          引用编号（从 1 开始，与 <doc id> 一致）。
        title:       标题（已过 sanitize，可安全渲染）。
        url:         原始链接；离线层（wiki chunk / KG / 历史）可能为空。
        domain:      域名，前端用于显示 favicon 与判断权威度。
        layer:       来自哪一层（"L2_wiki" / "L4_web" / …）。
        layer_label: 层的中文可读名（"维基百科" / "实时网页" / …）。
        confidence:  该条证据的**校准后**相关概率（P0-2），跨层可比。
        risks:       证据清洗时命中的风险标签（P0-4），如 ["injection"]。
                     前端应对含风险的来源给出视觉提示。
        snippet:     摘录片段，供来源卡片预览。
        cited:       本次回答是否**真的引用**了这条来源。
                     这个字段很有用：如果 10 条资料只有 2 条被引用，
                     说明检索精度偏低（可作为检索质量的在线指标）。
    """

    id: int
    title: str = ""
    url: str = ""
    domain: str = ""
    layer: str = ""
    layer_label: str = ""
    confidence: float = 0.0
    risks: list[str] = field(default_factory=list)
    snippet: str = ""
    cited: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        """从 `evidence.build_evidence_block()` 返回的 dict 构造。

        只取已知字段，忽略多余键，保证上下游可以独立演进。
        """
        return cls(
            id=int(d.get("id", 0)),
            title=str(d.get("title", "") or ""),
            url=str(d.get("url", "") or ""),
            domain=str(d.get("domain", "") or ""),
            layer=str(d.get("layer", "") or ""),
            layer_label=str(d.get("layer_label", "") or ""),
            confidence=float(d.get("confidence", 0.0) or 0.0),
            risks=list(d.get("risks") or []),
            snippet=str(d.get("snippet", "") or ""),
            cited=bool(d.get("cited", False)),
        )

    # ---- 前端展示辅助 ---- #
    @property
    def display_name(self) -> str:
        """来源卡片上显示的主标识：优先域名，其次标题，最后层名。"""
        return self.domain or self.title or self.layer_label or "未知来源"

    @property
    def is_clickable(self) -> bool:
        """是否可点击跳转（离线层如 KG / 历史问答没有 URL）。"""
        return self.url.startswith(("http://", "https://"))


# ════════════════════════════════════════════════════════════════════════
#                             Citation
# ════════════════════════════════════════════════════════════════════════
@dataclass
class Citation:
    """答案文本中的一处引用标记 `[n]`。

    Fields:
        source_id: 引用的 source 编号。
        start:     在答案文本中的起始字符下标（含）。
        end:       结束下标（不含）。前端可据此把 `[n]` 替换为可点击链接。
        raw:       原始标记文本，如 "[3]"。
        valid:     该编号是否指向一个**真实存在**的 source。
                   False 表示模型编造了不存在的编号 —— 这是常见的幻觉形式，
                   必须能被检测出来（否则用户点击时才发现是空的）。
        sentence:  该引用所在句子（截断），便于后续做 span 级蕴含校验（P2）。
    """

    source_id: int
    start: int
    end: int
    raw: str = ""
    valid: bool = True
    sentence: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ════════════════════════════════════════════════════════════════════════
#                        引用解析 / 校验
# ════════════════════════════════════════════════════════════════════════
# 匹配 [1] / [12] / [1,2] / [1, 2, 3] / [1-3] 等常见写法。
# 为什么要支持多种：不同模型的引用习惯不同，
# DeepSeek 常写 [1,2]，GPT 常写 [1][2]，Claude 有时写 [1-3]。
# 统一在这里归一化，避免上层为每个模型写适配。
_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*[-,，、]\s*\d+)*)\s*\]")

# 句子切分（用于给每个引用附上所在句子，供 P2 的蕴含校验）
_SENTENCE_SPLIT_RE = re.compile(r"[。！？；\n]|(?<=[.!?])\s+")


def _expand_ids(group: str) -> list[int]:
    """把 `[n]` 里的编号串展开为整数列表。

    支持：
        "3"       → [3]
        "1,2,3"   → [1, 2, 3]
        "1、2"    → [1, 2]（中文顿号）
        "1-3"     → [1, 2, 3]（区间；上限 20 防止 [1-9999] 之类异常输入）
    """
    ids: list[int] = []
    # 先按逗号/顿号切
    for part in re.split(r"[,，、]", group):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                lo, hi = (int(x) for x in part.split("-", 1))
                if 0 < lo <= hi and hi - lo < 20:   # 防御异常区间
                    ids.extend(range(lo, hi + 1))
                    continue
            except ValueError:
                pass
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return ids


def _sentence_at(text: str, pos: int, max_len: int = 160) -> str:
    """取 `pos` 所在的句子（用于引用的上下文记录）。"""
    if not text:
        return ""
    # 向前找最近的句子边界
    start = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text[:pos]):
        start = m.end()
    # 向后找最近的句子边界
    m = _SENTENCE_SPLIT_RE.search(text, pos)
    end = m.start() if m else len(text)
    return text[start:end].strip()[:max_len]


def parse_citations(
    text: str,
    sources: Iterable[Source],
) -> tuple[list[Citation], list[Source]]:
    """解析答案中的所有 `[n]` 引用，并校验编号有效性。

    这是 P0.5 的**校验环节**：把"模型声称的引用"变成"系统确认过的引用"。

    做三件事：
      1. 定位每个 `[n]` 的字符区间（前端据此渲染可点击链接）。
      2. 校验 n 是否落在真实存在的 source id 集合里 —— 越界即
         `valid=False`，说明模型编造了来源编号（常见幻觉）。
      3. 回写 `Source.cited`，让前端能区分"被引用的来源"与
         "检索到但未被使用的来源"，同时产出检索精度的在线指标。

    Args:
        text:    答案文本。
        sources: 本轮提供给模型的来源列表。

    Returns:
        (citations, sources)
        sources 是**同一批对象**（已就地更新 `cited` 标志），返回它只为
        方便链式调用。

    示例：
        >>> ss = [Source(id=1), Source(id=2)]
        >>> cs, ss = parse_citations("如上所述[1]，另见[5]。", ss)
        >>> [(c.source_id, c.valid) for c in cs]
        [(1, True), (5, False)]
        >>> ss[0].cited, ss[1].cited
        (True, False)
    """
    src_list = list(sources)
    valid_ids = {s.id for s in src_list}
    by_id = {s.id: s for s in src_list}

    citations: list[Citation] = []
    if not text:
        return citations, src_list

    for m in _CITATION_RE.finditer(text):
        for sid in _expand_ids(m.group(1)):
            is_valid = sid in valid_ids
            citations.append(Citation(
                source_id=sid,
                start=m.start(),
                end=m.end(),
                raw=m.group(0),
                valid=is_valid,
                sentence=_sentence_at(text, m.start()),
            ))
            if is_valid:
                by_id[sid].cited = True

    return citations, src_list


def dedup_sources(source_dicts: Iterable[dict]) -> list[Source]:
    """按 URL 归并重复来源，保留置信度最高的那一条。

    为什么必要：
      * 同一页面可能被多层同时命中（L2 wiki 的 chunk 与 L4 web 的结果
        指向同一 URL），来源面板会出现重复卡片；
      * 同一条新闻被多家门户转载时，虽然 URL 不同，但至少要先把
        **完全同 URL** 的合并掉（语义级近重去除属 P2 的 MMR 工作）。

    ⚠️ 注意：归并后会**重新编号**（id 从 1 连续）。因此本函数必须在
    `evidence.build_evidence_block()` **之前**调用，否则 `<doc id>` 与
    `Source.id` 会错位，引用映射就全乱了。当前 agent 里的调用顺序是：
        passages → build_evidence_block() → Source.from_dict()
    build_evidence_block 内部已经是"边构建边编号"，天然不会重复编号；
    本函数主要用于**没有走 evidence 模块**的兼容路径（如裸 web_search）。

    Args:
        source_dicts: dict 形式的来源列表。
    Returns:
        去重且重新连续编号的 Source 列表。
    """
    seen: dict[str, Source] = {}
    order: list[str] = []
    for d in source_dicts:
        s = Source.from_dict(d)
        # 无 URL 的（KG / 历史问答）用 layer+title 作为去重键
        key = s.url or f"{s.layer}::{s.title[:40]}"
        if key in seen:
            # 保留置信度更高的那份元信息
            if s.confidence > seen[key].confidence:
                s.id = seen[key].id           # 保持原编号不变
                seen[key] = s
            continue
        seen[key] = s
        order.append(key)

    out: list[Source] = []
    for i, key in enumerate(order, start=1):
        s = seen[key]
        s.id = i          # 重新连续编号
        out.append(s)
    return out


# ════════════════════════════════════════════════════════════════════════
#                            AnswerResult
# ════════════════════════════════════════════════════════════════════════
@dataclass
class AnswerResult:
    """一次 `agent.chat()` 的结构化返回值。

    ⚠️ 向后兼容：`__str__` 返回 `.text`，且 `agent.chat()` 默认仍返回裸 `str`。
    只有显式 `chat(..., return_result=True)` 才会拿到本对象。
    因此 `scripts/search.py` 等既有调用方**零改动**。

    Fields:
        text:         最终答案文本。
        sources:      本轮提供给模型的来源（含 cited 标志）。
        citations:    答案里解析出的 `[n]` 引用（含 valid 校验结果）。
        confidence:   整体证据置信度（P0-2 的 `aggregate_confidence`）。
                      0 表示无外部资料（闲聊 / NO_SEARCH）。
        low_evidence: 是否证据不足（P0-2 的 abstention 信号）。
                      True 时前端应提示"资料有限，回答可能不完整"。
        cache_hit:    是否走了 L1 缓存短路（毫秒级返回）。
        used_tool:    命中的专用工具名（天气 / 时间 / GitHub / arXiv），未用则 None。
        tool_failed:  P0-4：工具**调用失败并降级**到检索通路时为 True。
                      这个字段让上层能观测"工具挂了多少次"。
        layer_hits:   各层召回条数，如 `{"L2_wiki": 3, "L4_web": 5}`。
        rewritten:    改写后的检索 query（debug / 展示用）。
        trace:        各步骤事件列表（与 `on_event` 的 payload 同构）。
        elapsed_ms:   端到端总耗时。
        followups:    P2-3 追问推荐 —— 用户可能想继续问的 2~4 个问题。

                      ⚠️ 这些问题是从**答案文本里剥离**出来的（模型按
                      `followup.FOLLOWUP_MARKER` 分隔符输出），所以
                      `text` 一定是**已经剥干净**的正文 —— 前端直接渲染
                      `text` 不会看到分隔符或裸问题列表。

                      为空的正常情形：闲聊 / NO_SEARCH / 证据不足时
                      模型被要求不输出追问 / FOLLOWUP_MODE="off"。
                      空列表不代表故障，前端应静默不渲染该区块。
    """

    text: str = ""
    sources: list[Source] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    low_evidence: bool = False
    cache_hit: bool = False
    used_tool: Optional[str] = None
    tool_failed: bool = False
    layer_hits: dict[str, int] = field(default_factory=dict)
    rewritten: str = ""
    trace: list[dict] = field(default_factory=list)
    elapsed_ms: int = 0
    followups: list[str] = field(default_factory=list)

    # ---- 兼容：让 AnswerResult 在字符串场景下等价于 text ---- #
    def __str__(self) -> str:          # noqa: D105
        return self.text

    def __len__(self) -> int:          # noqa: D105
        return len(self.text)

    def __contains__(self, item: str) -> bool:   # noqa: D105
        """支持 `"关键词" in result`，方便既有断言/单测直接沿用。"""
        return item in self.text

    # ---- 派生指标 ---- #
    @property
    def cited_sources(self) -> list[Source]:
        """真正被答案引用的来源（前端来源面板的主列表）。"""
        return [s for s in self.sources if s.cited]

    @property
    def uncited_sources(self) -> list[Source]:
        """检索到但未被引用的来源（可折叠展示，也是检索精度指标）。"""
        return [s for s in self.sources if not s.cited]

    @property
    def invalid_citation_count(self) -> int:
        """答案里编造的、指向不存在编号的引用数。

        > 0 就意味着模型出现了"引用幻觉"。这个指标应该在线监控：
        持续 > 0 说明 prompt 里的编号约束不够强，或候选数与 prompt 不一致。
        """
        return sum(1 for c in self.citations if not c.valid)

    @property
    def citation_coverage(self) -> float:
        """被引用来源占全部来源的比例 ∈ [0,1]。

        偏低（如 < 0.3）说明检索召回了大量无用资料 ——
        既浪费 token，也增加模型被无关内容干扰的概率。
        可作为「检索精度」的在线代理指标。
        """
        if not self.sources:
            return 0.0
        return round(len(self.cited_sources) / len(self.sources), 4)

    @property
    def has_risky_source(self) -> bool:
        """是否有来源在清洗阶段命中了注入风险（P0-4）。"""
        return any(s.risks for s in self.sources)

    def to_dict(self) -> dict:
        """序列化（供 HTTP API / 日志 / scripts 输出 JSON）。"""
        return {
            "text": self.text,
            "sources": [s.to_dict() for s in self.sources],
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
            "low_evidence": self.low_evidence,
            "cache_hit": self.cache_hit,
            "used_tool": self.used_tool,
            "tool_failed": self.tool_failed,
            "layer_hits": self.layer_hits,
            "rewritten": self.rewritten,
            "trace": self.trace,
            "elapsed_ms": self.elapsed_ms,
            "followups": list(self.followups),
            # 派生指标一并输出，前端/监控无需重算
            "metrics": {
                "invalid_citations": self.invalid_citation_count,
                "citation_coverage": self.citation_coverage,
                "cited_count": len(self.cited_sources),
                "source_count": len(self.sources),
                "has_risky_source": self.has_risky_source,
            },
        }

    # ---- 渲染：CLI / Web 共用 ---- #
    def render_sources_markdown(self, *, only_cited: bool = False) -> str:
        """把来源渲染成 Markdown 面板（CLI 与 Web 共用一套实现）。

        输出示例：

            **来源**
            1. [量子计算 — zh.wikipedia.org](https://…) · 维基百科 · 0.88 ✓
            2. 某营销页 · 实时网页 · 0.41 ⚠️含可疑指令

        标记含义：
            ✓          被答案引用
            （无标记）   检索到但未被引用
            ⚠️          该来源在清洗阶段命中注入风险

        Args:
            only_cited: True 时只显示被引用的来源（更简洁）。
        """
        items = self.cited_sources if only_cited else self.sources
        if not items:
            return ""
        lines = ["**来源**"]
        for s in items:
            label = s.title or s.display_name
            head = f"[{label} — {s.domain}]({s.url})" if s.is_clickable else label
            marks = []
            if s.cited:
                marks.append("✓")
            if s.risks:
                marks.append("⚠️含可疑指令")
            tail = ("  " + " ".join(marks)) if marks else ""
            lines.append(
                f"{s.id}. {head} · {s.layer_label} · 置信度 {s.confidence:.2f}{tail}"
            )
        # 置信度与证据充分性提示
        foot = [f"_整体证据置信度: {self.confidence:.2f}_"]
        if self.low_evidence:
            foot.append("⚠️ _检索证据不足，回答可能不完整_")
        if self.invalid_citation_count:
            foot.append(
                f"⚠️ _检测到 {self.invalid_citation_count} 处无效引用编号_"
            )
        lines.append("")
        lines.append(" · ".join(foot))
        return "\n".join(lines)

    def render_followups_markdown(self) -> str:
        """把追问推荐渲染成 Markdown（P2-3；CLI 与 Web 共用）。

        实现委托给 `followup.render_followups_markdown()`，这里只做转发：
        渲染逻辑与解析逻辑放在同一个模块，改格式时不会漏改。
        followup 模块导入失败时返回空串（追问是锦上添花，不能拖垮渲染）。
        """
        if not self.followups:
            return ""
        try:
            from followup import render_followups_markdown
            return render_followups_markdown(self.followups)
        except Exception:
            return ""


# ════════════════════════════════════════════════════════════════════════
#                          StreamingAnswer
# ════════════════════════════════════════════════════════════════════════
class StreamingAnswer:
    """可迭代的流式答案包装器，迭代结束后 `.result` 可用（P0.5）。

    ════════════════════════════════════════════════════════════════════
    为什么需要这个类
    ════════════════════════════════════════════════════════════════════
    流式场景下有个时序矛盾：
      - 来源面板需要 `AnswerResult`（含 citations，而 citations 只能在
        **完整答案文本就绪后**才能解析）；
      - 但流式的本质就是"边生成边返回"，函数早就 return 了。

    最初想到的做法是给生成器对象挂一个 `.result` 属性，但 CPython 的
    generator 是 C 层实现、**没有 `__dict__`**：

        >>> def g(): yield 1
        >>> g().result = x
        AttributeError: 'generator' object has no attribute 'result'
                        and no __dict__ for setting new attributes

    所以必须用一个普通 Python 对象包一层。本类就是这个包装器：
      * 实现 `__iter__` / `__next__` → 对调用方而言与生成器完全等价，
        既有的 `for piece in result:` 代码零改动；
      * 实现 `close()` / `throw()` → 保留 CLI 的 Ctrl-C 打断语义
        （`main.py` 的 `_print_stream` 会调 `gen.close()`）；
      * 暴露 `.result` → 迭代结束后由内部生成器回填完整 AnswerResult。

    用法：
        stream = agent.chat(q, is_stream=True, return_result=True)
        for piece in stream:          # 与普通生成器一样用
            print(piece, end="")
        print(stream.result.render_sources_markdown())   # 结束后拿来源
    """

    def __init__(self, gen, result: Optional["AnswerResult"] = None):
        self._gen = gen
        # 先放一个占位结果，避免调用方在迭代完成前访问 `.result` 报错。
        # 占位里已经带上了 sources / confidence（它们在生成前就已确定），
        # 只有 text / citations 需要等生成结束才能填。
        self.result: "AnswerResult" = result or AnswerResult()

    # ---- 迭代协议：让本对象在使用上完全等价于生成器 ---- #
    def __iter__(self):
        return self

    def __next__(self) -> str:
        return next(self._gen)

    # ---- 生成器控制协议：保留打断语义 ---- #
    def close(self) -> None:
        """提前关闭底层生成器（CLI Ctrl-C / 前端断开时调用）。

        底层 `_gen()` 的 `finally` 块会被触发，从而执行
        "是否写记忆 / 是否归档" 的判定（见 agent.chat 的 save_on_interrupt）。
        """
        try:
            self._gen.close()
        except Exception:
            pass

    def throw(self, *args, **kwargs):
        """把异常抛进底层生成器（完整实现生成器协议）。"""
        return self._gen.throw(*args, **kwargs)

    def send(self, value):
        """转发 send（本场景不用，但补全协议以防调用方误用时静默出错）。"""
        return self._gen.send(value)


# ════════════════════════════════════════════════════════════════════════
#                              自检 / 演示
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":  # pragma: no cover
    print("=" * 76)
    print("引用解析与校验 parse_citations()")
    print("=" * 76)
    srcs = [
        Source(id=1, title="量子计算", url="https://zh.wikipedia.org/wiki/量子计算",
               domain="zh.wikipedia.org", layer="L2_wiki",
               layer_label="维基百科", confidence=0.88),
        Source(id=2, title="某营销页", url="https://spam.example.com/a",
               domain="spam.example.com", layer="L4_web",
               layer_label="实时网页", confidence=0.41, risks=["injection"]),
        Source(id=3, title="历史问答", layer="L3_history",
               layer_label="历史问答", confidence=0.55),
    ]
    answer = (
        "量子计算利用量子比特实现并行计算[1]。"
        "其核心优势在于对特定问题的指数级加速[1,3]。"
        "另有报道称相关产品已商用[7]。"          # ← 7 不存在，故意的
    )
    cits, srcs = parse_citations(answer, srcs)
    for c in cits:
        flag = "✓" if c.valid else "✗ 编号不存在"
        print(f"  [{c.source_id}] @{c.start}-{c.end} {flag}   句子: {c.sentence[:34]!r}")

    res = AnswerResult(
        text=answer, sources=srcs, citations=cits,
        confidence=0.9302, low_evidence=False,
        layer_hits={"L2_wiki": 3, "L3_history": 1, "L4_web": 5},
        rewritten="量子计算 原理 优势", elapsed_ms=4210,
    )
    print()
    print("=" * 76)
    print("派生指标（可直接接监控）")
    print("=" * 76)
    print(f"  被引用来源      : {[s.id for s in res.cited_sources]}")
    print(f"  未被引用来源    : {[s.id for s in res.uncited_sources]}")
    print(f"  无效引用数      : {res.invalid_citation_count}   ← >0 即引用幻觉")
    print(f"  引用覆盖率      : {res.citation_coverage}   ← 偏低说明检索精度差")
    print(f"  含风险来源      : {res.has_risky_source}")

    print()
    print("=" * 76)
    print("来源面板渲染 render_sources_markdown()")
    print("=" * 76)
    print(res.render_sources_markdown())

    print()
    print("=" * 76)
    print("向后兼容性验证")
    print("=" * 76)
    print(f'  str(res)[:20]          = {str(res)[:20]!r}')
    print(f'  len(res)               = {len(res)}')
    print(f'  "量子比特" in res       = {"量子比特" in res}')
    print(f'  f-string 插值           = {res!s:.18}…')
