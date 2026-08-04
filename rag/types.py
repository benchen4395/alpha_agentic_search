# rag/types.py
"""统一的数据结构定义，供各层与融合器使用。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

LayerName = Literal["L1_qa", "L2_wiki", "L3_history", "L4_web", "L5_kg"]


@dataclass
class Passage:
    """一段可被 LLM 引用的检索结果。

    Fields:
        text     : 主体文本（喂给 LLM 的内容）
        title    : 标题（可选，便于生成 citation）
        url      : 出处（可选）
        score    : 该层内部相似度/相关性分数（未归一，仅供本层排序）
        layer    : 来自哪一层
        metadata : 额外元数据（entity_id、pageviews、等）
    """
    text: str
    title: str = ""
    url: str = ""
    score: float = 0.0
    layer: LayerName = "L4_web"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "title": self.title,
            "url": self.url,
            "score": self.score,
            "layer": self.layer,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    """LayeredRetriever.retrieve 的返回值。

    Fields:
        query      : 原始查询
        passages   : 融合、rerank 后的 top-k 段落
        layer_hits : 每层各返回了几条（用于 debug / 命中统计）
        cache_hit  : L1 是否精准 / 模糊命中（若命中，answer 会直接可用）
        cache_answer: 若 L1 命中，此处直接是最终答案，可以短路

        —— P0-2 新增（跨层分数校准的产出）——
        confidence   : 整体证据置信度 ∈ [0,1]。
                       由 `rag.calibration.aggregate_confidence()` 计算：
                       各层原始分先校准成 P(relevant)，再用噪声-OR 聚合
                       top-3。这是**唯一**应该用来做阈值判定的量，
                       因为它跨层可比（原始 score 不可比，见 calibration.py）。
        low_evidence : 是否证据不足（confidence < ABSTAIN_CONFIDENCE）。
                       为 True 时 agent 会在 prompt 里加一条提示，
                       让模型明确说"资料不足"而不是基于无关资料臆测。
        web_fallback : 本轮是否触发了 L4 兜底（可观测指标：
                       这个比例突然升高通常意味着离线索引覆盖度下降）。

        —— Stage-1 新增（证据可答性信号）——
        term_coverage : query 实词在离线证据里的**覆盖率** ∈ [0,1]。
                       与 `confidence` **正交**：confidence 看语义相似度，
                       这个看关键词是不是真的出现了。
                       实测「小丑鱼 外来物种 USGS 邮编」：
                       confidence=0.98 但 term_coverage=0.20
                       —— 证据主题相关却完全不含答案。
                       详见 `rag/answerability.py`。
        missing_terms : 未在证据里出现的实词列表（排查用）。
                       它直接告诉你"缺什么" —— 比一个光禿禿的
                       置信度数字有用得多。
    """
    query: str
    passages: list[Passage] = field(default_factory=list)
    layer_hits: dict[str, int] = field(default_factory=dict)
    cache_hit: bool = False
    cache_answer: Optional[str] = None
    confidence: float = 0.0
    low_evidence: bool = False
    web_fallback: bool = False
    # Stage-1：默认 1.0 / 空列表 —— 表示"无异常"。
    # 这样 L1 命中、工具短路等不走覆盖率判定的路径不会被误认为"证据不足"。
    term_coverage: float = 1.0
    missing_terms: list[str] = field(default_factory=list)

    def as_context_block(self, max_len: int = 8000) -> str:
        """把 passages 拼成可直接喂给 LLM 的文本（**旧版纯文本格式**）。

        ⚠️ P0-4 起，主链路已改用 `evidence.build_evidence_block()`
        —— 它会把每段包进 `<doc id="n">` 定界标签、做 injection 清洗，
        并同时产出 sources 列表供引用归因（P0.5）。

        本方法**保留**是为了：
          1. 向后兼容任何直接调用它的外部代码 / 脚本；
          2. 在 `enable_evidence_guard=False`（做 A/B 对比时）作为对照组。
        新代码请优先用 `evidence.build_evidence_block(result.passages)`。
        """
        if not self.passages:
            return "(无外部资料)"
        buf: list[str] = []
        total = 0
        for i, p in enumerate(self.passages, 1):
            header = f"[{i}] {p.title or p.layer}"
            if p.url:
                header += f"\nURL: {p.url}"
            block = f"{header}\n{p.text}"
            if total + len(block) > max_len:
                break
            buf.append(block)
            total += len(block)
        return "\n\n".join(buf)
