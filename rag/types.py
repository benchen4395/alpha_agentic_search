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
    """
    query: str
    passages: list[Passage] = field(default_factory=list)
    layer_hits: dict[str, int] = field(default_factory=dict)
    cache_hit: bool = False
    cache_answer: Optional[str] = None

    def as_context_block(self, max_len: int = 8000) -> str:
        """把 passages 拼成可直接喂给 LLM 的文本。"""
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
