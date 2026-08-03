# rag/router.py
"""Router：根据 query 决定激活哪些层。

策略：
    - offline_only : 只走 L1 + L2 + L3 + L5（不联网）
    - web_only     : 只走 L1 + L4
    - hybrid       : L1 → 若命中直接短路；否则并行 (L2, L3, L5)，
                     若整体证据置信度不够高再补 L4（推荐，默认）

未来可以在此接入 LLM Router（或轻量判别式分类器）；当前用规则即可。

P0-2 改造要点
-------------
1. **时效判定去重 + 年份动态化**
   原实现把时效词典硬编码在本文件，且写死了 "2024"/"2025"/"2026"——
   到 2027 年这套判断就完全失效（且会把已成历史的年份继续误判为时效敏感）。
   现在统一复用 `cache_policy.is_time_sensitive()`：那里的年份按
   「当前年份 ±1」**动态生成**，且区分强/弱时效信号。
   好处：写入侧（L1 准入）与检索侧（L4 激活）用**同一套判定**，
   不会出现"L1 认为不时效所以缓存了，Router 却认为时效要联网"的自相矛盾。

2. **兜底判定改用校准后的聚合置信度**
   `should_fallback_to_web()` 的入参语义从「离线层最高原始分」
   变为「校准后的聚合置信度 conf」。原因见 rag/calibration.py 顶部说明：
   拿四个不同量纲的原始分与单一标量阈值比较，在统计上没有意义。
"""
from __future__ import annotations

from typing import Iterable

# P0-2：复用 cache_policy 的时效判定（年份动态生成、强/弱信号区分），
# 避免本文件与 cache_policy 各维护一份词典导致行为不一致。
from cache_policy import is_time_sensitive as _policy_is_time_sensitive

from . import config as rag_config
from .types import LayerName


def is_time_sensitive(query: str) -> bool:
    """判断 query 是否时效敏感（→ 直接叠加 L4 实时联网）。

    实现委托给 `cache_policy.is_time_sensitive()`，那里：
      - 年份按「当前年份 ±1」动态生成，不会随时间失效；
      - 区分强时效信号（今天/最新/股价/天气 → 必须联网）与
        弱易变信号（具体年份/版本号 → 交由 L1 短 TTL 处理，不强制联网）。

    保留本函数作为薄封装，是为了不破坏既有 import 路径
    （`from rag.router import is_time_sensitive`）。
    """
    return _policy_is_time_sensitive(query)


def route(query: str, strategy: str = "") -> list[LayerName]:
    """返回本次要激活的层名列表（不含 L1，L1 由 Retriever 单独短路）。"""
    strategy = strategy or rag_config.ROUTER_STRATEGY

    if strategy == "offline_only":
        return ["L2_wiki", "L3_history", "L5_kg"]
    if strategy == "web_only":
        return ["L4_web"]

    # hybrid（默认）：时间敏感 query 直接叠加 L4
    layers: list[LayerName] = ["L2_wiki", "L3_history", "L5_kg"]
    if is_time_sensitive(query):
        layers.append("L4_web")
    return layers


def should_fallback_to_web(confidence: float) -> bool:
    """当离线层的**校准聚合置信度**低于阈值时，追加 L4。

    Args:
        confidence: `rag.calibration.aggregate_confidence()` 的输出，
                    语义为"至少有一条真正相关的离线证据"的概率 ∈ [0,1]。
                    ⚠️ 这**不再是**原来的"最高原始分"——原始分跨层不可比，
                    详见 rag/calibration.py 顶部的两个 bug 说明。

    Returns:
        True 表示应该补 L4 web 检索。
    """
    return confidence < rag_config.WEB_FALLBACK_CONFIDENCE


def should_abstain(confidence: float) -> bool:
    """判断是否应当"承认资料不足"而不是硬编答案（P0-2 → P0.5 用）。

    即使补了 L4，整体置信度仍然极低（默认 < 0.30）时，说明本轮检索
    确实没找到相关资料。这时正确做法是让 summary 阶段明确说明信息不足，
    而不是基于无关资料生成看似合理的内容——后者是幻觉的主要来源之一。

    返回值会通过 `RetrievalResult.low_evidence` 传给 agent，
    再由 agent 注入到 prompt（并在 P0.5 的 AnswerResult.confidence 里体现）。
    """
    return confidence < rag_config.ABSTAIN_CONFIDENCE
