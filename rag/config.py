# rag/config.py
"""RAG 系统配置（分层记忆 / 融合 / 增量索引）。

集成说明（本次改造）
--------------------
* **Embedding 统一（D1）**：全库统一到 FlagEmbedding BGE-M3，默认模型名
  ``BAAI/bge-m3``（供 :class:`rag.embedder.Embedder` 使用）。
* **数据路径（D3）**：所有 RAG 数据统一放到项目根下的 ``data/rag_data/``。
  L2/L5 的真实索引文件路径由 ``rag/configs/default.yaml`` 定义，并通过
  :func:`get_wiki_rag_config` 读取（已锚定到项目根）。本文件仅保留 L1/L3
  等「rag 编排器自身」用到的路径与检索超参。
* **CN-DBpedia 已删除**：只用 Wikipedia dump 构建 L2，不再有任何 cndb 配置。

设计取向：环境变量优先级高于此处默认值，便于生产环境覆盖。
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

# ============== Embedding 模型（D1：统一 FlagEmbedding BGE-M3） ==============
# 注意：这里从旧的 "bge-m3"（ollama tag）改为 HuggingFace 仓库名 "BAAI/bge-m3"，
# 因为底层编码器已切换为 FlagEmbedding.BGEM3FlagModel。
RAG_EMBED_MODEL: str = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")
RAG_EMBED_DIM: int = int(os.getenv("RAG_EMBED_DIM", "1024"))   # bge-m3 = 1024 维
# 编码设备：auto -> CUDA > MPS > CPU；也可显式 "cuda:0" / "cpu"
RAG_EMBED_DEVICE: str = os.getenv("RAG_EMBED_DEVICE", "auto")

# ============== 存储路径（D3：统一 data/rag_data/） ==============
# 项目根 = 本文件上一级（rag/）的上一级
_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_RAG_DIR)
# RAG 知识库数据根。作用：存放 L2 Wiki 索引/chunks、L5 KG sqlite/热门向量、
# L3 历史增量归档等所有 RAG 相关落盘文件。
# 位置：<项目根>/data/rag_data/（与 L1 qa_cache、search_cache 同处 data/ 下，风格统一）。
# 可用环境变量 RAG_DATA_DIR 覆盖；若上层设置了 DATA_DIR，这里默认仍指向 data/rag_data。
_DEFAULT_DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
RAG_DATA_DIR: str = os.getenv("RAG_DATA_DIR", os.path.join(_DEFAULT_DATA_DIR, "rag_data"))

# L3 历史归档（rag 编排器自身维护的增量 FAISS，越用越强）
L3_HISTORY_INDEX_DIR: str = os.path.join(RAG_DATA_DIR, "l3_history")

# 说明：
#   - L2 wiki 索引 / chunks、L5 KG sqlite / 热门向量 等大文件的真实路径
#     统一由 rag/configs/default.yaml 管理（见 get_wiki_rag_config）。
#   - 不再有 L2_CNDBPEDIA_INDEX_DIR（CN-DBpedia 已彻底移除）。

# ============== 检索参数 ==============
# 每一层召回的候选数
L1_TOP_K: int = 1        # 命中即返回
L2_TOP_K: int = int(os.getenv("RAG_L2_TOP_K", "8"))
L3_TOP_K: int = int(os.getenv("RAG_L3_TOP_K", "5"))
L4_TOP_K: int = int(os.getenv("RAG_L4_TOP_K", "5"))
L5_TOP_K: int = int(os.getenv("RAG_L5_TOP_K", "3"))

# Fusion 融合后返回的最终段落数
FUSION_TOP_K: int = int(os.getenv("RAG_FUSION_TOP_K", "6"))

# RRF 常数
RRF_K: int = 60

# ============== Rerank 策略（D5：默认 RRF 零依赖） ==============
# 可选值："rrf"（默认，零依赖）| "bge"（BGE 交叉编码器）| "cascade"（RRF 粗排 → BGE 精排）| "none"
# 通过环境变量 RAG_RERANK_STRATEGY 切换；BGE/cascade 需要额外安装 FlagEmbedding reranker。
# 兼容旧开关：RAG_ENABLE_RERANK=true 等价于把策略切到 bge（仅当未显式指定 strategy 时）。
def _resolve_rerank_strategy() -> str:
    s = os.getenv("RAG_RERANK_STRATEGY", "").lower()
    if s:
        return s
    if os.getenv("RAG_ENABLE_RERANK", "").lower() == "true":
        return "bge"
    return "rrf"

RERANK_STRATEGY: str = _resolve_rerank_strategy()
RERANK_MODEL: str = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# ============== L5 KG 多跳 ==============
# 是否启用知识图谱多跳扩展（BFS）。默认关闭以控延迟，可用环境变量开启。
KG_MULTI_HOP: bool = os.getenv("RAG_KG_MULTI_HOP", "false").lower() == "true"
KG_MAX_HOPS: int = int(os.getenv("RAG_KG_MAX_HOPS", "2"))

# ============== 增量索引 worker（L3） ==============
INCR_QUEUE_MAXSIZE: int = 4096
INCR_BATCH_SIZE: int = 32
INCR_FLUSH_INTERVAL_SEC: float = 5.0

# ============== Router（层激活） ==============
# 默认策略：并联 L2 + L3 + L5，若 top1 分数不够高再走 L4
ROUTER_STRATEGY: str = os.getenv("RAG_ROUTER_STRATEGY", "hybrid")

# ============== 分层检索的延迟预算（deadline） ==============
#
# ⚡ 为什么需要它：**慢层不能拖垮整条链路**
#
# `_parallel_search()` 用 `as_completed` 等待**所有**激活层返回，
# 所以整层检索的耗时 = max(各层耗时)。一旦某层退化，全链路一起退化。
#
# 实测数据（本机）：
#     L2 wiki   :   65 ms   （BGE-M3 编码 64ms + FAISS 1ms）
#     L3 history:   66 ms
#     L5 kg     :  434 ms
#     L4 web    : 16412 ms ~ 37957 ms   ← 比离线层慢 2~3 个数量级！
#
# L4 走 DuckDuckGo，而 `searcher._ddg` 内部还有 `DDG_MAX_RETRIES=3`
# 次重试，每次 `DDG_TIMEOUT=15s`，失败之间还 `sleep(1.0~2.5s)` ——
# 最坏情况单是 DDG 就能耗掉 ~50s，之后还要继续尝试
# Tavily / Serper / Bing。理论最坏值远超一分钟。
#
# 【业界做法】Perplexity / Bing Chat 这类产品对每一路召回都设**硬预算**：
# 到点就用"已经拿到的证据"作答，而不是无限等待。理由很直接：
#   - 用户对"5 秒内给出 90 分答案"的满意度远高于"40 秒给 95 分"；
#   - 而且在本例中离线证据的置信度已经 0.99，等这 30 秒**一分不涨**。
# 这就是 tail-latency 治理里的 "hedging / deadline budget" 模式。
#
# 【本实现】给每层一个 `LAYER_TIMEOUT_SEC` 预算：
#   到点未返回的层，其 future 被放弃（结果视为空），
#   其余层的结果照常融合。降级是**优雅的**——少一路召回而已，
#   而不是整个请求卡死或报错。
#
# 取值依据：离线层实测最慢 434ms，给 10 倍余量到 5s 足够宽松
# （即使冷启动 mmap 缺页也够）；而 L4 单独给更长的 8s ——
# 联网本来就慢，但 8s 已能覆盖绝大多数正常响应的搜索请求。
LAYER_TIMEOUT_SEC: float = float(os.getenv("RAG_LAYER_TIMEOUT", "5.0"))
L4_TIMEOUT_SEC: float = float(os.getenv("RAG_L4_TIMEOUT", "8.0"))

# ============== L4 兜底 / abstention 阈值（P0-2 改造） ==============
#
# ⚠️ 重要变更：判定口径从「原始分」改为「校准后的聚合置信度」。
#
# 【改造前】
#   WEB_FALLBACK_THRESHOLD = 0.55，与 `max(各层原始 p.score)` 比较。
#   两个致命问题：
#     ① L5 的 `or 0.9` bug 让 offline_best 恒为 0.9 → L4 永不触发；
#     ② L2 是 BGE 余弦、L4 是位次衰减、L5 是人工混合分——用一个标量
#        阈值裁决四种不同量纲的分数，统计上没有意义。
#
# 【改造后】
#   各层原始分先经 `rag/calibration.py` 映射到统一的 P(relevant)，
#   再用噪声-OR 聚合成整体置信度 `conf ∈ [0,1]`，然后：
#       conf < WEB_FALLBACK_CONFIDENCE  → 补 L4 web
#       conf < ABSTAIN_CONFIDENCE       → 标记证据不足（供 P0.5 前端提示 /
#                                          让 summary 明确说"资料不足"）
#
# 阈值取值依据（对照 calibration.py 的默认参数）：
#   0.55 ≈ 「单条 L2 余弦 0.57」或「单条 web top-2」的水平。
#   低于它说明离线三层都没给出足够相关的证据，值得花 1-3s 去联网。
WEB_FALLBACK_CONFIDENCE: float = float(os.getenv("RAG_WEB_FALLBACK_CONF", "0.55"))

#   0.30 ≈ 「单条 L2 余弦 0.47」的水平，属于"基本不相关"。
#   低于它即使补了 L4 也没救回来，应该让模型明确表示信息不足，
#   而不是硬着头皮基于无关资料编答案（这是幻觉的主要来源之一）。
ABSTAIN_CONFIDENCE: float = float(os.getenv("RAG_ABSTAIN_CONF", "0.30"))

# —— 向后兼容：保留旧名字 ——
# 老代码 / 外部脚本可能还在读 WEB_FALLBACK_THRESHOLD 与
# should_fallback_to_web(top_score)。保留别名避免破坏性变更，
# 但内部主链路已切换到 WEB_FALLBACK_CONFIDENCE。
WEB_FALLBACK_THRESHOLD: float = float(
    os.getenv("RAG_WEB_FALLBACK", str(WEB_FALLBACK_CONFIDENCE))
)


# ============== wiki_rag YAML 配置桥接 ==============
@lru_cache(maxsize=1)
def get_wiki_rag_config() -> dict[str, Any]:
    """加载 vendored 的 wiki_rag YAML 配置（含 L2/L5 全部真实文件路径）。

    路径已在 :func:`rag.wiki_rag.config.load_config` 中锚定到项目根，
    因此这里拿到的 ``cfg["paths"][...]`` 都是绝对 Path，指向
    ``agentic_search/data/rag_data/`` 下的数据文件。

    用 lru_cache 保证进程内只读一次 YAML。
    """
    from .wiki_rag.config import load_config

    return load_config()
