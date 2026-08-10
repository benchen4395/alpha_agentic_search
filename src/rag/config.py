# rag/config.py
"""RAG 系统配置（分层记忆 / 融合 / 增量索引）。

集成说明
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
# src/rag/ → src/ → 项目根（同 configs/config.py，见那里的说明）
PROJECT_ROOT = os.path.dirname(os.path.dirname(_RAG_DIR))
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

# ============== 并列实体配额（quota fusion） ==============
#
# ⚡ 为什么需要它：**强实体会把弱实体饿死**（winner-takes-most）
#
# 实测故障（用户报告）：
#     提问「国庆期间，俄罗斯、希腊、巴厘岛的气候和景色分别如何？」
#     融合后 6 段的实体分布 {'俄罗斯': 4, '希腊': 1, '巴厘岛': 1}
#     模型回答「希腊完全没有相关资料」
#
# 希腊的证据**其实检索到了**，是在融合阶段被 RRF 挤掉的：RRF 只有
# 「全局相关度」一个维度，它不知道这条 query 有 3 个**并列**对象、
# 每个都必须有料才答得出来。于是资料更丰富的那个实体独占 FUSION_TOP_K。
#
# 【为什么不是"检索次数不够"】
# 我实测过：只把 query 拆成 3 个子 query 并发搜、不改融合，
# 饥饿只是**换了个实体**：
#     单次长 query :  俄罗斯 1 / 希腊 0 / 巴厘岛 3
#     拆 3 个子query:  俄罗斯 0 / 希腊 1 / 巴厘岛 3   ← 俄罗斯反而归零
# 6 个席位分给 3 个对象，没有配额约束照样有人归零。
# 所以配额是**前置**修复，多 query 并发是它之上的可选增强。
#
# 【取值依据】默认 2：
#   1 段往往只够证明"这个实体存在"，不够回答"它的气候和景色如何"
#   这种双维度问题；2 段是实测下来"够说一句有内容的话"的下限。
#   当 FUSION_TOP_K // 实体数 < 该值时，`quota_fuse` 会自动下调到
#   至少 1 —— 宁可每个实体只有 1 段，也不要有实体是 0 段：
#   0 段会让模型说"完全没有资料"，1 段至少能说点什么。
FUSION_MIN_PER_ENTITY: int = int(os.getenv("RAG_FUSION_MIN_PER_ENTITY", "2"))

# 融合阶段的常态调试日志开关（默认关）。
#
# 打开后 `quota_fuse` 会打印每次多实体融合的配额分布，调参时很直观。
# 默认关闭的原因：每个多实体 query 都打一行会污染生产日志，且 print
# 持有 stdout 锁，高 QPS 下是可测量的开销。
# 注意：「某实体 0 段证据」的**告警**不受此开关控制，始终打印 ——
# 那是需要人工关注的异常信号，不是调试信息。
VERBOSE_FUSION: bool = os.getenv("RAG_VERBOSE_FUSION", "false").lower() in (
    "1", "true", "yes", "on",
)

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

# ============== 近重去重 + MMR 多样性 ==============
#
# ⚡ 为什么需要它：**证据席位被「同一份信息」占满**
#
# FUSION_TOP_K=6 是喂给 LLM 的证据预算，很小很贵。而 `fusion._passage_key`
# 的去重只做**精确**匹配（url 全等，或 title[:40]+text[:80] 全等），
# 挡不住真实世界最常见的重复 —— **同一条新闻被 N 家转载**：
#
#     新华网《我国上半年GDP同比增长5.3%》 https://news.cn/...
#     人民网《我国上半年GDP同比增长5.3%》 https://people.cn/...
#     央视网《上半年GDP同比增长5.3%》     https://cctv.com/...
#
# url 各不相同、正文前 80 字也常因导语改写而不同 → 全部通过去重，
# 吃掉 6 个席位里的 3~4 个。后果有三层：
#   ① 信息量坍缩：本该进来的补充视角（分季度拆解、同期对比）被挤掉；
#   ② **虚假共识**：模型因"多个来源一致"提高确信度，但这些来源同源，
#      独立性是假的 —— 这是幻觉的一种隐性来源；
#   ③ 来源面板出现 4 条几乎一样的条目，体验很差。
#
# 【为什么不能只按域名去重】域名相同 ≠ 内容相同：维基同一实体的不同
# 小节、政府站的「政策原文 + 答记者问」、大站的系列专题报道，都是同
# 域名但内容互补。要判定的是"讲的是不是同一件事"，这是语义问题，
# 只能用向量余弦解。域名只作辅助信号（详见 rag/dedup.py）。
#
# 【两个阶段职责不同，顺序不可颠倒】
#   阶段 A 近重去重：阈值 0.95 硬删 —— **事实判断**（是不是同一份文本）
#   阶段 B MMR 重排：λ 权衡软排 —— **偏好判断**（在有差异的候选里偏好互补）
#   必须先 A 再 B：MMR 的惩罚是连续的，对余弦 0.98 的转载只会"较大惩罚"，
#   若其相关性也高仍可能入选。硬删必须在前面，否则转载稿会参与多样性
#   计算、污染整个选择过程。
ENABLE_NEAR_DUP: bool = os.getenv("RAG_ENABLE_NEAR_DUP", "true").lower() == "true"
ENABLE_MMR: bool = os.getenv("RAG_ENABLE_MMR", "true").lower() == "true"

# 实验操作分析:
# export RAG_ENABLE_NEAR_DUP=false        # 关硬去重
# export RAG_ENABLE_MMR=false             # 关 MMR  -- 注: 关闭可以去掉90ms的向量耗时
# export RAG_DEDUP_LAYERS=""              # 改为全量模式（更彻底但更慢）
# export RAG_FUSION_CANDIDATE_MULT=2      # 候选池 3 倍→2 倍
# export RAG_ENABLE_SNIPPET_CLEAN=false   # 关噪声清洗
# export FOLLOWUP_MODE=off                # 关追问推荐
# export ENABLE_CLARIFY=false             # 澄清提问（本来就默认关）

# 候选池放大倍数。
#
# 为什么必须放大：去重会**删掉**段落。如果只召回 FUSION_TOP_K=6 就去重，
# 删掉 3 条转载后只剩 3 段 —— 席位没有被"腾给更好的证据"，而是**凭空
# 消失**了。那样去重反而降低了信息量，完全违背初衷。
#
# 正确做法是「多召回 → 去重 → 再截断到 top_k」：
#     召回 18 段 → 去掉 5 段转载 → 剩 13 段 → MMR 选出最互补的 6 段
# 这样 6 个席位装的是 6 份**不同**的信息，而不是 3 份信息 + 3 份复制。
#
# 取 3 倍的依据：实测 L4 web top-5 里同一事件的转载通常 2~3 条，
# 4 层加起来最坏情况约一半是近重。3 倍（18 段）留出足够余量，
# 且额外成本只是多编码 12 段向量（BGE-M3 批量约 +30ms）。
FUSION_CANDIDATE_MULTIPLIER: int = int(
    os.getenv("RAG_FUSION_CANDIDATE_MULT", "3")
)

# ------------------------------------------------------------------------
# 只对指定层做语义去重（延迟优化：方案 B）
# ------------------------------------------------------------------------
#
# ⚡ 为什么需要它：**唯一的真实开销是向量编码，而它随段数近线性增长**
#
# 实测（本机 BGE-M3 / MPS，中位数）：
#      3 段  491 ms |  5 段  675 ms |  8 段  767 ms
#     13 段 1241 ms | 16 段 1501 ms | 18 段 1691 ms
#     → 约 80 ms/段 + 约 250 ms 固定开销
#
# 而算法本身几乎免费：近重去重 1.25 ms、MMR 0.80 ms。
# 也就是说 **99%+ 的耗时都在编码**，要降延迟只有一条路：**少编码几段**。
#
# 【为什么可以只挑 L4】转载重复**几乎只发生在 web 检索**：
#   L4 web    —— 同一新闻被新华网/人民网/央视网转载 ← 转载问题的全部来源
#   L2 wiki   —— 同一实体的不同小节，内容互补
#   L3 history—— 用户自己的历史问答，天然不重复
#   L5 kg     —— 结构化三元组，本身已去重
# 把语义去重限定在 L4，用约 1/3 的编码量拿到绝大部分收益。
#
# 【收益测算】按各层 top_k（L2=8 L3=5 L4=5 L5=3）：
#   纯离线命中（L4 未触发）：目标层 0 段 → **零编码，省 100%**
#     ↑ 这也是最常见的场景，且此时本来就没有转载问题
#   L4 兜底触发：只编码 L4 的 5 段 → 约 675 ms 而非 1691 ms
#   全层命中：18 段里只挑 L4 的 5 段 → 同上
#
# 【代价（这是个妥协，必须写清楚）】
#   * 跨层近重挡不住：若 L2 wiki 与 L4 web 各自命中同主题的不同页面且
#     语义高度重合，不会被删。缓解：url 完全相同的情况
#     `fusion._passage_key` 的精确去重已覆盖；真正漏掉的是"不同 url
#     但内容近重"的跨层组合，实测远比 L4 内部的转载罕见。
#   * MMR 只在目标层内生效：非目标层按原 RRF 顺序保留。
#
# 【如何回到全量模式】设为空串即可（更彻底但更慢）：
#     export RAG_DEDUP_LAYERS=""
# 也可自定义，逗号分隔，例如同时覆盖 L4 与 L3：
#     export RAG_DEDUP_LAYERS="L4_web,L3_history"
#
# 取值必须与 `rag/types.py` 的 LayerName 字面量一致，
# 否则会静默匹配不上（表现为"去重没生效"，且不会报错）。
def _parse_dedup_layers() -> frozenset[str]:
    """解析 RAG_DEDUP_LAYERS。空串 → 空集合（全量模式）。

    这里用函数而不是一行表达式，是为了让"未设置"与"显式设为空串"
    这两种语义区分开：
        未设置        → 用默认值 {"L4_web"}（分层模式）
        显式设为 ""   → 空集合（全量模式）
    若写成 `os.getenv(k, "L4_web").split(",")`，显式空串会得到 [""]，
    是个包含空字符串的集合，永远匹配不上任何 layer —— 效果等同于
    "彻底关掉去重"，而不是用户想要的"对所有层去重"，语义正好相反。
    """
    raw = os.getenv("RAG_DEDUP_LAYERS")
    if raw is None:
        return frozenset({"L4_web"})
    return frozenset(x.strip() for x in raw.split(",") if x.strip())


DEDUP_LAYERS: frozenset[str] = _parse_dedup_layers()

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

# ============== 超时宽限期（修复③：软放弃） ==============
#
# ⚡ 为什么需要它：**成本已经付了，别把结果扔掉**
#
# 上面的 deadline 机制有一个真实的浪费。用户实测日志里最刺眼的一幕：
#
#     [retriever] ⏱️ 层检索超预算 8.0s，放弃未返回的层 ['L4_web']
#     [searcher] DDG 命中 5 条            ← 仅 0.7 秒后就返回了！
#
# L4 只超时 0.7 秒，DuckDuckGo 真的召回了 5 条**有效**证据，却因为
# "过了截止线"被直接丢弃 —— 那一题（日本今年地震次数）本来是三个
# 失败案例里唯一有机会答对的。
#
# 更关键的是：`ThreadPoolExecutor` **无法中断已启动的任务**，所以那个
# 线程一定会跑完并把结果放进 future。也就是说，8 秒的等待成本已经
# 完全付掉了，结果却没人去取。这是最坏的一种结果 —— 付了钱不拿货。
#
# 【业界做法】Bing / Perplexity 的 deadline 不是一刀切的硬截断，而是
# 「主预算 + 短宽限」两段式：主预算到点后再快速看一眼，已经完成的
# 顺手收下，仍未完成的才真正放弃。这样：
#   * 最坏前台延迟 = 主预算 + 宽限期（仍然可预测、有上界）；
#   * 而"差一点就成功"的请求被救回来，召回率明显改善。
#
# 【取值依据】宽限期必须**远小于**主预算，否则等于变相放宽 deadline、
# 前台延迟失控。1.0~1.5s 的量级刚好覆盖"网络抖动导致的临界超时"
# （实测那次只差 0.7s），又不会让用户明显感知到额外等待。
LAYER_GRACE_SEC: float = float(os.getenv("RAG_LAYER_GRACE", "1.0"))
L4_GRACE_SEC: float = float(os.getenv("RAG_L4_GRACE", "1.5"))

# ============== L4 兜底 / abstention 阈值（改造） ==============
#
# ⚠️ 重要变更：判定口径从「原始分」改为「校准后的聚合置信度」。
#
# 朴素做法及其问题
#   WEB_FALLBACK_THRESHOLD = 0.55，与 `max(各层原始 p.score)` 比较。
#   两个致命问题：
#     ① L5 的 `or 0.9` bug 让 offline_best 恒为 0.9 → L4 永不触发；
#     ② L2 是 BGE 余弦、L4 是位次衰减、L5 是人工混合分——用一个标量
#        阈值裁决四种不同量纲的分数，统计上没有意义。
#
# 做法
#   各层原始分先经 `rag/calibration.py` 映射到统一的 P(relevant)，
#   再用噪声-OR 聚合成整体置信度 `conf ∈ [0,1]`，然后：
#       conf < WEB_FALLBACK_CONFIDENCE  → 补 L4 web
#       conf < ABSTAIN_CONFIDENCE       → 标记证据不足（供 前端提示 /
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
