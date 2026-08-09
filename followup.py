# followup.py
"""追问推荐 + 澄清提问 —— ════════════════════════════════════════════════════════════════════════
两个功能，一个模块，但解决的是**方向相反**的两个问题
════════════════════════════════════════════════════════════════════════
放在一个模块里是因为它们共享"query 意图分析"这套基础设施，
但产品语义完全相反，务必分清：

  【追问推荐 / related questions】—— 回答**之后**，向前推进
      答完了，猜用户下一步还想知道什么，给 2~4 条可点击的问题。
      这是 Perplexity 体验里最"廉价高感知"的一环：
      实现成本极低，但让产品从"问答工具"变成"探索伙伴"。
      失败代价：推荐不准 → 用户不点，**无任何损失**。

  【澄清提问 / clarification】—— 回答**之前**，暂停确认
      发现 query 本身有歧义，与其猜错方向答一大堆，
      不如反问一句。典型："苹果多少钱"（水果还是手机？）
      失败代价：**很高** —— 该答的不答、反而反问，
      用户会觉得这助手很笨。所以判定必须**极度保守**。

这个代价不对称是本模块所有设计取舍的根源：
    追问推荐  → 宁可多给，反正不点就没损失
    澄清提问  → 宁可不问，只在**证据明确显示分裂**时才触发

════════════════════════════════════════════════════════════════════════
为什么澄清判定要基于「检索证据」而不是「query 文本」
════════════════════════════════════════════════════════════════════════
最直觉的实现是维护一张歧义词表（苹果/小米/华为…），命中就反问。
但这个方案在实践中很糟：

  ① **词表永远不全**：歧义是长尾的，新品牌新术语每天都在出现；
  ② **上下文能消解大部分歧义**："苹果股价" 一点都不歧义，
     但词表方案会照样触发反问 —— 这是最常见的误触发；
  ③ **有歧义 ≠ 需要澄清**："苹果多少钱" 若检索回来的 10 条
     全是 iPhone 报价，那实际上没有歧义，直接答就好。

所以本模块的判据是**检索结果的分裂程度**：
只有当召回的证据**确实分成了互不相关的几簇**（用已有的
`rag/dedup.py` 的向量相似度来度量），才说明"这个问题真的指向
多个不同的东西"。这是**数据驱动**而非规则驱动的判定，
天然覆盖长尾、天然被上下文消解。

════════════════════════════════════════════════════════════════════════
成本控制：追问推荐**默认复用主答案的那次 LLM 调用**
════════════════════════════════════════════════════════════════════════
最简单的实现是"答完后再调一次 LLM 生成推荐问题"，但这会：
    * 让端到端延迟 +1~3s（用户已经看到答案了，还在转圈）
    * 成本翻倍（summary 阶段用的是付费 API）

本模块给两种模式：
    "prompt"（默认）—— 把"末尾附 3 条追问"写进 summary 的 system prompt，
                       **零额外调用、零额外延迟、零额外成本**，
                       生成的问题还天然贴合答案内容（模型刚写完，最清楚）。
    "llm"           —— 独立调一次轻量模型（router 阶段的本地 qwen3）。
                       适合"不想动主 prompt"或"要严格控制格式"的场景。

默认选 prompt 模式：这一项本来就是"低成本高感知"，
如果实现得让延迟涨 2 秒，就把自己的价值抵消了。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

# ============================================================
#                        配置
# ============================================================
# 追问推荐生成模式："prompt"（复用主调用，默认）| "llm"（独立调用）| "off"
FOLLOWUP_MODE: str = os.getenv("FOLLOWUP_MODE", "prompt").lower()

# 推荐几条。3 条是业界（Perplexity / Bing）的通行取值：
#   2 条显得敷衍、5 条以上用户不会看完还占屏幕。
FOLLOWUP_COUNT: int = int(os.getenv("FOLLOWUP_COUNT", "3"))

# 单条追问的长度上限（字符）。超过就丢弃 ——
# 太长说明模型没理解"追问是一个简短问题"，而是又写了一段话。
FOLLOWUP_MAX_LEN: int = int(os.getenv("FOLLOWUP_MAX_LEN", "40"))

# 是否启用澄清提问。**默认关闭**。
#
# 为什么默认关，而追问推荐默认开：见模块 docstring 的"代价不对称"。
# 误触发澄清会**打断**正常问答，是负体验；而追问推荐不准只是没人点。
# 建议先在日志里观察 `should_clarify()` 的触发率与样本，
# 确认精度够高再打开。
ENABLE_CLARIFY: bool = os.getenv("ENABLE_CLARIFY", "false").lower() == "true"

# 澄清触发的簇间相似度上限。
#
# 语义：把召回证据聚成 2 簇后，**簇间**平均相似度低于此值，
# 才认为"这些证据讲的是完全不同的东西" → query 真有歧义。
#
# 取值依据（与 rag/dedup.py 的分布表同一量纲，BGE-M3 中文短文本）：
#     0.88~0.95  同一事件的不同报道        → 不是歧义
#     0.80~0.88  同一主题的不同子问题      → 不是歧义（CEO vs CFO）
#     0.65~0.80  相关但不同的话题          → 边界
#     <0.65      基本无关                  → **真歧义**（苹果公司 vs 苹果水果）
# 取 0.65 是刻意的保守值：宁可漏判，不可误触发。
CLARIFY_MAX_INTER_SIM: float = float(os.getenv("CLARIFY_MAX_INTER_SIM", "0.65"))

# 触发澄清所需的最少证据数。证据太少时"分裂"没有统计意义 ——
# 2 条证据不相似完全可能只是检索质量差，而不是 query 有歧义。
CLARIFY_MIN_PASSAGES: int = int(os.getenv("CLARIFY_MIN_PASSAGES", "4"))


# ============================================================
#             追问推荐：写进 summary prompt 的指令
# ============================================================
# 为什么用一个**极其显眼**的分隔符 `###FOLLOWUP###`：
#   1. 它必须能被 `parse_followups()` 从答案正文里**可靠切出来**，
#      所以不能用可能自然出现在正文里的记号（如 "---" 或 "相关问题："）；
#   2. 万一模型没遵守格式、或流式输出被用户中断，
#      解析失败时**必须**能优雅退化 —— 分隔符没出现就当没有推荐，
#      正文完整保留（绝不能因为解析失败而吃掉答案内容）。
FOLLOWUP_MARKER: str = "###FOLLOWUP###"

FOLLOWUP_PROMPT: str = f"""
【追问推荐】
回答完成后，另起一行输出分隔符 {FOLLOWUP_MARKER}，然后用 {FOLLOWUP_COUNT} 行
分别给出 {FOLLOWUP_COUNT} 个用户可能想继续追问的问题，每行一个，不加序号和符号。

要求：
1. 必须是**问句**，且能独立看懂（不要用"它/这个"指代上文）。
2. 每条不超过 {FOLLOWUP_MAX_LEN} 字。
3. 应当是本次回答的**自然延伸**（更深一层、相邻方面、或具体化），
   不要重复用户刚问过的问题，也不要问你刚刚已经答过的内容。
4. 如果本次因资料不足未能给出有效回答，则**不要**输出分隔符和追问。
"""


def build_followup_prompt() -> str:
    """返回要追加到 summary system prompt 的追问推荐指令。

    模式为 "prompt" 时才返回内容，否则返回空串 ——
    这样调用方（configs/prompts.py）无需关心模式判断。
    """
    return FOLLOWUP_PROMPT if FOLLOWUP_MODE == "prompt" else ""


# ============================================================
#                      解析与清洗
# ============================================================
# 行首的各种装饰：序号（1. / 1、/ (1) ）、项目符号（- * • ）、引号
_LINE_DECOR_RE = re.compile(r"^\s*(?:[-*•·]+|\(?\d+[\.\)、]|\d+\s*[:：])\s*")
_QUOTES_RE = re.compile(r'^["\'“”‘’「」『』]+|["\'“”‘’「」『』]+$')


def _clean_line(line: str) -> str:
    """清洗单行追问：去掉序号/项目符号/包裹引号/首尾空白。"""
    s = _LINE_DECOR_RE.sub("", line.strip())
    s = _QUOTES_RE.sub("", s).strip()
    return s


def _is_valid_followup(q: str, question: str, answer_body: str) -> bool:
    """判断一条候选追问是否值得展示。

    过滤规则（每一条都对应一种实际会出现的坏输出）：

      ① 空 / 过长 —— 模型没理解"简短问句"，又写了一段话。
      ② 不是问句 —— 有时模型会输出陈述句（"GDP的构成"），
         点进去当 query 效果差；要求以问号结尾或含疑问词。
      ③ 与用户原问题几乎相同 —— 推荐用户刚问过的东西毫无价值。
         用归一化后的包含关系判断（比编辑距离更简单且够用）。
      ④ 含指代词开头 —— "它的增速呢"脱离上下文看不懂，
         而追问是要**独立**作为新 query 发起检索的。
    """
    if not q or len(q) > FOLLOWUP_MAX_LEN:
        return False

    # ② 必须像个问句
    if not (q.endswith(("?", "？")) or re.search(
        r"(什么|哪些|哪个|哪里|如何|怎么|怎样|为何|为什么|多少|是否|谁|吗|呢)", q
    )):
        return False

    # ③ 与原问题重复
    def _norm(s: str) -> str:
        return re.sub(r"[\s，。？！、,.?!;:：；\"'“”‘’]+", "", s).lower()

    nq, norig = _norm(q), _norm(question)
    if nq == norig or (len(nq) > 6 and (nq in norig or norig in nq)):
        return False

    # ④ 以指代词开头 → 无法独立成为 query
    if re.match(r"^\s*(它|他|她|这个|那个|这些|那些|此|该)", q):
        return False

    return True


@dataclass
class FollowupResult:
    """`parse_followups` 的返回值。

    Fields:
        body:      **剥掉追问区之后**的答案正文。
                   调用方必须用这个替换原始答案 ——
                   否则用户会在答案末尾看到 `###FOLLOWUP###` 和裸问题列表。
        followups: 清洗后的追问列表（可能为空）。
        raw_count: 模型原始输出了几行（用于观测过滤率）。
                   若 raw_count 远大于 len(followups)，说明模型
                   格式遵守得不好，值得调 prompt。
    """
    body: str = ""
    followups: list[str] = field(default_factory=list)
    raw_count: int = 0


def parse_followups(answer: str, question: str = "") -> FollowupResult:
    """从答案文本里切出追问区，返回（正文, 追问列表）。

    ⚠️ **失败必须优雅**：这是本函数最重要的性质。
    如果分隔符没出现（模型没遵守 / 流式被中断 / 模式是 off），
    必须原样返回全文作为正文、追问为空。
    绝不能因为"解析追问失败"而丢掉、截断或污染答案内容 ——
    追问是锦上添花，答案是刚需。

    Args:
        answer:   模型原始输出（可能含追问区）。
        question: 用户原问题，用于过滤"推荐了用户刚问过的东西"。

    Returns:
        FollowupResult(body=正文, followups=[...], raw_count=n)
    """
    if not answer:
        return FollowupResult(body=answer or "")

    if FOLLOWUP_MARKER not in answer:
        # 绝大多数降级路径都走这里：原样返回，零副作用
        return FollowupResult(body=answer)

    # 用 split(maxsplit=1) 而不是 rsplit：分隔符只应出现一次，
    # 若模型抽风输出了多次，取**第一次**之前的内容作正文最安全
    # （后面的都算追问区，宁可少展示也不要把正文截断到一半）。
    head, _, tail = answer.partition(FOLLOWUP_MARKER)
    body = head.rstrip()

    raw_lines = [ln for ln in tail.splitlines() if ln.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for ln in raw_lines:
        q = _clean_line(ln)
        if not _is_valid_followup(q, question, body):
            continue
        # 去重：模型偶尔会输出两条措辞不同但归一化后相同的问题
        key = re.sub(r"[\s，。？！?!]+", "", q).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= FOLLOWUP_COUNT:
            break

    return FollowupResult(body=body, followups=out, raw_count=len(raw_lines))


# ============================================================
#          澄清提问：基于「证据分裂度」的保守判定
# ============================================================
@dataclass
class ClarifyDecision:
    """`should_clarify` 的返回值。

    Fields:
        need:      是否建议反问。
        question:  建议的澄清问句（need=False 时为空）。
        options:   识别出的几种可能指向（供前端做成按钮）。
        reason:    判定依据，用于日志与调参。
                   **即使 need=False 也会填** —— 上线初期靠这个
                   观察"差一点就触发"的样本，判断阈值是否合适。
        inter_sim: 实测的簇间相似度（观测用）。
    """
    need: bool = False
    question: str = ""
    options: list[str] = field(default_factory=list)
    reason: str = ""
    inter_sim: float = 1.0


def _split_two_clusters(mat) -> tuple[list[int], list[int], float]:
    """把向量矩阵粗分成 2 簇，返回 (簇A下标, 簇B下标, 簇间平均相似度)。

    用**最远点种子 + 就近归并**，而不是 KMeans：
      * 零依赖（不引入 sklearn）；
      * 确定性（KMeans 的随机初始化会让同一 query 两次给出不同
        澄清判定 —— 这在产品上是不可接受的，用户会觉得系统不稳定）；
      * 只需要 2 簇，这个简单算法完全够用。

    步骤：
      1. 找全局最不相似的一对 (a, b) 作为两个种子 ——
         如果连最远的一对都很相似，就说明根本没有分裂，可以提前返回；
      2. 其余每个点归到与之更相似的那个种子；
      3. 算两簇之间所有跨簇对的**平均**相似度作为分裂度指标。

    为什么用平均而不是最小：最小值只反映一个极端点对，
    容易被单个离群证据（一条完全跑偏的检索结果）触发。
    平均值更稳健，符合"整体分成两拨"的语义。
    """
    import numpy as np

    n = mat.shape[0]
    sims = mat @ mat.T                       # 已归一化 → 点积即余弦

    # 1) 最远点对作种子（把对角线排除，否则最小值总是自己和自己=1）
    tmp = sims.copy()
    np.fill_diagonal(tmp, 2.0)               # 填一个不可能成为最小值的大数
    a, b = np.unravel_index(int(np.argmin(tmp)), tmp.shape)
    a, b = int(a), int(b)

    # 2) 就近归并
    ca, cb = [a], [b]
    for i in range(n):
        if i in (a, b):
            continue
        (ca if sims[i, a] >= sims[i, b] else cb).append(i)

    # 3) 跨簇平均相似度
    cross = [float(sims[i, j]) for i in ca for j in cb]
    inter = sum(cross) / len(cross) if cross else 1.0
    return ca, cb, inter


def should_clarify(
    question: str,
    passages: Sequence,
    embedder=None,
    enable: Optional[bool] = None,
) -> ClarifyDecision:
    """判断是否该向用户反问以澄清歧义。

    ════════════════════════════════════════════════════════════════
    判据：检索证据是否**分裂成互不相关的几簇**
    ════════════════════════════════════════════════════════════════
    不用歧义词表（永远不全、会被上下文误触发、有歧义≠需澄清），
    而是看召回的证据本身。详见模块 docstring。

    三重保守约束（任一不满足就不反问）：
      ① 功能开关必须显式打开（默认关闭）；
      ② 证据数 ≥ CLARIFY_MIN_PASSAGES —— 太少时"分裂"没有统计意义，
         2 条证据不相似很可能只是检索差，不是 query 歧义；
      ③ 簇间平均相似度 < CLARIFY_MAX_INTER_SIM(0.65) ——
         这个阈值刻意取得很低，落在"基本无关"区间。
         0.80~0.95 那些"同主题不同侧面"的正常情况绝不会触发。

    Args:
        question: 用户原问题。
        passages: 检索到的 Passage 列表（需有 .text/.title）。
        embedder: BGE-M3 Embedder；None 则无法判定（直接返回 need=False）。
        enable:   覆盖 `ENABLE_CLARIFY`（测试用）。

    Returns:
        ClarifyDecision。**任何异常/不确定情形都返回 need=False**
        —— 宁可不反问，也不要误打断正常问答。
    """
    on = ENABLE_CLARIFY if enable is None else enable
    if not on:
        return ClarifyDecision(reason="功能未启用（ENABLE_CLARIFY=false）")

    n = len(passages) if passages else 0
    if n < CLARIFY_MIN_PASSAGES:
        return ClarifyDecision(
            reason=f"证据数 {n} < {CLARIFY_MIN_PASSAGES}，分裂度无统计意义"
        )
    if embedder is None:
        return ClarifyDecision(reason="无 embedder，无法判定分裂度")

    # 复用 dedup 的向量解析：能拿现成的就不重算（零成本复用 L2/L3 的向量）
    try:
        from rag.dedup import _resolve_vectors
        mat = _resolve_vectors(list(passages), embedder)
    except Exception as e:
        return ClarifyDecision(reason=f"向量解析异常，跳过澄清判定: {e}")
    if mat is None:
        return ClarifyDecision(reason="拿不到向量，跳过澄清判定")

    try:
        ca, cb, inter = _split_two_clusters(mat)
    except Exception as e:
        return ClarifyDecision(reason=f"聚类异常，跳过澄清判定: {e}")

    # 两簇都要有一定规模。
    # 若一簇只有 1 条，那更可能是**单个离群证据**（检索噪声），
    # 而不是"query 指向两个东西" —— 这种情况不该反问。
    if min(len(ca), len(cb)) < 2:
        return ClarifyDecision(
            inter_sim=inter,
            reason=f"次簇仅 {min(len(ca), len(cb))} 条，更像离群噪声而非歧义",
        )

    if inter >= CLARIFY_MAX_INTER_SIM:
        return ClarifyDecision(
            inter_sim=inter,
            reason=(
                f"簇间相似度 {inter:.3f} >= {CLARIFY_MAX_INTER_SIM}，"
                f"证据同源，无需澄清"
            ),
        )

    # ---- 判定为歧义：用两簇的代表标题作为选项 ----
    # 取每簇里 score 最高的那条的标题（截断到 20 字），
    # 让用户能一眼看出"你是想问这个还是那个"。
    def _label(idx_list: list[int]) -> str:
        best = max(idx_list, key=lambda i: float(getattr(passages[i], "score", 0.0)))
        p = passages[best]
        t = (getattr(p, "title", "") or getattr(p, "text", ""))[:20].strip()
        return t or "其他"

    opts = [_label(ca), _label(cb)]
    return ClarifyDecision(
        need=True,
        question=(
            f"你的问题「{question}」可能指向不同方面，"
            f"我检索到的资料明显分成了两类。请问你更想了解哪一个？"
        ),
        options=opts,
        inter_sim=inter,
        reason=f"簇间相似度 {inter:.3f} < {CLARIFY_MAX_INTER_SIM}，证据分裂",
    )


# ============================================================
#            "llm" 模式：独立调用生成追问（可选）
# ============================================================
_LLM_FOLLOWUP_TEMPLATE = """根据下面的问答，给出 {n} 个用户可能想继续追问的问题。

原问题：{question}
回答摘要：{answer}

要求：每行一个问题，不加序号；必须是能独立看懂的问句；
每条不超过 {maxlen} 字；不要重复原问题。直接输出问题，不要任何前言。"""


def generate_followups_via_llm(
    question: str,
    answer: str,
    stage: str = "router",
    count: Optional[int] = None,
) -> list[str]:
    """独立调一次轻量 LLM 生成追问（`FOLLOWUP_MODE="llm"` 时使用）。

    默认用 `stage="router"` —— 那是本地 ollama 的 qwen3:4b，
    **不花钱**，且已经因常驻策略（keep_alive=-1）留在内存里，
    延迟约 0.5s。绝不要用 summary 阶段（付费 API）做这件事：
    追问推荐的价值不值得让主链路成本翻倍。

    Returns:
        追问列表；任何异常都返回 `[]`（静默降级，不影响主答案）。
    """
    n = count or FOLLOWUP_COUNT
    try:
        import llm_client
        # 答案只取前 600 字：追问推荐需要的是"答了什么方向"，
        # 不需要全文。截断能明显降低 prompt token 与延迟。
        prompt = _LLM_FOLLOWUP_TEMPLATE.format(
            n=n, question=question, answer=(answer or "")[:600],
            maxlen=FOLLOWUP_MAX_LEN,
        )
        raw = llm_client.complete(stage, prompt)
    except Exception as e:
        print(f"[followup] LLM 生成追问失败（已忽略）: {e}")
        return []

    out: list[str] = []
    seen: set[str] = set()
    for ln in (raw or "").splitlines():
        q = _clean_line(ln)
        if not _is_valid_followup(q, question, answer):
            continue
        key = re.sub(r"[\s，。？！?!]+", "", q).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= n:
            break
    return out


# ============================================================
#                        渲染
# ============================================================
def render_followups_markdown(followups: Sequence[str]) -> str:
    """把追问渲染成 Markdown（Web 前端用）。

    用有序列表而非按钮：Gradio 的 Chatbot 渲染 Markdown，
    做成真按钮需要额外的组件与事件绑定。列表已经能让用户
    复制/点击（若前端支持），成本最低。
    """
    if not followups:
        return ""
    lines = ["", "---", "**🔮 你可能还想问**", ""]
    lines += [f"{i}. {q}" for i, q in enumerate(followups, 1)]
    return "\n".join(lines)


def render_followups_cli(followups: Sequence[str]) -> str:
    """把追问渲染成 CLI 纯文本。"""
    if not followups:
        return ""
    lines = ["", "🔮 你可能还想问："]
    lines += [f"   {i}. {q}" for i, q in enumerate(followups, 1)]
    return "\n".join(lines)


# ============================================================
#                  流式场景的分隔符抑制
# ============================================================
class StreamFilter:
    """流式输出时把追问区**挡在用户视野之外**。

    ════════════════════════════════════════════════════════════════
    为什么必须有这个类（这是 prompt 模式最容易漏掉的一个坑）
    ════════════════════════════════════════════════════════════════
    非流式路径很简单：拿到完整答案 → `parse_followups()` 剥掉追问区 →
    展示正文。但**流式**路径是边生成边 yield 的，等不到"完整答案"。
    如果原样透传，用户会亲眼看到：

        上半年GDP同比增长5.3%[1]。
        ###FOLLOWUP###          ← 内部实现细节泄漏到 UI
        三次产业各自的增速是多少？
        与去年同期相比表现如何？

    这比不做追问推荐还糟糕。所以流式必须有一个过滤器，在分隔符出现的
    那一刻**停止向用户吐字**，但继续把 token 收进缓冲区（供结束后解析）。

    ════════════════════════════════════════════════════════════════
    难点：分隔符会被**切碎**在多个 chunk 里
    ════════════════════════════════════════════════════════════════
    LLM 的流式 chunk 边界是任意的，`###FOLLOWUP###` 很可能被切成
    `##`, `#FOLLO`, `WUP##`, `#` 这样几段。逐 chunk 做 `in` 判断会漏检，
    然后分隔符的碎片就漏给用户了。

    【解法】滞后输出（hold-back）：始终**保留最后 len(marker)-1 个字符
    不吐**。这样任何跨 chunk 的分隔符都必然在缓冲区里被完整看到。
    代价是用户看到的文字比实际生成落后 14 个字符 —— 在打字机效果下
    完全无感（约 0.25 秒的视觉延迟）。

    这个"保留 n-1 个字符"是流式模式匹配的标准手法，同样的思路也用于
    流式敏感词过滤、流式 XML 标签解析。

    用法：
        f = StreamFilter()
        for piece in llm_stream():
            out = f.feed(piece)      # 可能返回 ""（被滞后）
            if out:
                yield out
        yield f.flush()              # 吐出尾部残留（若未出现分隔符）
        body, followups = f.result(question)
    """

    def __init__(self, marker: str = FOLLOWUP_MARKER):
        self._marker = marker
        # 滞后长度：marker 长度 - 1。保证跨 chunk 的 marker 一定能被完整匹配到。
        self._hold = max(len(marker) - 1, 0)
        self._pending = ""      # 尚未吐出的尾部
        self._full: list[str] = []   # 完整原始文本（含追问区），供结束后解析
        self._cut = False       # 是否已遇到分隔符（之后一律不再吐字）

    def feed(self, piece: str) -> str:
        """喂入一个 chunk，返回**可以安全展示**的文本（可能为空串）。"""
        if not piece:
            return ""
        self._full.append(piece)
        if self._cut:
            # 已经进入追问区：只收集，不再吐字
            return ""

        self._pending += piece
        idx = self._pending.find(self._marker)
        if idx >= 0:
            # 分隔符出现：吐出它**之前**的内容，之后全部丢弃
            self._cut = True
            out = self._pending[:idx]
            self._pending = ""
            return out

        # 未出现分隔符：吐出除末尾 hold 个字符以外的部分
        if len(self._pending) > self._hold:
            out = self._pending[: len(self._pending) - self._hold]
            self._pending = self._pending[len(self._pending) - self._hold:]
            return out
        return ""

    def flush(self) -> str:
        """流结束时吐出滞后缓冲区里的残留。

        若已遇到分隔符则返回 ""（残留属于追问区，不该展示）。
        """
        if self._cut:
            return ""
        out, self._pending = self._pending, ""
        return out

    def raw(self) -> str:
        """返回完整原始文本（含追问区），供 `parse_followups` 解析。"""
        return "".join(self._full)

    def result(self, question: str = "") -> FollowupResult:
        """解析出 (正文, 追问列表)。等价于对 `raw()` 调 `parse_followups`。"""
        return parse_followups(self.raw(), question)


# ============================================================
#                        自测
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("追问解析：正常情况")
    print("=" * 70)
    demo = f"""上半年GDP同比增长5.3%[1]，增速比一季度加快0.1个百分点[2]。

{FOLLOWUP_MARKER}
1. 三次产业各自的增速是多少？
2. 与去年同期相比表现如何？
- 居民收入增长了多少？
"""
    r = parse_followups(demo, "上半年GDP增长多少")
    print(f"正文: {r.body!r}")
    print(f"追问({r.raw_count}行→{len(r.followups)}条):")
    for q in r.followups:
        print(f"   • {q}")
    assert FOLLOWUP_MARKER not in r.body, "分隔符泄漏到正文！"
    assert len(r.followups) == 3

    print()
    print("=" * 70)
    print("追问解析：降级路径（无分隔符 → 原样返回）")
    print("=" * 70)
    plain = "这是一个没有追问区的普通答案。"
    r2 = parse_followups(plain, "问题")
    assert r2.body == plain and not r2.followups
    print(f"✓ 正文完整保留: {r2.body!r}")

    print()
    print("=" * 70)
    print("追问过滤：各类坏输出")
    print("=" * 70)
    bad = f"""答案。
{FOLLOWUP_MARKER}
上半年GDP增长多少
它的增速呢？
这是一个非常非常长的问题需要超过四十个字符才能触发长度过滤规则所以我要继续写下去
三次产业的构成
第二产业增速是多少？
"""
    r3 = parse_followups(bad, "上半年GDP增长多少")
    print(f"5 行原始输出 → 保留 {len(r3.followups)} 条: {r3.followups}")
    assert r3.followups == ["第二产业增速是多少？"], r3.followups
    print("✓ 重复原问题/指代开头/过长/非问句 全部被过滤")

    print()
    print("=" * 70)
    print("澄清判定：默认必须关闭")
    print("=" * 70)
    d = should_clarify("苹果多少钱", [], None)
    assert not d.need
    print(f"✓ need={d.need}  reason={d.reason}")

    print()
    print("=" * 70)
    print("流式过滤：分隔符被切碎在多个 chunk 里也不能漏给用户")
    print("=" * 70)
    # 刻意把 ###FOLLOWUP### 切得很碎，模拟真实 LLM 的任意 chunk 边界
    chunks = ["上半年GDP同比", "增长5.3%[1]。", "\n#", "##FOL", "LOW",
              "UP##", "#\n三次产业增速?", "\n居民收入增长多少?"]
    f = StreamFilter()
    shown = []
    for c in chunks:
        out = f.feed(c)
        if out:
            shown.append(out)
    shown.append(f.flush())
    visible = "".join(shown)
    print(f"用户看到: {visible!r}")
    assert FOLLOWUP_MARKER not in visible, "分隔符泄漏到用户视野！"
    assert "#" not in visible, f"分隔符碎片泄漏: {visible!r}"
    assert "三次产业" not in visible, "追问内容泄漏到正文！"
    assert visible == "上半年GDP同比增长5.3%[1]。\n", f"正文被破坏: {visible!r}"
    fr = f.result("GDP增长多少")
    print(f"解析追问: {fr.followups}")
    assert len(fr.followups) == 2, fr.followups
    print("✓ 碎片分隔符被完整拦截，正文无损，追问正常解析")

    print()
    print("=" * 70)
    print("流式过滤：无追问区时正文必须逐字完整（最常见路径）")
    print("=" * 70)
    f2 = StreamFilter()
    plain_chunks = ["这是", "一个普通", "答案，没有追问区。"]
    got = "".join(f2.feed(c) for c in plain_chunks) + f2.flush()
    assert got == "".join(plain_chunks), f"正文被吃掉: {got!r}"
    print(f"✓ 输出完整: {got!r}")

    print()
    print("=" * 70)
    print("渲染")
    print("=" * 70)
    print(render_followups_cli(r.followups))
    print(render_followups_markdown(r.followups))
    print()
    print("✅ followup.py 自测通过")
