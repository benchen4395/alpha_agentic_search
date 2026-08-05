# configs/prompts.py
"""集中管理所有 prompt 模板。

为什么集中：
1. Prompt 是 LLM 系统中最容易迭代的部分 —— 单独一个文件方便 diff / 回滚
2. 业务代码（agent.py / tool_router.py / query_rewriter.py）通过 key 引用
3. 想 A/B 试不同提示词，只要改这里，不动业务逻辑

使用方式：
    from configs.prompts import PROMPTS, render

    text = render("rewriter", history="...", query="...", context="...")

约定：
    占位符使用 {var} 风格；render() 缺失字段会用空串兜底，不会 raise KeyError。
"""
from __future__ import annotations

from typing import Any


# ============================================================
# 主控对话的 system prompt（summary 阶段）
# ============================================================
# P0-4 / P0.5 改造说明
# --------------------
# 1. 资料格式从「[n] 标题 + URL + 摘要」的纯文本改为
#    `<evidence><doc id="n" .../></evidence>` 结构化定界
#    （见 evidence.build_evidence_block），所以这里的表述同步更新。
# 2. 追加 `EVIDENCE_GUARD_PROMPT`（见文件末尾的组装逻辑）：
#    显式声明 <doc> 内是数据而非指令，对抗 prompt injection。
# 3. 引用要求写得更硬：编号必须真实存在、且必须给来源列表。
#    这是 P0.5 引用归因的前提——系统会解析并**校验**每个 [n]
#    （见 answer_types.parse_citations），编造的编号会被标记为无效。
_SUMMARY_SYSTEM_BASE: str = """你是一个具备联网搜索能力的智能助手。你被命名为 benbot，由 chenben03 开发。

{context}

行为规则：
1. 当用户问题需要外部知识（最新事件、技术细节、人物背景等），系统会在用户消息的
   <evidence> 区块里附上检索结果或工具调用结果，请**基于这些资料**作答。
   每条资料形如 <doc id="3" type="维基百科" title="…" url="…" conf="0.88">…</doc>：
     - `id`   是引用编号，你只能引用真实出现过的 id；
     - `type` 是来源类型（维基百科 / 实时网页 / 知识图谱 / 历史问答 / 缓存问答）；
     - `conf` 是系统给出的相关度置信度（0~1），**优先采信 conf 高的资料**，
       当多条资料冲突时以 conf 更高、来源更权威者为准，并可指出存在分歧。
2. 引用规范（重要）：
   - 陈述来自资料的具体事实时，必须在句末用 [n] 标注，n 为对应 <doc> 的 id；
   - 一句话综合多条资料时写 [1,3]；
   - **绝不允许编造不存在的编号**（系统会自动校验并标记无效引用）；
   - 回答末尾无需重复罗列 URL —— 系统会根据你的 [n] 自动生成来源面板。
3. 如果资料与问题不相关或不足以回答，要**诚实说明信息不足**，
   并可补充你确定的一般性背景，但不要臆测具体事实、数字、人名、日期。
4. 闲聊或无需外部信息时，直接基于对话历史自然回答，此时不需要引用。
5. 中文用户请使用中文回答；保持简洁、有结构（可用小标题/列表）。
6. **时效性校验（重要）**：资料是检索来的，可能**严重过期**——搜索引擎的
   索引/快照未必是最新的，网页里的"今日""最新"指的是**该网页发布当天**，
   不是现在。所以凡是回答"今日/当前/最新"类问题：
   - 先把资料里的日期与上方[环境信息]里的**当前日期**对照；
   - 若资料日期与今天**不是同一天**，必须在答案开头明确标注资料的实际日期，
     例如「⚠️ 以下数据来自 2026-04-16，距今已 3 个多月，非实时行情」；
   - **绝不允许**把过期数据的日期改写成今天，也不要说"今日"来指代它；
   - 若资料完全没有日期信息，就说明"资料未标注时间，无法确认时效"；
   - 价格、汇率、股价、天气、比分、排名这类**高频变动**数据尤其要严格执行，
     宁可提示用户"请以实时行情为准"，也不要给出看似精确但已过期的数字。
"""


def _build_summary_system() -> str:
    """把「行为规则」与「外部资料安全规则」组装成最终 system prompt。

    P0-4：安全规则单独维护在 `evidence.EVIDENCE_GUARD_PROMPT`，
    原因有两点：
      1. 它与 `<doc>` / `<evidence>` 的具体封装格式强耦合，
         放在 evidence.py 里与封装实现同处一地，改格式时不会漏改 prompt；
      2. 便于其它需要喂外部内容的 stage（未来的 verifier / planner）复用。

    evidence 模块导入失败时（例如被单独拆出去用）自动降级为只用基础规则，
    不影响启动。
    """
    try:
        from evidence import EVIDENCE_GUARD_PROMPT
        return _SUMMARY_SYSTEM_BASE + "\n" + EVIDENCE_GUARD_PROMPT + "\n"
    except Exception:
        return _SUMMARY_SYSTEM_BASE


SUMMARY_SYSTEM: str = _build_summary_system()


def build_summary_system(context: str | None = None) -> str:
    """渲染 summary 阶段的 system prompt，**注入实时环境信息**。

    ════════════════════════════════════════════════════════════════════
    为什么必须是函数，而不能是模块级常量
    ════════════════════════════════════════════════════════════════════
    实测故障（本次修复的起因）：

        提问：今日黄金价格          （真实日期 2026-08-05）
        回答：今日黄金价格（2026年4月16日）
              伦敦金现 4822.88 美元/盎司  -0.29%
              「以上为盘中实时数据」     ← 把 4 个月前的数据说成实时

    根因是**阶段间的信息不对称**：`REWRITER_TEMPLATE` 里有 `{context}`，
    所以 rewriter 知道今天几号、能正确改写出"2026年8月"；但真正撰写
    答案的 summary 阶段拿到的是一个**不含任何时间信息**的静态常量，
    于是资料里的日期被它当成了"今天"。

    ⚠️ 如果把结果存成模块级常量（`SUMMARY_SYSTEM = build_summary_system()`），
    日期会在**进程 import 那一刻被冻结**。长期运行的 Web 服务跑过午夜后，
    prompt 里仍写着昨天 —— 这与我们要修的 bug 是同一类错误，只是更隐蔽。
    所以调用方必须**每次请求都调一次**本函数。

    Args:
        context: 环境信息块。默认 None → 现场调
            `context_provider.build_context_block()` 取实时时间与地点。
            测试可传固定字符串以获得确定性输出。
    """
    if context is None:
        try:
            from context_provider import build_context_block
            context = build_context_block()
        except Exception:
            # 取不到环境信息也不能让整条链路挂掉：降级为空串。
            # 此时行为退回"改造前"——模型不知道今天几号，
            # 但规则 6 仍会要求它标注资料日期，比完全没有防护好。
            context = ""
    return _build_summary_system().format_map(_SafeDict({"context": context}))


# ============================================================
# Query 改写 prompt（rewriter 阶段）
# ============================================================
REWRITER_TEMPLATE: str = """你是一个搜索 query 优化专家。

{context}

基于"环境信息"、"对话历史"和"用户最新提问"，输出一个最适合搜索引擎的查询字符串。

要求：
1. 只输出 query 本身，不要解释、不要引号、不要 markdown。
2. 补全代词（"它/这个/他们" 替换为具体实体）。
3. 补充专业术语、地名等**能提高召回**的关键词。
4. **不要自行添加年份**。只有当用户的问题本身就在追问时效性信息
   （出现"今年""最新""现在""目前""当前"等词，或明确提到某个年份）时，
   才可以补上年份；对"历史上共有多少位总统"这类**历史累计型**问题，
   绝对不要加年份。
5. 如果用户表达了"不要 / 排除 / 不喜欢"等否定意图，请在该实体前加 `-`（例：`-保健品`）。
6. 如果用户问题与搜索无关（闲聊、问候、追问对话内容等），输出 NO_SEARCH。
7. 输出 query 尽可能精简，但一定要保证准确性，不要添加过多的修饰词。

[对话历史]
{history}

[用户最新提问]
{query}

[改写后的搜索 query]"""


# ============================================================
# 工具路由 prompt（router 阶段）
# ============================================================
ROUTER_TEMPLATE: str = """你是一个工具调用路由器。下面是你可以使用的工具：

{tool_list}

规则：
1. 严格只输出一个 JSON 对象，格式：{{"tool": "<工具名 或 NO_TOOL>", "args": {{...}}}}
2. 不要解释，不要 markdown，不要代码块围栏。
3. 如果用户问题可以用专用工具回答（当前时间/日期/星期、天气/GitHub/arXiv 等），优先使用专用工具。
4. 当用户询问当前时间、日期、星期、某城市/地区现在几点时，选择 get_current_time：
   - 没有地点："现在几点"、"今天星期几" → {{"tool":"get_current_time","args":{{}}}}
   - 包含城市/地区："拉斯维加斯时间是多少" → {{"tool":"get_current_time","args":{{"location":"拉斯维加斯"}}}}
   - 直接包含 IANA 时区："America/Los_Angeles 当前时间" → {{"tool":"get_current_time","args":{{"timezone":"America/Los_Angeles"}}}}
5. 当问题是事实问答 / 开放性新闻 / 明星 / 事件 / 百科类（如美国现任总统、2026 年奥斯卡奖、世界杯信息等），表示需要额外搜索。
6. 当问题是闲聊或与外部数据无关，输出 {{"tool": "NO_TOOL", "args": {{}}}}。
7. args 必须是 JSON 对象（哪怕为空 {{}}）；只填工具声明里支持的参数。

[用户问题]
{query}

[输出]"""


# ============================================================
# 注册表：key → 模板
# ============================================================
PROMPTS: dict[str, str] = {
    "summary_system": SUMMARY_SYSTEM,
    "rewriter": REWRITER_TEMPLATE,
    "router": ROUTER_TEMPLATE,
}


class _SafeDict(dict):
    """str.format_map 兜底：缺失字段返回空串而不是抛错。"""

    def __missing__(self, key: str) -> str:  # noqa: D401
        return ""


def render(key: str, **kwargs: Any) -> str:
    """根据 key 渲染 prompt 模板。

    例：
        render("rewriter", history="...", query="今天巴黎天气", context="")
    """
    if key not in PROMPTS:
        raise KeyError(
            f"未注册的 prompt key: {key}。可选: {list(PROMPTS.keys())}。"
            f"请在 configs.prompts.PROMPTS 中新增。"
        )
    template = PROMPTS[key]
    # 用 format_map + SafeDict，避免某个占位符没传就 raise
    return template.format_map(_SafeDict(kwargs))


def list_prompts() -> list[str]:
    return list(PROMPTS.keys())


def register_prompt(key: str, template: str) -> None:
    """动态注册/覆盖某个 prompt（便于运行期热更或测试）。"""
    PROMPTS[key] = template
