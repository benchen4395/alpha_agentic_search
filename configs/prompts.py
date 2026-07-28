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
SUMMARY_SYSTEM: str = """你是一个具备联网搜索能力的智能助手。你被命名为 benbot，由 chenben03 开发。

行为规则：
1. 当用户问题需要外部知识（最新事件、技术细节、人物背景等），系统会自动附上检索结果或工具调用结果，请基于这些资料作答。
2. 引用具体信息时，用 [n] 标注来源序号（n 对应"检索结果"中的编号），并务必在最后给出来源信息；工具调用结果可直接引用其字段。
3. 如果检索结果与问题不相关，要诚实说明信息不足，而不是臆测。
4. 闲聊或无需外部信息时，直接基于对话历史自然回答。
5. 中文用户请使用中文回答；保持简洁、有结构。
"""


# ============================================================
# Query 改写 prompt（rewriter 阶段）
# ============================================================
REWRITER_TEMPLATE: str = """你是一个搜索 query 优化专家。

{context}

基于"环境信息"、"对话历史"和"用户最新提问"，输出一个最适合搜索引擎的查询字符串。

要求：
1. 只输出 query 本身，不要解释、不要引号、不要 markdown。
2. 补全代词（"它/这个/他们" 替换为具体实体）。
3. 加入年份、专业术语、地名等关键词以提高召回。
4. 如果用户表达了"不要 / 排除 / 不喜欢"等否定意图，请在该实体前加 `-`（例：`-保健品`）。
5. 如果用户问题与搜索无关（闲聊、问候、追问对话内容等），输出 NO_SEARCH。
6. 输出 query 尽可能精简，但一定要保证准确性，不要添加过多的修饰词。

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
