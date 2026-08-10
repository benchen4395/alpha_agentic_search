# rag/textclean.py
"""Web snippet 噪声清洗 —— 零成本提升证据质量。

════════════════════════════════════════════════════════════════════════
要解决的问题：证据预算被「页面模板」吃掉
════════════════════════════════════════════════════════════════════════
L4 web 的 `snippet` 并不是干净的正文。实测缓存里 20 条真实结果：

    中位数 1339 字，85% 超过 500 字     ← 长度其实够用
    但其中混着大量**页面结构噪声**

真实样本（旅游/电商站，噪声可占近半）：

    ||  |  |  |                                  ← 空表格骨架
    |---                                         ← Markdown 表格分隔线
    || 曼谷 | 清邁 | 網卡／交通 |                  ← 表头
    TWD 210 起立即預訂                            ← 价格 + CTA 按钮
    4.5                                          ← 裸评分
    30K+ 個已訂購                                 ← 销量标签
    很好177 則評論 / 4.7/51358 reviews            ← 评分块
    白金 / 鑽石 / SIM 卡                          ← 会员等级、品类标签
    Trip.com 用戶 / tntparadise                   ← 评论者昵称
    review picture                               ← 图片占位
    batch Thailand ... AShutterstock 389485657   ← 图片文件名
    Source：Shutterstock                          ← 图片版权
    ▶︎泰國入境規定                                ← 站内导航链接列表
    © OpenStreetMap contributors                  ← 页脚版权
    12345678910月1112                             ← 月份选择器控件

这些噪声有三重实际代价：

  ① **挤占证据预算**：`evidence.py` 的 `max_total_len=8000` 是硬预算。
     噪声占一半就等于**证据条数腰斩** —— 本该进来的第 5、6 条被挤掉。
  ② **干扰模型注意力**：`TWD 210 起立即預訂` 对"泰国几月去最好"毫无用处，
     但它占据 token、并且和真正的答案句混在同一个 <doc> 里。
  ③ **污染语义去重**：`rag/dedup.py` 的 `_texts_for_embedding` 截断到
     512 字。若前 512 字有一半是导航栏，算出的余弦反映的是
     **"页面模板像不像"**而不是"内容像不像" —— 去重判据被带偏。

════════════════════════════════════════════════════════════════════════
为什么不直接上 trafilatura / readability（P1-1）
════════════════════════════════════════════════════════════════════════
那条路要**重新抓取原始 HTML**（每个 url 一次 HTTP 请求），代价是：
    延迟 +1~3s（N 个 url 并发抓取）、新增依赖、抓取失败/反爬要兜底
而本模块是**纯正则、零联网、零依赖、零额度消耗**，耗时微秒级。

两者不冲突：即使将来做了正文抽取，抽出来的正文**仍然需要清洗**
（trafilatura 也挡不住"TWD 210 起立即預訂"这类正文内嵌的营销块）。
所以本模块是先做、且长期有效的一层。

════════════════════════════════════════════════════════════════════════
设计原则：宁可漏删，不可错删
════════════════════════════════════════════════════════════════════════
和 `rag/dedup.py` 同一个取向，因为**代价不对称**：

    漏删一行噪声  → 浪费几十字预算
    错删一行正文  → **可能删掉答案本身**

具体体现在三点：
  * 每条规则都**尽量窄**，只覆盖能确定的模式；
  * Markdown 标题、含句子标记的行**优先放行**，即使它同时命中某条规则；
  * 全流程有**兜底**：若清洗后内容少于原文的 `MIN_KEEP_RATIO`，
    直接放弃清洗、返回原文。宁可带着噪声，也不能把证据洗没了。

════════════════════════════════════════════════════════════════════════
为什么在 L4 层清洗，而不是在 searcher 里
════════════════════════════════════════════════════════════════════════
`searcher.web_search` 的结果会写进 diskcache（TTL 通常数小时~数天）。
若在那里清洗，**缓存里存的就是洗过的文本**，那么：
  * 规则调整后，老缓存不会重新清洗，新旧策略混在一起；
  * 想对比"清洗前 vs 清洗后"效果时，原始数据已经没了。

在 L4 层清洗则是：缓存存原始文本，每次检索时现洗（纯正则，微秒级）。
调规则立即全量生效，且随时能拿原文做对照。这是"**尽量晚地做有损变换**"
的一般原则。
"""
from __future__ import annotations

import os
import re

# ============================================================
#                          开关与阈值
# ============================================================
ENABLE_SNIPPET_CLEAN: bool = (
    os.getenv("RAG_ENABLE_SNIPPET_CLEAN", "true").lower() == "true"
)

# 清洗后至少要保留原文的多少比例，否则视为"规则过激"，放弃清洗返回原文。
#
# 取 0.3 的依据：实测最脏的样本噪声占比约 45~50%，清洗后保留约一半。
# 0.3 给出充足余量 —— 只有当规则把 70% 以上内容都判成噪声时才触发兜底，
# 那几乎只可能是规则写错了或页面结构极端异常（如整页都是表格），
# 此时"原样返回"比"返回一句话"更安全。
MIN_KEEP_RATIO: float = float(os.getenv("RAG_CLEAN_MIN_KEEP_RATIO", "0.3"))

# 短于此长度的文本不清洗：DDG/Serper/Bing 的摘要通常只有 40~100 字
# （实测 DDG 中位数 61 字），本身就是搜索引擎抽好的句子，
# 几乎没有模板噪声，清洗只有风险没有收益。
MIN_LEN_TO_CLEAN: int = int(os.getenv("RAG_CLEAN_MIN_LEN", "200"))

# 「孤立短标签」的长度上限：短于此且完全无句子标记的行判为噪声
# （评论者昵称、POI 卡片标题、品类标签）。
#
# 取 12 的依据：实测放宽到 20 会开始误删
# `雨季時間：每年約 11 月至隔年 3 月` 这类**冒号式要点** ——
# 它们是高价值的结构化事实。12 字刚好只覆盖
# `巴杜爾火山` / `Trip.com 用戶` / `tntparadise` 这类纯标签。
SHORT_LABEL_MAX: int = int(os.getenv("RAG_CLEAN_SHORT_LABEL_MAX", "12"))


# ============================================================
#                        噪声行的判定规则
# ============================================================
# ⚠️ 绝大多数规则用 fullmatch（匹配**整行**）而不是 search（匹配子串）。
# 理由：`TWD 210` 出现在正文句子里（"門票 TWD 210，建議提前預訂"）是
# 有效信息；只有当整行就是 `TWD 210 起立即預訂` 时才是 CTA 按钮。
# 用子串匹配会把含价格的正常句子整句删掉。

# ---- ① 表格骨架：`||  |  |  |` / `|---` / `| --- | --- |` ----
# 只有分隔符没有内容，删掉零风险。
_RE_TABLE_SKELETON = re.compile(r"^[|\s\-:+]+$")

# ---- ② 纯符号行：`▶︎` / `***` / `···` ----
# \w 覆盖中英文与数字，所以"不含 \w"就意味着没有任何实质内容。
_RE_SYMBOLS_ONLY = re.compile(r"^[^\w]+$")

# ---- ③ 裸数字 / 裸评分行：`4.5` / `100` / `85%` ----
# 单独一行的数字是评分、序号、分页码之类的 UI 元素。
#
# ⚠️ 必须限制在"整行只有一个数字"，否则会删掉 `一月：27-31°C`
# 这类关键数据。所以不允许出现任何非数字字符（百分号除外）。
_RE_BARE_NUMBER = re.compile(r"^\d+(?:[.,]\d+)?%?$")

# ---- ④ 连续数字串（选择器控件）：`12345678910月1112` ----
# 月份选择器把 1~12 连着渲染出来。
#
# ⚠️ 实现坑（回归测试抓到过）：不能用单条
# "数字开头 + 8位连续数字 + 非\w字符" 的正则 —— 因为"月"属于 \w
# （Python re 默认 Unicode 语义），它夹在数字中间时那条规则静默失效。
# 正确做法拆成两个条件，必须**同时**满足：
#   ANY  : 整行存在 8 位以上连续数字段
#   ONLY : 整行除了数字与少量单位字（年/月/日/页）之外没有别的内容
# 两个都要，是为了不误删 `订单号 20260807123456 已生成，请查收。`
_RE_DIGIT_RUN_ANY = re.compile(r"\d{8,}")
_RE_DIGIT_RUN_ONLY = re.compile(r"^[\d\s年月日页頁第共/·.\-]+$")

# ---- ⑤ 版权 / 图片来源声明 ----
# `© OpenStreetMap contributors` / `Source：Shutterstock` / `Photo by xxx`
_RE_ATTRIBUTION = re.compile(
    r"^(?:©|\(c\)|copyright|source\s*[:：]|來源\s*[:：]|来源\s*[:：]"
    r"|圖片來源\s*[:：]|图片来源\s*[:：]|photo\s+by|image\s*[:：])",
    re.IGNORECASE,
)

# ---- ⑥ 电商 CTA / 价格标签 ----
# `TWD 210 起立即預訂` / `TWD 0 起` / `30K+ 個已訂購` / `立即預訂`
_RE_PRICE_CTA = re.compile(
    r"^(?:(?:TWD|USD|CNY|JPY|HKD|RMB|NT\$|US\$|\$|￥|¥|€|£)\s*[\d.,]+[\s\w]{0,12}"
    r"|[\d.,]+[KkMm]?\+?\s*(?:個已訂購|个已订购|已訂|已订|人已購買|人已购买)"
    r"|(?:立即預訂|立即预订|馬上預訂|马上预订|加入購物車|加入购物车))$"
)

# ---- ⑥b 评分 / 评论数 / 时间戳（点评与电商站的固定 UI 块）----
# 实测残留最多的一类：
#     `很好177 則評論` / `滿意45 則評論` / `4.7/51358 reviews`
#     `4.5/5902 reviews` / `22648 已訂` / `1 週前` / `review picture`
# 它们与"这个地方几月去最好"完全无关，纯粹吃预算。
_RE_RATING_BLOCK = re.compile(
    r"^(?:"
    r"(?:很好|滿意|满意|極佳|极佳|優|优|好評|好评|一般|差評|差评)?\s*"
    r"[\d.,]+\s*(?:則評論|则评论|條評論|条评论|個評論|个评论|則評價|条评价)"
    r"|[\d.]+\s*/\s*[\d.]+\s*[\d,]*\s*reviews?"
    r"|[\d,]+\s*(?:已訂|已订|人參加|人参加|人看過|人看过)"
    r"|\d+\s*(?:週前|周前|天前|小時前|小时前|分鐘前|分钟前|個月前|个月前|年前)"
    r"|review\s+picture"
    r")$",
    re.IGNORECASE,
)

# ---- ⑥c 会员等级 / 品类标签 ----
# `白金` / `鑽石` / `SIM 卡` / `AI 精選摘要` —— 无上下文的标签行。
# 用集合精确匹配**整行**，避免删掉正文里的同词。
_UI_TAGS = {
    "白金", "鑽石", "钻石", "黃金", "黄金", "白銀", "白银", "極佳", "极佳",
    "很好", "滿意", "满意", "一般", "普通", "優秀", "优秀", "推薦", "推荐",
    "熱門", "热门", "新品", "特價", "特价", "限時", "限时", "免費", "免费",
    "SIM 卡", "SIM卡", "eSIM", "AI 精選摘要", "AI精選摘要", "AI 精选摘要",
}

# ---- ⑦ 图片文件名 / CDN 资源名 ----
# `batch Thailand Bangkok ... AShutterstock 389485657`
# 特征：图库名紧跟长数字 ID。这类是 <img alt> 被抽出来的，
# 语义价值极低但很长（吃预算）。这条**用 search**（子串匹配），
# 因为图库名可能出现在行中间。
_RE_IMAGE_ASSET = re.compile(
    r"(?:shutterstock|fotolia|unsplash|getty|istock|pixabay|adobestock|123rf)"
    r"[\s_-]*\d{5,}",
    re.IGNORECASE,
)

# ---- ⑧ 常见 UI 按钮词 ----
# `链接 下载 比较` / `分享 收藏 举报` —— 整行只由 UI 动作词构成。
_UI_WORDS = {
    "链接", "連結", "下载", "下載", "比较", "比較", "分享", "收藏", "舉報",
    "举报", "登录", "登入", "註冊", "注册", "搜索", "搜尋", "首页", "首頁",
    "返回", "更多", "展开", "展開", "收起", "打印", "列印", "复制", "複製",
    "点赞", "點贊", "评论", "評論", "关注", "關注", "订阅", "訂閱",
    "share", "download", "compare", "login", "signup", "subscribe",
    "print", "copy", "more", "back", "home", "search", "menu",
}

# ---- ⑨ 站内导航链接列表 ----
# `▶︎泰國入境規定` / `▶︎曼谷景點、曼谷自由行、泰國跳島合輯`
#
# ⚠️ 与正文的正确区分是**句末有没有终止标点**，而不是"有没有顿号"：
#     ▶︎曼谷景點、曼谷自由行、泰國跳島合輯   ← 顿号列表、无句末标点 = 导航
#     ▶ 這裡風景優美，景色迷人，值得一遊。    ← 有句号 = 正文强调项
# 我最初写的判据是"箭头后有句子标记就保留"，但顿号本身就在句子标记里，
# 结果顿号分隔的导航列表全被放过（回归测试抓到）。
_RE_NAV_ARROW = re.compile(r"^[▶►▸➤·•‣⇨→>]{1,2}[\s︎]*\S")
# 句末终止标点：只有它能证明"这是一个说完的句子"
_RE_SENTENCE_END = re.compile(r"[。；！？!?.]\s*$")

# 判定"这一行有没有实质内容"：中日韩字符或 3 字母以上英文单词
_RE_HAS_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
_RE_HAS_WORD = re.compile(r"[A-Za-z]{3,}")

# 句子标记：标点或中文虚词。出现它们说明这是**自然语言句子**
# 而不是标签堆砌，用于给"短但有价值"的行开绿灯
# （如 `雨季時間：每年約 11 月至隔年 3 月` 靠冒号被放行）。
_RE_SENTENCE_MARK = re.compile(r"[，。；：！？、,;.!?]|的|是|在|有|了|和|与|與")


def _is_noise_line(line: str) -> bool:
    """判断单行是否为噪声。

    判定顺序是刻意设计的，从"最安全"到"最激进"：
        白名单放行 → 无损删除 → 强特征删除 → 结构启发式 → 兜底
    这样即使后面的启发式规则有偏差，高价值内容也已经在前面被放行了。
    """
    s = line.strip()
    if not s:
        return True                       # 空行：无损删除

    # ---- 白名单：Markdown 标题一律保留 ----
    # `## 巴厘岛气候指南` / `### 七月` 是**结构信息**，帮模型理解
    # "接下来这段在讲什么"，价值很高。放最前面是因为标题往往很短，
    # 会被下面的"孤立短标签"规则误判。
    if s.startswith("#"):
        return False

    # ---- 无损删除：纯结构 / 纯符号，不含任何实质内容 ----
    if _RE_TABLE_SKELETON.fullmatch(s):
        return True
    if _RE_SYMBOLS_ONLY.fullmatch(s):
        return True
    if _RE_BARE_NUMBER.fullmatch(s):
        return True
    if _RE_DIGIT_RUN_ANY.search(s) and _RE_DIGIT_RUN_ONLY.fullmatch(s):
        return True

    # ---- 强特征删除：即使含实词也删 ----
    # 这几类的"实词"本身就是噪声内容（版权方名、图库名、价格、评分），
    # 保留它们没有意义。
    if _RE_ATTRIBUTION.match(s):
        return True
    if _RE_PRICE_CTA.fullmatch(s):
        return True
    if _RE_RATING_BLOCK.fullmatch(s):
        return True
    if _RE_IMAGE_ASSET.search(s):
        return True
    if s in _UI_TAGS:
        return True
    if _RE_NAV_ARROW.match(s) and not _RE_SENTENCE_END.search(s):
        return True

    # ---- UI 词组行 ----
    # 整行拆开后**每一个** token 都是 UI 词 → 是按钮栏。
    # 用"全部命中"而非"任一命中"：`下载安装包后请重启` 含"下载"
    # 但显然是正文。
    tokens = [t for t in re.split(r"[\s|/、,，]+", s) if t]
    if tokens and all(t.lower() in _UI_WORDS for t in tokens):
        return True

    # ---- 表格数据行 ----
    # `|| 曼谷 | 清邁 | 網卡／交通 |` —— 以 | 开头、被 | 切成多段，
    # 每段都很短（<12 字）且整行无句子标记 → 表头/单元格堆砌。
    #
    # ⚠️ 不能因为"以 | 开头"就删：真实数据里有
    # `| 云彩 十月 雅典出现快速增加的云量，整个月份...` 这种
    # **正文被裹在表格里**的情况，它含逗号和长句，必须保留。
    if s.startswith("|"):
        cells = [c.strip() for c in s.strip("|").split("|")]
        cells = [c for c in cells if c]
        if cells and all(len(c) < 12 for c in cells) and \
                not _RE_SENTENCE_MARK.search(s):
            return True

    # ---- 孤立短标签行 ----
    # 实测残留最多的一类：`Trip.com 用戶` / `巴杜爾火山` /
    # `tntparadise` / `KSENIIA CHERHUSHKINA` —— 评论者昵称、
    # POI 卡片标题，单独一行、极短、没有任何句子标记。
    #
    # ⚠️ 这是本模块**最激进**的一条，所以阈值很小（12 字）。
    # 两个保护：
    #   ① Markdown 标题已在最上面放行，不会走到这里；
    #   ② 冒号在 _RE_SENTENCE_MARK 里，所以
    #      `雨季時間：每年約 11 月至隔年 3 月` 这类高价值要点天然被放行。
    if len(s) < SHORT_LABEL_MAX and not _RE_SENTENCE_MARK.search(s):
        return True

    # ---- 兜底：完全没有实质内容 ----
    # 既无中日韩字符、也无 3 字母以上英文单词 → 无信息量。
    if not _RE_HAS_CJK.search(s) and not _RE_HAS_WORD.search(s):
        return True

    return False


def clean_snippet(text: str, *, enable: bool | None = None) -> str:
    """清洗 web snippet 里的页面模板噪声。

    处理流程：
        1. 短文本直接放过（DDG/Serper 的摘要本身就干净）
        2. 按 `[...]`（Tavily 的片段拼接标记）与换行拆成行
        3. 逐行过噪声判定
        4. **行级去重** —— Tavily 拼接片段时常把同一段重复两遍
        5. 兜底：保留比例过低 → 放弃清洗，返回原文

    Args:
        text:   原始 snippet。
        enable: None → 读 `ENABLE_SNIPPET_CLEAN`；显式传值便于测试与 A/B。

    Returns:
        清洗后的文本。任何降级情形都返回**原文**（绝不返回空串）。
    """
    on = ENABLE_SNIPPET_CLEAN if enable is None else enable
    if not on or not text:
        return text

    # 短文本不动：这类是搜索引擎抽好的句子，清洗只有风险没有收益。
    if len(text) < MIN_LEN_TO_CLEAN:
        return text

    # `[...]` 是 Tavily 拼接多个正文片段的标记（实测 17/20 条都有）。
    # 先转成换行，让下面的行级处理能正确切分片段边界 —— 否则两个片段
    # 会被当成同一行，一行里混着噪声和正文就没法分开处理。
    normalized = text.replace("[...]", "\n")

    kept: list[str] = []
    seen: set[str] = set()
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if _is_noise_line(line):
            continue

        # ---- 行级去重 ----
        # Tavily 拼接片段时常把同一段落重复输出。实测样本里
        # `## 解读巴厘岛的季节：旱季与雨季` 连续出现两次、
        # `以下是巴厘岛每月天气的快速细分：` 出现两次。
        # 用**去掉空白后**的文本做 key，避免仅因空格不同而漏判。
        key = re.sub(r"\s+", "", line)
        if key in seen:
            continue
        seen.add(key)
        kept.append(line)

    out = "\n".join(kept)

    # ---- 兜底：规则过激时放弃清洗 ----
    # 见模块 docstring「宁可漏删，不可错删」。带噪声的证据仍然可用，
    # 被洗空的证据则等于丢了这一路召回。
    if not out or len(out) < len(text) * MIN_KEEP_RATIO:
        return text
    return out


def clean_stats(text: str) -> dict:
    """返回清洗前后的统计（供观测 / A-B 对比，不参与主链路）。

    上线后可把这个打进日志，观察各站点的噪声占比：
      * 某域名 `removed_ratio` 长期为 0 → 规则对它无效，可据此补规则；
      * 某域名接近触发兜底（>0.7）      → 规则对它过激，需要收窄。
    """
    cleaned = clean_snippet(text, enable=True)
    raw_lines = [x.strip() for x in text.replace("[...]", "\n").split("\n")]
    raw_lines = [x for x in raw_lines if x]
    kept_lines = [x for x in cleaned.split("\n") if x.strip()]
    return {
        "raw_len": len(text),
        "clean_len": len(cleaned),
        "removed_chars": len(text) - len(cleaned),
        "removed_ratio": round(1 - len(cleaned) / max(len(text), 1), 4),
        "raw_lines": len(raw_lines),
        "kept_lines": len(kept_lines),
        "fell_back": cleaned == text and len(text) >= MIN_LEN_TO_CLEAN,
    }
