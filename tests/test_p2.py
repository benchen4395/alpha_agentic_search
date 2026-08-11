# test_p2.py
"""P2 改造的回归测试。

════════════════════════════════════════════════════════════════════════
覆盖范围
════════════════════════════════════════════════════════════════════════
近重去重 + MMR（rag/dedup.py）
    TestNearDupDedup     —— 转载被删、不同答案绝不误删、降级路径
    TestMMRRerank        —— 相关性量纲归一化、λ=1 退化、首段取最相关
    TestDiversifyE2E     —— 两阶段串联（MMR 单独不够，这是设计核心）
    TestLayeredDedup     —— 只对 L4 编码（延迟优化）、向量只编一次、metadata 不污染
    TestRetrieverWiring  —— 候选池放大、开关全关时零行为差异

追问推荐（followup.py）
    TestFollowupParse    —— 剥离、降级、坏输出过滤
    TestStreamFilter     —— 分隔符被切碎也不能漏给用户（最容易漏的坑）
    TestClarify          —— 默认关闭、保守判定
    TestFollowupAgent    —— 端到端：memory / L1 缓存绝不能被追问区污染
    TestFollowupFrontend —— CLI / Web / scripts 渲染契约

其他
    TestSnippetClean     —— L4 snippet 噪声清洗（真实样本标定）

════════════════════════════════════════════════════════════════════════
为什么单独开一个文件而不加到 test_p0.py
════════════════════════════════════════════════════════════════════════
test_p0.py 已经 2800+ 行。P2 与 P0 的关注点、fixture 需求都不同
（需要真实 BGE-M3 向量，P0 大量用 mock），混在一起会让
"只想跑某一块"变得困难。分文件后可以 `pytest test_p2.py -k NearDup`
精准定位。

⚠️ 部分测试需要真实 BGE-M3（首次加载约 5s，之后进程内复用）。
用 module 级 fixture 保证只加载一次。
"""
from __future__ import annotations

import os as _os

# ── 源码路径解析 ────────────────────────────────────────────────────────
# 有几个测试是**读源码文本**做静态断言（验证某段代码确实被接线了）。
# 原来写的是相对路径 `open("rag/layers.py")`，这隐含假设"pytest 必须从
# 仓库根目录启动" —— 换个 cwd（如在 tests/ 里跑）就 FileNotFoundError。
# 这里改成以本文件位置为锚点，与 cwd 彻底解耦。
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


def _repo(rel: str) -> str:
    """把仓库相对路径解析为绝对路径。"""
    return _os.path.join(_REPO_ROOT, rel)


import math as _math
from pathlib import Path as _Path

import pytest


# ════════════════════════════════════════════════════════════════════════
#                              公共 fixture
# ════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def emb():
    """真实 BGE-M3 编码器（module 级：整个文件只加载一次）。

    为什么不用 mock：本模块测的核心就是**语义相似度阈值**是否合适
    （0.95 能分开"转载"与"不同报道"吗？）。用假向量测阈值毫无意义 ——
    那只是在测我自己编的数字。这类"阈值标定"必须用真模型。
    """
    from src.rag.embedder import Embedder
    return Embedder()


def _P(title, text, url="", score=1.0, layer="L4_web"):
    """构造 Passage 的简写。"""
    from src.rag.types import Passage
    return Passage(text=text, title=title, url=url, score=score, layer=layer)


@pytest.fixture()
def reposts():
    """同一条新闻的 3 家转载（真实场景里最常见的重复形态）。"""
    return [
        _P("我国上半年GDP同比增长5.3%",
           "国家统计局今日发布数据，上半年国内生产总值同比增长5.3%，"
           "增速比一季度加快0.1个百分点，国民经济运行总体平稳。",
           "https://www.news.cn/2026/gdp.html", score=0.030),
        _P("我国上半年GDP同比增长5.3%",
           "国家统计局今日发布数据，上半年国内生产总值同比增长5.3%，"
           "增速比一季度加快0.1个百分点，国民经济运行总体平稳。",
           "https://people.com.cn/2026/gdp.html", score=0.028),
        _P("上半年GDP同比增长5.3%",
           "统计局发布：上半年国内生产总值同比增长5.3%，比一季度加快"
           "0.1个百分点，经济运行总体平稳。",
           "https://tv.cctv.com/2026/gdp.html", score=0.026),
    ]


@pytest.fixture()
def complements():
    """与上面转载**互补**的证据（讲的是别的侧面，绝不该被删）。"""
    return [
        _P("上半年三次产业增加值分别增长",
           "分产业看，第一产业增加值增长3.5%，第二产业增长5.8%，"
           "第三产业增长5.2%。工业对经济增长的贡献率明显提升。",
           "https://www.stats.gov.cn/2026/industry.html", score=0.024),
        _P("居民收入", "上半年全国居民人均可支配收入增长5.4%。",
           "https://d.com/income", score=0.020),
    ]


# ════════════════════════════════════════════════════════════════════════
#                    ①：近重去重（硬删，事实判断）
# ════════════════════════════════════════════════════════════════════════
class TestNearDupDedup:
    """余弦 >= 0.95 判为"同一份文本"并硬删。

    这一组测试的核心矛盾是**代价不对称**：
        漏删一条转载 → 浪费 1 个证据席位
        误删一条独立证据 → 可能丢掉答案本身
    所以 `test_different_answers_never_dropped` 是本组最重要的一条，
    它守的是"不可误删"这条底线。
    """

    def test_reposts_are_dropped(self, emb, reposts, complements):
        """同一新闻的多家转载必须被压缩成 1 条代表。"""
        from src.rag.dedup import drop_near_duplicates
        cands = reposts + complements[:1]
        kept = drop_near_duplicates(cands, embedder=emb, verbose=False)
        assert len(kept) == 2, f"期望 3 转载压成 1 条，实际保留 {len(kept)}"
        # 保留的必须是**排名最高**的那条转载（RRF 顺序天然是更好的代表）
        assert kept[0].url.endswith("news.cn/2026/gdp.html")
        # 互补的那条绝不能被误删
        assert any("三次产业" in p.title for p in kept)

    def test_different_answers_never_dropped(self, emb):
        """⚠️ 最重要：语义相近但**答案不同**的段落绝不能删。

        这几对在 BGE-M3 上的余弦实测 0.85~0.88（见 cache_policy.py 的
        标定数据），**且刻意用同域名**以触发"同域放宽 0.02"的分支。
        若被误删，系统会返回错误答案 —— 比不去重严重得多。
        """
        from src.rag.dedup import drop_near_duplicates
        tricky = [
            _P("苹果公司CEO", "苹果公司的首席执行官是蒂姆·库克，他于2011年接任。",
               "https://a.com/ceo", score=0.030),
            _P("苹果公司CFO", "苹果公司的首席财务官是卢卡·梅斯特里，负责财务事务。",
               "https://a.com/cfo", score=0.028),
            _P("2024年GDP", "2024年国内生产总值为134.9万亿元。",
               "https://b.com/2024", score=0.026),
            _P("2025年GDP", "2025年国内生产总值为140.2万亿元。",
               "https://b.com/2025", score=0.024),
        ]
        kept = drop_near_duplicates(tricky, embedder=emb, verbose=False)
        assert len(kept) == 4, (
            f"误删了不同答案的段落，只剩 {[p.title for p in kept]}"
        )

    def test_graceful_degradation_without_embedder(self, reposts):
        """拿不到向量时必须**原样返回**，不能抛异常也不能删。

        这是"优化项不该变成故障点"的具体体现：与 `_safe_search`
        的取向一致 —— 宁可不优化，也不能让主链路失败。
        """
        from src.rag.dedup import drop_near_duplicates
        out = drop_near_duplicates(reposts, embedder=None, verbose=False)
        assert len(out) == len(reposts)
        assert out is reposts or [p.url for p in out] == [p.url for p in reposts]

    def test_single_or_empty_input(self, emb):
        """0/1 条输入直接短路，不该触发编码。"""
        from src.rag.dedup import drop_near_duplicates
        assert drop_near_duplicates([], embedder=emb) == []
        one = [_P("t", "x")]
        assert drop_near_duplicates(one, embedder=emb) == one

    def test_reuses_metadata_embedding(self, reposts):
        """metadata 里已有向量时必须**复用**，不再调编码器（零成本路径）。

        这是性能上的关键优化：L2/L3 检索时本来就编码过 passage，
        若它们把向量写进 metadata，去重就完全免费。
        """
        from src.rag.dedup import drop_near_duplicates

        class _CountingEmb:
            def __init__(self):
                self.calls = 0

            def embed_batch(self, texts):
                self.calls += 1
                return [[1.0, 0.0, 0.0] for _ in texts]

        # 手动给每条塞上向量：前两条完全相同 → 应被判为近重
        for i, p in enumerate(reposts):
            p.metadata["embedding"] = [1.0, 0.0] if i < 2 else [0.0, 1.0]
        ce = _CountingEmb()
        kept = drop_near_duplicates(reposts, embedder=ce, verbose=False)
        assert ce.calls == 0, "metadata 里有向量却仍调用了编码器"
        assert len(kept) == 2

    def test_mixed_dim_metadata_is_rejected(self, reposts):
        """metadata 里向量维度不一致时整体降级（不可比，算余弦无意义）。"""
        from src.rag.dedup import drop_near_duplicates
        reposts[0].metadata["embedding"] = [1.0, 0.0]
        reposts[1].metadata["embedding"] = [1.0, 0.0, 0.0]   # 维度不同
        reposts[2].metadata["embedding"] = [0.0, 1.0]
        out = drop_near_duplicates(reposts, embedder=None, verbose=False)
        assert len(out) == 3, "维度不一致应整体降级为不去重"


# ════════════════════════════════════════════════════════════════════════
#                    ②：MMR（软重排，偏好判断）
# ════════════════════════════════════════════════════════════════════════
class TestMMRRerank:
    """MMR = λ·Rel − (1−λ)·max Sim。

    这一组主要守两件事：
      ① 相关性分数的**量纲**必须被归一化（RRF 分数只有 ~0.02，
         不归一化会让多样性项完全压倒相关性项）；
      ② λ=1.0 必须逐段退化为原排序（"关闭 MMR"的正确方式）。
    """

    @staticmethod
    def _cands():
        return [
            _P("GDP总量", "上半年国内生产总值同比增长5.3%。",
               "https://a.com/1", score=0.0303),
            _P("GDP总量重复", "上半年GDP同比增长5.3%，增速平稳。",
               "https://b.com/2", score=0.0294),
            _P("三次产业", "第一产业增长3.5%，第二产业5.8%，第三产业5.2%。",
               "https://c.com/3", score=0.0286),
            _P("居民收入", "上半年全国居民人均可支配收入增长5.4%。",
               "https://d.com/4", score=0.0278),
        ]

    def test_first_pick_is_most_relevant(self, emb):
        """首段必须是相关性最高的（此时已选集为空，公式退化为 argmax Rel）。"""
        from src.rag.dedup import mmr_rerank
        out = mmr_rerank(self._cands(), embedder=emb, top_k=3)
        assert out[0].title == "GDP总量"

    def test_relevance_not_crushed_by_scale(self, emb):
        """⚠️ 量纲归一化必须生效。

        RRF 分数量级 ~0.02，而余弦 ~0.9。若不归一化：
            λ·0.0294 = 0.021   vs   (1−λ)·0.913 = 0.274
        惩罚项大一个数量级 → MMR 退化成"只挑最不相似的"，
        相关性排序被整个丢掉。

        判据：相关性仍在起作用 —— rel 更高的「GDP总量重复」
        应排在 rel 更低的「三次产业」之前。

        ⚠️ 注意这**不是**在说"重复项该被选中"。实测（λ=0.7）：
            GDP总量重复  MMR = 0.7*0.640 − 0.3*0.913 = 0.174
            三次产业     MMR = 0.7*0.320 − 0.3*0.704 = 0.013
        所以 MMR 确实该先选重复项 —— 这正是
        `test_mmr_alone_is_insufficient` 要记录的设计依据。
        """
        from src.rag.dedup import mmr_rerank
        out = mmr_rerank(self._cands(), embedder=emb, top_k=3)
        titles = [p.title for p in out]
        assert titles[1] == "GDP总量重复", (
            f"相关性项被量纲压倒：第 2 段是 {titles[1]!r}"
        )

    def test_lambda_one_is_identity(self, emb):
        """λ=1.0 → 多样性项系数为 0 → 逐段等价于原排序。"""
        from src.rag.dedup import mmr_rerank
        cands = self._cands()
        out = mmr_rerank(cands, embedder=emb, top_k=4, lambda_=1.0)
        assert [p.title for p in out] == [p.title for p in cands]

    def test_diversity_wins_at_low_lambda(self, emb):
        """λ 很小时多样性主导，重复项应被压到后面（验证 λ 真的是旋钮）。"""
        from src.rag.dedup import mmr_rerank
        out = mmr_rerank(self._cands(), embedder=emb, top_k=2, lambda_=0.1)
        assert out[1].title != "GDP总量重复", (
            "λ=0.1 时多样性应主导，重复项不该排第 2"
        )

    def test_graceful_degradation(self):
        """无 embedder → 原序截断，不抛异常。"""
        from src.rag.dedup import mmr_rerank
        cands = self._cands()
        out = mmr_rerank(cands, embedder=None, top_k=2)
        assert [p.title for p in out] == [p.title for p in cands[:2]]

    def test_identical_scores_no_crash(self, emb):
        """所有分数相同时（min==max）不能除零。"""
        from src.rag.dedup import mmr_rerank
        cands = self._cands()
        for p in cands:
            p.score = 0.02
        out = mmr_rerank(cands, embedder=emb, top_k=3)
        assert len(out) == 3


# ════════════════════════════════════════════════════════════════════════
#              ③：两阶段串联（本设计的核心价值）
# ════════════════════════════════════════════════════════════════════════
class TestDiversifyE2E:
    def test_mmr_alone_is_insufficient(self, emb, reposts, complements):
        """⚠️ 记录设计依据：**单靠 MMR 挡不住转载**。

        这条测试的价值不是"验证功能"，而是**锁定一个反直觉的事实**：
        MMR 的惩罚是连续的，对余弦 0.98 的转载只施加"较大惩罚"，
        若其相关性也高，仍会被选中。所以两阶段（硬去重 → MMR）
        不是冗余设计，而是必要的。

        如果哪天有人想"简化"掉阶段 A，这条测试会立刻失败并解释原因。
        """
        from src.rag.dedup import mmr_rerank
        cands = reposts + complements
        only_mmr = mmr_rerank(cands, embedder=emb, top_k=3)
        repost_urls = {p.url for p in reposts}
        n = sum(1 for p in only_mmr if p.url in repost_urls)
        assert n > 1, (
            "前提失效：MMR 单独应该会让多条转载入选。"
            "若此断言失败，说明 λ 或模型变了，需重新标定阈值"
        )

    def test_two_stage_compresses_reposts(self, emb, reposts, complements):
        """两阶段串联后转载只占 1 席，腾出的席位装互补信息。"""
        from src.rag.dedup import diversify
        cands = reposts + complements
        out = diversify(cands, embedder=emb, top_k=3, verbose=False)
        repost_urls = {p.url for p in reposts}
        n = sum(1 for p in out if p.url in repost_urls)
        assert n == 1, f"转载仍占 {n} 席（应恰好 1 席代表）"
        assert len(out) == 3, "席位应被填满（去重后仍有足够候选）"

    def test_both_disabled_is_identity(self, emb, reposts, complements):
        """两个开关都关时必须**逐段等价**于原输入（零回归保证）。"""
        from src.rag.dedup import diversify
        cands = reposts + complements
        out = diversify(cands, embedder=emb, top_k=3,
                        enable_near_dup=False, enable_mmr=False, verbose=False)
        assert [p.url for p in out] == [p.url for p in cands[:3]]

    def test_empty_input(self, emb):
        from src.rag.dedup import diversify
        assert diversify([], embedder=emb, top_k=3) == []


# ════════════════════════════════════════════════════════════════════════
#          ⑤：分层去重（方案 B —— 只对 L4 编码，降延迟）
# ════════════════════════════════════════════════════════════════════════
class TestLayeredDedup:
    """只对 `DEDUP_LAYERS`（默认 {"L4_web"}）做语义去重。

    动机是纯粹的延迟：编码约 80ms/段，而转载重复几乎只发生在 web 检索。
    这一组测试守两件事：
      ① **省下的编码真的省掉了**（零编码路径必须零调用）——
         这是本优化的全部价值所在，必须用调用计数硬验证，
         不能只看"结果对不对"（结果对但仍编码了 18 段 = 优化失效）；
      ② **非目标层的段落一条都不能丢**（不能为了省时误删证据）。
    """

    @staticmethod
    def _spy():
        """返回一个记录编码段数的 embedder 包装。

        用真实 BGE-M3 而不是假向量：本组要验证的是"有没有调用"，
        但结果正确性仍依赖真实语义（转载必须被认出来）。
        """
        from src.rag.embedder import Embedder

        class _Spy:
            def __init__(self):
                self._e = Embedder()
                self.encoded = 0
                self.calls = 0

            def embed_batch(self, texts):
                self.calls += 1
                self.encoded += len(texts)
                return self._e.embed_batch(texts)

            def embed(self, text):
                return self._e.embed(text)

        return _Spy()

    def test_offline_only_zero_encoding(self):
        """⚠️ 核心收益：纯离线命中（无 L4）时**一次编码都不该发生**。

        这是最常见的场景（离线三层置信度够高、L4 不触发），
        也是本优化省时最多的地方 —— 从 1691ms 直接降到 0。
        且此时本来就没有转载问题，省下的编码是纯粹的浪费消除。
        """
        from src.rag.dedup import diversify
        spy = self._spy()
        offline = [
            _P("量子计算原理", "量子计算利用量子叠加与纠缠进行并行计算。",
               "", 0.030, layer="L2_wiki"),
            _P("量子比特", "量子比特是量子信息的基本单位。",
               "", 0.028, layer="L2_wiki"),
            _P("历史问答", "历史问答：\nQ: 量子计算\nA: 一种计算范式。",
               "", 0.026, layer="L3_history"),
            _P("量子计算机 —— 类型：计算设备", "量子计算机 —— 类型：计算设备",
               "", 0.024, layer="L5_kg"),
        ]
        out = diversify(offline, embedder=spy, top_k=6, verbose=False)
        assert spy.calls == 0, f"离线场景仍编码了 {spy.encoded} 段（优化失效）"
        assert len(out) == 4, "非目标层的段落不能被丢弃"
        assert [p.title for p in out] == [p.title for p in offline]

    def test_only_l4_encoded(self, reposts, complements):
        """混合候选时，**只有 L4 的段落被送去编码**。"""
        from src.rag.dedup import diversify
        spy = self._spy()
        offline = [
            _P("维基百科：国内生产总值", "国内生产总值是衡量经济规模的指标。",
               "", 0.022, layer="L2_wiki"),
            _P("历史问答", "历史问答：\nQ: GDP\nA: 经济总量指标。",
               "", 0.018, layer="L3_history"),
        ]
        cands = reposts + complements + offline   # 3 转载 + 2 互补 + 2 离线
        out = diversify(cands, embedder=spy, top_k=6, verbose=False)
        n_l4 = sum(1 for p in cands if p.layer == "L4_web")
        assert spy.encoded == n_l4, (
            f"编码了 {spy.encoded} 段，应只编码 L4 的 {n_l4} 段"
        )
        # 转载仍被正确压缩
        repost_urls = {p.url for p in reposts}
        assert sum(1 for p in out if p.url in repost_urls) == 1

    def test_offline_passages_never_dropped(self, reposts):
        """⚠️ 底线：非目标层的段落绝不能因去重而消失。

        分层模式的实现要"拆开处理再合并"，最容易出的 bug 就是
        合并时漏掉 others。这条测试直接守这个。
        """
        from src.rag.dedup import diversify
        offline = [
            _P("离线证据A", "第一产业增加值增长3.5%。", "", 0.021, layer="L2_wiki"),
            _P("离线证据B", "居民可支配收入增长5.4%。", "", 0.019, layer="L3_history"),
            _P("离线证据C", "GDP —— 单位：万亿元", "", 0.017, layer="L5_kg"),
        ]
        out = diversify(reposts + offline, embedder=self._spy(),
                        top_k=10, verbose=False)
        titles = {p.title for p in out}
        for p in offline:
            assert p.title in titles, f"离线段落 {p.title} 被误删"

    def test_full_mode_encodes_everything(self, reposts):
        """全量模式（target_layers=frozenset()）应编码所有候选，**且只编一次**。

        这是"更彻底但更慢"的逃生舱，必须仍然可用。
        """
        from src.rag.dedup import diversify
        spy = self._spy()
        offline = [_P("离线", "维基内容。", "", 0.02, layer="L2_wiki")]
        cands = reposts + offline
        diversify(cands, embedder=spy, top_k=6,
                  target_layers=frozenset(), verbose=False)
        assert spy.encoded == len(cands), (
            f"全量模式编码了 {spy.encoded} 段，应恰好 {len(cands)} 段"
            f"（大于则说明两阶段各编了一遍）"
        )

    def test_vectors_encoded_only_once(self, reposts, complements):
        """⚠️ 两阶段串联时向量只能编码**一次**。

        这是回归测试真实抓到的 bug：`drop_near_duplicates` 和
        `mmr_rerank` 各自都会调 `_resolve_vectors`，不做处理就是
            阶段 A 编 N 段 → 删掉 k 段 → 阶段 B 又编 N−k 段
        实测 5 段候选编了 8 段，按 80ms/段 算白花约 240ms ——
        恰好抵消掉"只对 L4 去重"省下的一部分。

        修法是先统一编码并写进 metadata，让两阶段走复用分支。
        """
        from src.rag.dedup import diversify
        spy = self._spy()
        l4 = reposts + complements          # 5 段，全是 L4
        diversify(l4, embedder=spy, top_k=3, verbose=False)
        assert spy.encoded == len(l4), (
            f"编码了 {spy.encoded} 段，应恰好 {len(l4)} 段（两阶段重复编码）"
        )
        assert spy.calls == 1, f"调了 {spy.calls} 次 embed_batch，应只 1 次"

    def test_metadata_not_polluted_by_vectors(self, reposts, complements):
        """⚠️ 借用的向量必须从 metadata 里清掉。

        metadata 会一路带到 `Source` → `AnswerResult.to_dict()` → JSON。
        1024 个 float 序列化出来是几十 KB，会让
        `scripts/search.py` 的输出膨胀几十倍，也会进 L3 归档。
        """
        from src.rag.dedup import diversify, EMBEDDING_META_KEY
        cands = reposts + complements
        out = diversify(cands, embedder=self._spy(), top_k=3, verbose=False)
        for p in out:
            assert EMBEDDING_META_KEY not in p.metadata, (
                f"向量泄漏到 metadata（{p.title}），会让 JSON 输出膨胀"
            )
        # 被删掉的那几段也要清干净（它们可能被调用方另存了引用）
        for p in cands:
            assert EMBEDDING_META_KEY not in p.metadata

    def test_caller_embedding_preserved(self, reposts):
        """调用方**本来就写好**的向量不能被清掉。

        清理逻辑只能删"自己塞进去的"，否则会破坏调用方的数据。
        """
        from src.rag.dedup import diversify, EMBEDDING_META_KEY
        for p in reposts:
            p.metadata[EMBEDDING_META_KEY] = [1.0, 0.0, 0.0]
        diversify(reposts, embedder=None, top_k=3, verbose=False)
        for p in reposts:
            assert p.metadata.get(EMBEDDING_META_KEY) == [1.0, 0.0, 0.0], (
                "调用方原有的向量被误删"
            )

    def test_preserves_cross_layer_interleaving(self, reposts):
        """⚠️ 合并必须**保留原始的跨层交错位置**，而不是简单拼接。

        输入顺序是 quota_fuse 的相关性降序，L4 段落与离线段落是**交错**的。
        两种直觉做法都是错的：
          ✗ 按 score 重排 → 抹掉 MMR 的成果（MMR 的产出就是一个顺序）
          ✗ kept + others → 把所有 L4 无条件排到离线之前

        正确做法是"槽位回填"：目标层段落只在自己原来占的位置上重新排列。
        这里构造 [L4, 离线, L4, L4, 离线]，验证离线段落的位置不变。
        """
        from src.rag.dedup import diversify
        cands = [
            reposts[0],                                              # idx0 L4
            _P("离线甲", "维基内容甲：国内生产总值定义。", "", 0.029, layer="L2_wiki"),
            reposts[1],                                              # idx2 L4
            reposts[2],                                              # idx3 L4
            _P("离线乙", "维基内容乙：三次产业划分。", "", 0.018, layer="L2_wiki"),
        ]
        out = diversify(cands, embedder=self._spy(), top_k=10, verbose=False)
        titles = [p.title for p in out]
        # 3 条转载压成 1 条 → 共 3 段
        assert len(out) == 3, f"期望 3 段，实际 {titles}"
        # 离线段落必须仍在 L4 代表之后 / 之间，相对次序不变
        i_a, i_b = titles.index("离线甲"), titles.index("离线乙")
        assert i_a < i_b, "离线段落之间的相对顺序被改变"
        # 第一段应仍是 L4 代表（它在原列表里就排第一）
        assert out[0].layer == "L4_web", f"首段变成了 {out[0].layer}"

    def test_mmr_order_not_destroyed_by_merge(self, reposts, complements):
        """⚠️ 合并不能按 score 重排 —— 那会把 MMR 的产出抹掉。

        判据：让 MMR 真正改变目标层顺序（λ 调小放大多样性权重），
        然后确认合并后的 L4 段落顺序**不等于**按 score 降序。
        """
        from src.rag.dedup import diversify, mmr_rerank
        l4 = reposts[:1] + complements     # 1 转载 + 2 互补，都是 L4
        offline = [_P("离线", "维基内容。", "", 0.025, layer="L2_wiki")]
        cands = l4 + offline

        out = diversify(cands, embedder=self._spy(), top_k=10, verbose=False)
        out_l4 = [p.title for p in out if p.layer == "L4_web"]
        # 单独跑 MMR 得到的目标层顺序，应与合并后 L4 的相对顺序一致
        expect = [p.title for p in mmr_rerank(l4, embedder=self._spy())]
        assert out_l4 == expect, (
            f"合并破坏了 MMR 顺序：合并后 {out_l4}，MMR 产出 {expect}"
        )

    def test_no_double_truncation(self, reposts):
        """⚠️ 分层模式不能"双重截断"。

        实现上若先把目标层裁到 top_k、合并后再裁一次，L4 会被过度压缩。
        这里给足量候选，验证最终恰好填满 top_k。
        """
        from src.rag.dedup import diversify
        offline = [
            _P(f"离线{i}", f"这是第{i}条维基材料，内容各不相同。",
               "", 0.02 - i * 0.001, layer="L2_wiki")
            for i in range(5)
        ]
        out = diversify(reposts + offline, embedder=self._spy(),
                        top_k=6, verbose=False)
        assert len(out) == 6, f"席位未填满，只剩 {len(out)} 段"

    def test_config_parses_layers(self):
        from src.rag import config as rc
        assert isinstance(rc.DEDUP_LAYERS, frozenset)
        assert "L4_web" in rc.DEDUP_LAYERS, "默认应只对 L4 去重"

    def test_explicit_empty_env_means_full_mode(self, monkeypatch):
        """显式 RAG_DEDUP_LAYERS="" 必须解析成空集合（全量模式）。

        ⚠️ 这里有个易错点：若写成 `getenv(k, "L4_web").split(",")`，
        空串会得到 [""] —— 一个含空字符串的集合，永远匹配不上任何
        layer，效果是"彻底关掉去重"，与用户想要的"对所有层去重"
        语义正好相反。
        """
        import importlib
        from src.rag import config as rc
        monkeypatch.setenv("RAG_DEDUP_LAYERS", "")
        importlib.reload(rc)
        try:
            assert rc.DEDUP_LAYERS == frozenset(), (
                f"空串应解析为空集合，实际 {rc.DEDUP_LAYERS}"
            )
        finally:
            monkeypatch.delenv("RAG_DEDUP_LAYERS", raising=False)
            importlib.reload(rc)

    def test_custom_layers_from_env(self, monkeypatch):
        """支持逗号分隔的自定义层集合。"""
        import importlib
        from src.rag import config as rc
        monkeypatch.setenv("RAG_DEDUP_LAYERS", "L4_web, L3_history")
        importlib.reload(rc)
        try:
            assert rc.DEDUP_LAYERS == frozenset({"L4_web", "L3_history"})
        finally:
            monkeypatch.delenv("RAG_DEDUP_LAYERS", raising=False)
            importlib.reload(rc)

    def test_switches_off_still_zero_encoding(self, reposts):
        """两开关全关时不该编码（零回归 + 零成本）。"""
        from src.rag.dedup import diversify
        spy = self._spy()
        out = diversify(reposts, embedder=spy, top_k=3,
                        enable_near_dup=False, enable_mmr=False, verbose=False)
        assert spy.calls == 0
        assert [p.url for p in out] == [p.url for p in reposts[:3]]


# ════════════════════════════════════════════════════════════════════════
#                 ④：retriever 接线（候选池放大）
# ════════════════════════════════════════════════════════════════════════
class TestRetrieverWiring:
    def test_config_switches_exist(self):
        from src.rag import config as rc
        assert isinstance(rc.ENABLE_NEAR_DUP, bool)
        assert isinstance(rc.ENABLE_MMR, bool)
        assert rc.FUSION_CANDIDATE_MULTIPLIER >= 1

    def test_candidate_pool_is_enlarged(self):
        """⚠️ 去重会**删**段落，所以融合阶段必须多召回。

        若仍只融合 FUSION_TOP_K=6 就去重，删掉 3 条转载后只剩 3 段 ——
        席位不是"腾给更好的证据"，而是**凭空消失**，
        那样去重反而降低信息量，完全违背初衷。

        这里用源码断言（而非跑真实检索）：跑真检索需要 GB 级索引，
        在 CI 里不可行；而"倍数被用上了"是纯粹的接线正确性问题。
        """
        src = open(_repo("src/rag/retriever.py"), encoding="utf-8").read()
        assert "FUSION_CANDIDATE_MULTIPLIER" in src
        assert "_cand_k" in src
        # 必须传放大后的 _cand_k 给 quota_fuse，而不是 self.fusion_top_k
        assert "top_k=_cand_k" in src
        # diversify 必须收窄回 fusion_top_k
        assert "top_k=self.fusion_top_k" in src

    def test_dedup_runs_before_rerank(self):
        """去重必须在 rerank **之前**：rerank 是最贵的一步，先减候选省钱。"""
        src = open(_repo("src/rag/retriever.py"), encoding="utf-8").read()
        i_dedup = src.index("rag_dedup.diversify")
        i_rerank = src.index("fused = rerank(query")
        assert i_dedup < i_rerank, "diversify 必须在 rerank 之前"

    def test_dedup_after_quota_fuse(self):
        """去重必须在配额融合**之后**：否则弱实体的证据可能先被当近重删掉。"""
        src = open(_repo("src/rag/retriever.py"), encoding="utf-8").read()
        i_quota = src.index("fused = quota_fuse")
        i_dedup = src.index("rag_dedup.diversify")
        assert i_quota < i_dedup, "quota_fuse 必须在 diversify 之前"


# ════════════════════════════════════════════════════════════════════════
#                     ①：追问解析与过滤
# ════════════════════════════════════════════════════════════════════════
class TestFollowupParse:
    def test_normal_parse(self):
        from src.pipeline.followup import parse_followups, FOLLOWUP_MARKER
        raw = (
            "上半年GDP同比增长5.3%[1]。\n"
            f"{FOLLOWUP_MARKER}\n"
            "1. 三次产业各自的增速是多少？\n"
            "2. 与去年同期相比表现如何？\n"
            "- 居民收入增长了多少？\n"
        )
        r = parse_followups(raw, "上半年GDP增长多少")
        assert r.body == "上半年GDP同比增长5.3%[1]。"
        assert FOLLOWUP_MARKER not in r.body
        assert len(r.followups) == 3
        # 序号/项目符号必须被清洗掉
        assert r.followups[0] == "三次产业各自的增速是多少？"
        assert r.followups[2] == "居民收入增长了多少？"

    def test_no_marker_returns_intact(self):
        """⚠️ 降级保证：没有分隔符时必须**原样返回全文**。

        这是本函数最重要的性质。追问是锦上添花，答案是刚需 ——
        绝不能因为"解析追问失败"而丢掉、截断或污染答案内容。
        """
        from src.pipeline.followup import parse_followups
        plain = "这是一个没有追问区的普通答案。"
        r = parse_followups(plain, "问题")
        assert r.body == plain
        assert r.followups == []

    def test_empty_input(self):
        from src.pipeline.followup import parse_followups
        r = parse_followups("", "q")
        assert r.body == "" and r.followups == []

    @pytest.mark.parametrize("bad,why", [
        ("上半年GDP增长多少", "与用户原问题重复"),
        ("它的增速呢？", "指代词开头，无法独立成为 query"),
        ("三次产业的构成", "陈述句而非问句"),
        ("这是一个非常非常长的问题需要超过四十个字符才能触发长度过滤规则所以继续写",
         "超长，说明模型没理解'简短问句'"),
    ])
    def test_bad_followups_filtered(self, bad, why):
        from src.pipeline.followup import parse_followups, FOLLOWUP_MARKER
        raw = f"答案内容。\n{FOLLOWUP_MARKER}\n{bad}\n"
        r = parse_followups(raw, "上半年GDP增长多少")
        assert r.followups == [], f"未过滤掉「{why}」的输出: {r.followups}"

    def test_dedup_similar_followups(self):
        """归一化后相同的追问只保留一条。"""
        from src.pipeline.followup import parse_followups, FOLLOWUP_MARKER
        raw = (f"答案。\n{FOLLOWUP_MARKER}\n"
               "三次产业增速是多少？\n三次产业增速是多少\n")
        r = parse_followups(raw, "GDP")
        assert len(r.followups) == 1

    def test_count_capped(self):
        """最多返回 FOLLOWUP_COUNT 条（防止模型输出一长串）。"""
        from src.pipeline.followup import parse_followups, FOLLOWUP_MARKER, FOLLOWUP_COUNT
        qs = "\n".join(f"第{i}个问题是什么？" for i in range(1, 10))
        r = parse_followups(f"答案。\n{FOLLOWUP_MARKER}\n{qs}", "q")
        assert len(r.followups) <= FOLLOWUP_COUNT


# ════════════════════════════════════════════════════════════════════════
#            ②：流式分隔符抑制（最容易漏的坑）
# ════════════════════════════════════════════════════════════════════════
class TestStreamFilter:
    """流式下分隔符会被**切碎**在多个 chunk 里。

    逐 chunk 做 `in` 判断必然漏检，碎片会漏给用户。
    `StreamFilter` 用"滞后输出 len(marker)-1 个字符"解决。
    """

    def test_fragmented_marker_blocked(self):
        """⚠️ 核心：分隔符被切成碎片也不能漏给用户。"""
        from src.pipeline.followup import StreamFilter, FOLLOWUP_MARKER
        chunks = ["上半年GDP同比", "增长5.3%[1]。", "\n#", "##FOL", "LOW",
                  "UP##", "#\n三次产业增速?", "\n居民收入增长多少?"]
        f = StreamFilter()
        visible = "".join(f.feed(c) for c in chunks) + f.flush()
        assert FOLLOWUP_MARKER not in visible
        assert "#" not in visible, f"分隔符碎片泄漏: {visible!r}"
        assert "三次产业" not in visible, "追问内容泄漏到正文"
        assert visible == "上半年GDP同比增长5.3%[1]。\n"
        r = f.result("GDP增长多少")
        assert len(r.followups) == 2

    def test_plain_text_passthrough_intact(self):
        """无追问区时（最常见路径）正文必须逐字完整。"""
        from src.pipeline.followup import StreamFilter
        f = StreamFilter()
        chunks = ["这是", "一个普通", "答案，没有追问区。"]
        got = "".join(f.feed(c) for c in chunks) + f.flush()
        assert got == "".join(chunks)

    def test_single_char_chunks(self):
        """极端情况：每个 chunk 只有 1 个字符。"""
        from src.pipeline.followup import StreamFilter, FOLLOWUP_MARKER
        raw = f"答案文本。\n{FOLLOWUP_MARKER}\n追问是什么？\n"
        f = StreamFilter()
        visible = "".join(f.feed(c) for c in raw) + f.flush()
        assert FOLLOWUP_MARKER not in visible and "#" not in visible
        assert visible == "答案文本。\n"

    def test_raw_preserves_everything(self):
        """raw() 必须保留完整原始文本（含追问区），供解析用。"""
        from src.pipeline.followup import StreamFilter, FOLLOWUP_MARKER
        raw = f"答案。\n{FOLLOWUP_MARKER}\n问题是什么？\n"
        f = StreamFilter()
        for c in raw:
            f.feed(c)
        f.flush()
        assert f.raw() == raw

    def test_interrupted_mid_stream(self):
        """流被打断（未 flush）时仍能拿到已生成部分的正文。"""
        from src.pipeline.followup import StreamFilter
        f = StreamFilter()
        f.feed("上半年GDP同比增长")
        r = f.result("GDP")
        # 没有分隔符 → parse_followups 原样返回，正文完整、追问为空
        assert "上半年GDP" in r.body
        assert r.followups == []

    def test_empty_feed(self):
        from src.pipeline.followup import StreamFilter
        f = StreamFilter()
        assert f.feed("") == ""
        assert f.flush() == ""


# ════════════════════════════════════════════════════════════════════════
#                   ③：澄清提问（保守判定）
# ════════════════════════════════════════════════════════════════════════
class TestClarify:
    """澄清提问的**代价不对称**：误触发会打断正常问答（负体验），
    而漏判只是少了个功能。所以默认关闭 + 三重保守约束。
    """

    def test_disabled_by_default(self):
        """⚠️ 必须默认关闭 —— 精度未经线上验证前不该打断用户。"""
        from src.pipeline.followup import ENABLE_CLARIFY, should_clarify
        assert ENABLE_CLARIFY is False, "澄清提问不应默认开启"
        d = should_clarify("苹果多少钱", [], None)
        assert d.need is False
        assert "未启用" in d.reason

    def test_too_few_passages(self, emb):
        """证据太少时"分裂"没有统计意义 → 不反问。"""
        from src.pipeline.followup import should_clarify
        ps = [_P("a", "内容一"), _P("b", "内容二")]
        d = should_clarify("苹果", ps, emb, enable=True)
        assert d.need is False
        assert "统计意义" in d.reason

    def test_no_embedder(self):
        from src.pipeline.followup import should_clarify
        ps = [_P(f"t{i}", f"内容{i}") for i in range(5)]
        d = should_clarify("q", ps, None, enable=True)
        assert d.need is False

    def test_coherent_evidence_no_clarify(self, emb):
        """证据同源（都在讲一件事）时**绝不能**反问。"""
        from src.pipeline.followup import should_clarify
        ps = [
            _P("GDP增速", "上半年国内生产总值同比增长5.3%。"),
            _P("GDP数据", "统计局发布上半年GDP增长5.3%。"),
            _P("三次产业", "第一产业增长3.5%，第二产业5.8%。"),
            _P("经济运行", "上半年国民经济运行总体平稳，增速加快。"),
            _P("居民收入", "上半年居民人均可支配收入增长5.4%。"),
        ]
        d = should_clarify("上半年经济表现如何", ps, emb, enable=True)
        assert d.need is False, f"同源证据被误判为歧义: {d.reason}"

    def test_reason_always_filled(self, emb):
        """即使 need=False 也要填 reason（上线初期靠它观察边界样本）。"""
        from src.pipeline.followup import should_clarify
        d = should_clarify("q", [], None, enable=True)
        assert d.reason, "reason 为空会让线上无法调参"


# ════════════════════════════════════════════════════════════════════════
#         ④：agent 端到端（追问区绝不能污染 memory / 缓存）
# ════════════════════════════════════════════════════════════════════════
_RAW_ANSWER = None      # 在 fixture 里按 marker 组装


@pytest.fixture()
def followup_agent(monkeypatch, tmp_path):
    """mock 掉 LLM / 改写的 agent，避免联网与付费调用。"""
    import src.core.agent as agent_mod
    from src.core.agent import AgenticSearchAgent
    from src.pipeline.followup import FOLLOWUP_MARKER

    global _RAW_ANSWER
    _RAW_ANSWER = (
        "上半年GDP同比增长5.3%[1]，增速加快0.1个百分点[2]。\n"
        f"{FOLLOWUP_MARKER}\n"
        "三次产业各自的增速是多少？\n"
        "与去年同期相比表现如何？\n"
        "居民收入增长了多少？\n"
    )

    monkeypatch.setattr(agent_mod, "llm_chat",
                        lambda stage, messages, **kw: _RAW_ANSWER)

    def _stream(stage, messages, **kw):
        # 刻意用不规则切片，让分隔符跨 chunk
        s, out, i = _RAW_ANSWER, [], 0
        for size in (12, 9, 3, 2, 5, 4, 1, 7):
            if i >= len(s):
                break
            out.append(s[i:i + size])
            i += size
        if i < len(s):
            out.append(s[i:])
        return iter(out)

    monkeypatch.setattr(agent_mod, "llm_stream_chat", _stream)
    monkeypatch.setattr(agent_mod, "query_rewrite_route",
                        lambda q, **kw: "NO_SEARCH")

    return AgenticSearchAgent(
        enable_rag=False, enable_tools=False,
        qa_cache_dir=str(tmp_path / "qa"),
    )


_BODY = "上半年GDP同比增长5.3%[1]，增速加快0.1个百分点[2]。"


class TestFollowupAgent:
    def test_non_stream_strips_marker(self, followup_agent):
        res = followup_agent.chat("上半年GDP增长多少", verbose=False,
                                  return_result=True, session_id="s1")
        from src.pipeline.followup import FOLLOWUP_MARKER
        assert FOLLOWUP_MARKER not in res.text
        assert res.text == _BODY
        assert len(res.followups) == 3

    def test_memory_never_polluted(self, followup_agent):
        """⚠️ 追问区绝不能进 memory —— 会污染后续所有轮次的对话历史。"""
        from src.pipeline.followup import FOLLOWUP_MARKER
        followup_agent.chat("上半年GDP增长多少", verbose=False, session_id="s2")
        last = followup_agent._get_memory("s2").get_messages()[-1]["content"]
        assert FOLLOWUP_MARKER not in last
        assert "三次产业" not in last
        assert last == _BODY

    def test_l1_cache_never_polluted(self, followup_agent):
        """⚠️ 最严重：追问区若进 L1，下次命中会**直接吐给用户**并永久固化。"""
        from src.pipeline.followup import FOLLOWUP_MARKER
        followup_agent.chat("量子计算的基本原理是什么", verbose=False,
                            session_id="s3")
        cached = followup_agent.qa_cache.get("量子计算的基本原理是什么")
        if cached is not None:
            assert FOLLOWUP_MARKER not in cached
            assert "三次产业" not in cached

    def test_stream_hides_marker(self, followup_agent):
        """流式下分隔符（被切碎）不能漏给用户。"""
        from src.pipeline.followup import FOLLOWUP_MARKER
        stream = followup_agent.chat("GDP增长多少呢", verbose=False,
                                     is_stream=True, return_result=True,
                                     session_id="s4")
        visible = "".join(stream)
        assert FOLLOWUP_MARKER not in visible
        assert "#" not in visible
        assert "三次产业" not in visible
        assert visible.strip() == _BODY.strip()
        assert len(stream.result.followups) == 3

    def test_stream_memory_clean(self, followup_agent):
        from src.pipeline.followup import FOLLOWUP_MARKER
        stream = followup_agent.chat("再问GDP", verbose=False, is_stream=True,
                                     return_result=True, session_id="s5")
        list(stream)
        last = followup_agent._get_memory("s5").get_messages()[-1]["content"]
        assert FOLLOWUP_MARKER not in last and "三次产业" not in last

    def test_backward_compat_plain_str(self, followup_agent):
        """不传 return_result 时仍返回裸 str（且已剥离）。"""
        from src.pipeline.followup import FOLLOWUP_MARKER
        out = followup_agent.chat("问题", verbose=False, session_id="s6")
        assert isinstance(out, str)
        assert FOLLOWUP_MARKER not in out

    def test_to_dict_has_followups(self, followup_agent):
        res = followup_agent.chat("问题啊", verbose=False,
                                  return_result=True, session_id="s7")
        d = res.to_dict()
        assert "followups" in d and len(d["followups"]) == 3

    def test_trace_has_followup_event(self, followup_agent):
        """trace 里应有 followup 事件，供前端/监控消费。"""
        res = followup_agent.chat("问题呀", verbose=False,
                                  return_result=True, session_id="s8")
        stages = [e.get("stage") for e in res.trace]
        assert "followup" in stages


# ════════════════════════════════════════════════════════════════════════
#                   ⑤：前端渲染契约
# ════════════════════════════════════════════════════════════════════════
class TestFollowupFrontend:
    def test_prompt_contains_instruction(self):
        """summary system prompt 必须同时含追问指令**与**安全规则。

        后半句是防回归：组装逻辑改成 list+join 后，若不小心漏掉
        EVIDENCE_GUARD_PROMPT，injection 防护会静默失效。
        """
        from src.configs.prompts import build_summary_system
        from src.pipeline.followup import FOLLOWUP_MARKER
        p = build_summary_system(context="[环境信息] test")
        assert FOLLOWUP_MARKER in p
        assert "外部资料安全规则" in p, "安全规则被挤掉了（回归）"

    def test_cli_icon_and_render(self):
        import main as cli
        assert "followup" in cli.STAGE_ICON

    def test_cli_silent_when_empty(self):
        """无追问时 CLI 必须零输出（不能打"无追问推荐"之类的废话）。"""
        import io
        import contextlib
        import main as cli
        from src.core.answer_types import AnswerResult
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli._print_followups(AnswerResult(text="闲聊"))
            cli._print_followups(None)
        assert buf.getvalue() == ""

    def test_cli_renders_followups(self):
        import io
        import contextlib
        import main as cli
        from src.core.answer_types import AnswerResult
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli._print_followups(AnswerResult(text="a", followups=["问题一？"]))
        out = buf.getvalue()
        assert "你可能还想问" in out and "问题一？" in out

    def test_answer_result_render(self):
        from src.core.answer_types import AnswerResult
        r = AnswerResult(text="a", followups=["问题一？", "问题二？"])
        md = r.render_followups_markdown()
        assert "你可能还想问" in md and "问题一？" in md
        assert AnswerResult(text="a").render_followups_markdown() == ""

    def test_web_wiring(self):
        """Web 端必须有独立渲染函数，且 followup 事件不重复渲染成步骤块。"""
        src = open(_repo("main_web.py"), encoding="utf-8").read()
        assert "_render_followups_md" in src
        assert '("sources", "followup")' in src, (
            "followup 事件未被步骤块跳过，会与独立区块重复展示"
        )

    def test_scripts_contract(self):
        src = open(_repo("scripts/search.py"), encoding="utf-8").read()
        assert 'payload["followups"]' in src

# ════════════════════════════════════════════════════════════════════════
#              snippet 噪声清洗（rag/textclean.py）
# ════════════════════════════════════════════════════════════════════════
class TestSnippetClean:
    """L4 snippet 的页面模板噪声清洗。

    ⚠️ 本组所有样本**取自真实缓存**（Tavily 返回的旅游/电商站内容），
    不是我编的。凭想象写的噪声样本会让规则看起来很好但线上无效 ——
    这类"规则 + 阈值"的模块必须用真实数据标定。

    最重要的一条是 `test_never_empties_evidence`：清洗是**有损变换**，
    绝不能把证据洗空。宁可带着噪声，也不能丢掉这一路召回。
    """

    # ---- 真实噪声样本 ----
    NOISE = [
        "||  |  |  |",                          # 空表格骨架
        "|---",                                 # Markdown 表格分隔线
        "|--- |",
        "|| 曼谷﻿ | 清邁 | 網卡／交通 |",          # 表头
        "|| 涼季 約11－12月 | 氣候較涼爽 |",       # 表格数据行
        "4.5",                                  # 裸评分
        "TWD 210 起",                           # 价格标签
        "30K+ 個已訂購",                         # 销量
        "很好177 則評論",                        # 评分块
        "4.7/51358 reviews",
        "22648 已訂",
        "1 週前",                               # 时间戳
        "review picture",                       # 图片占位
        "白金",                                 # 会员等级
        "Thailand Chiang Mai Wat Phra That Doi Suthep AFotolia 107234124",
        "Source：Shutterstock",                  # 图片版权
        "© OpenStreetMap contributors",          # 页脚版权
        "12345678910月1112",                     # 月份选择器控件
        "▶︎泰國入境規定",                        # 导航链接
        "▶︎曼谷景點、曼谷自由行、泰國跳島合輯、普吉島自由行",  # 顿号式导航列表
        "链接  下载  比较",                      # UI 按钮栏
        "",
        "***",
    ]

    # ---- 真实正文样本（含 3 条陷阱）----
    CONTENT = [
        "泰國的11月到2月是氣候最為涼爽的時候（雖然四季都熱），但這個時間比較不會下雨。",
        "一月：雨季，气温在 27-31°C （81-88°F） 左右。",
        "十月：过渡到雨季，但仍然令人愉快;温度 26-31°C （79-88°F）。",
        "## 全球最佳旅遊時間：東南亞",              # Markdown 标题（结构信息）
        "### 七月",                               # 极短标题 —— 必须靠白名单救回
        "雨季時間：每年約 11 月至隔年 3 月",         # 冒号式要点（高价值事实）
        "| 云彩 十月 雅典出现快速增加的云量，整个月份天空多云的时间从26%增加至39%。",
        "門票 TWD 210，建議提前預訂以免額滿。",      # 陷阱：价格在句子里
        "下载安装包后请重启应用即可生效。",           # 陷阱：含 UI 词但是句子
        "▶ 這裡風景優美，景色迷人，值得一遊。",       # 陷阱：箭头开头但是完整句
        "冲浪：前往库塔海滩或乌鲁瓦图，感受世界上最好的海浪。",
    ]

    @pytest.mark.parametrize("line", NOISE)
    def test_noise_detected(self, line):
        from src.rag.textclean import _is_noise_line
        assert _is_noise_line(line), f"漏删噪声: {line!r}"

    @pytest.mark.parametrize("line", CONTENT)
    def test_content_preserved(self, line):
        from src.rag.textclean import _is_noise_line
        assert not _is_noise_line(line), f"误删正文: {line!r}"

    def test_markdown_heading_whitelist(self):
        """⚠️ Markdown 标题必须**最优先**放行。

        `### 七月` 只有 4 个字符、没有任何句子标记，会被"孤立短标签"
        规则判为噪声。但它是**结构信息**，告诉模型"接下来讲七月"，
        对多月份对比类问题价值很高。所以白名单必须在判定链最前面。
        """
        from src.rag.textclean import _is_noise_line, SHORT_LABEL_MAX
        assert len("### 七月") < SHORT_LABEL_MAX      # 确认它真的会命中短标签规则
        assert not _is_noise_line("### 七月")
        assert not _is_noise_line("# 概述")

    def test_colon_bullet_preserved(self):
        """⚠️ 冒号式要点是高价值结构化事实，不能被短标签规则删掉。

        `降雨高峰：通常集中在 12 月至 2 月` 这类是直接的答案素材。
        它们靠"冒号在句子标记里"被放行 —— 若哪天把冒号从
        `_RE_SENTENCE_MARK` 移除，这类要点会被大批误删。
        """
        from src.rag.textclean import _is_noise_line
        for s in ["降雨高峰：通常集中在 12 月至 2 月",
                  "天氣特色：午後陣雨較多",
                  "注意事項：建議攜帶輕便雨衣"]:
            assert not _is_noise_line(s), f"冒号要点被误删: {s!r}"

    def test_digit_run_needs_both_conditions(self):
        """连续数字串必须同时满足"有长数字段"和"整行只有数字"。

        只看"有 8 位连续数字"会误删
        `订单号 20260807123456 已生成，请查收。` 这类正文。
        """
        from src.rag.textclean import _is_noise_line
        assert _is_noise_line("12345678910月1112")
        assert not _is_noise_line("订单号 20260807123456 已生成，请查收。")

    def test_short_text_untouched(self):
        """短于 MIN_LEN_TO_CLEAN 的文本原样返回。

        DDG/Serper/Bing 的摘要实测 40~100 字（DDG 中位数 61），
        本身就是搜索引擎抽好的句子，清洗只有风险没有收益。
        """
        from src.rag.textclean import clean_snippet, MIN_LEN_TO_CLEAN
        s = "雅典, 希腊 - 秋季. 十月份的平均海水温度: 22.3°摄氏度."
        assert len(s) < MIN_LEN_TO_CLEAN
        assert clean_snippet(s) == s

    def test_never_empties_evidence(self):
        """⚠️ 本组最重要：清洗**绝不能**把证据洗空。

        清洗是有损变换。若规则过激（或遇到极端页面结构），
        必须放弃清洗、返回原文 —— 带噪声的证据仍然可用，
        被洗空的证据等于丢了这一路召回。
        """
        from src.rag.textclean import clean_snippet
        for name, s in [
            ("纯表格", "|| a | b |\n" * 80),
            ("纯分隔线", "|---\n" * 150),
            ("纯评分", "4.5\n" * 200),
            ("纯符号", "***\n···\n▶︎\n" * 100),
        ]:
            out = clean_snippet(s)
            assert out, f"{name} 被洗空了"
            assert out == s, f"{name} 应触发兜底原样返回"

    def test_fallback_ratio(self):
        """保留比例低于 MIN_KEEP_RATIO 时触发兜底。

        ⚠️ 样本构造要点：本模块是**逐行**处理的，所以样本必须有换行。
        我第一版写成 `"正文。" * 6`（一整行），结果既没有行可删也测不出
        任何效果 —— 这类"行级处理"的模块，测试样本的换行结构是本质的。
        """
        from src.rag.textclean import clean_snippet, MIN_KEEP_RATIO
        # 90% 噪声 + 10% 正文 → 应触发兜底
        dirty = "|---\n" * 100 + "这是唯一的一句正文内容，讲了一点点东西。"
        out = clean_snippet(dirty)
        assert out == dirty, "过激清洗未触发兜底"

        # 对照：噪声占比适中 → 正常清洗（每句独立成行才有行可删）
        body = "\n".join(
            f"第{i}段正文，讲述了当地气候的具体特征与适合出行的月份安排。"
            for i in range(8)
        )
        ok = "|---\n" * 5 + body
        out2 = clean_snippet(ok)
        assert out2 != ok, "正常场景反而触发了兜底"
        assert "|---" not in out2, "表格分隔线未被清掉"
        assert len(out2) >= len(ok) * MIN_KEEP_RATIO

    def test_line_dedup(self):
        """Tavily 拼接片段时会重复输出同一段，需行级去重。

        实测样本里 `## 解读巴厘岛的季节：旱季与雨季` 连续出现两次。
        """
        from src.rag.textclean import clean_snippet
        body = "巴厘岛的旱季从 4 月到 9 月是阳光爱好者的天堂，非常适合海滩郊游。"
        # ⚠️ 其余内容也必须**逐行**，否则总长不够、或没有换行可切分。
        others = "\n".join(
            f"第{i}段：雨季从 10 月持续到 3 月，带来郁郁葱葱的绿色植物。"
            for i in range(5)
        )
        dup = f"{body}\n{body}\n{body}\n{others}"
        out = clean_snippet(dup)
        assert out.count(body) == 1, f"重复行未去重: 出现 {out.count(body)} 次"
        assert "第0段" in out and "第4段" in out, "不同内容被误删"

    def test_tavily_fragment_marker_split(self):
        """`[...]` 必须被当作片段边界拆开。

        实测 17/20 条 Tavily 结果含这个标记。若不拆，两个片段会被当成
        同一行 —— 一行里混着噪声和正文就没法分开处理。
        """
        from src.rag.textclean import clean_snippet
        s = ("這是第一段正文，講述乾季的天氣狀況與適合的活動安排。" * 3 +
             "[...]4.5[...]" +
             "這是第二段正文，講述雨季的降雨量與注意事項。" * 3)
        out = clean_snippet(s)
        assert "4.5" not in out.split("\n"), "片段间的噪声未被清掉"
        assert "第一段正文" in out and "第二段正文" in out

    def test_switch_off_is_identity(self):
        """开关关闭时逐字原样返回（零回归保证）。"""
        from src.rag.textclean import clean_snippet
        s = "||  |  |\n4.5\nTWD 0 起\n" + "真实正文内容，包含足够多的中文字符。" * 10
        assert clean_snippet(s, enable=False) == s
        assert clean_snippet(s, enable=True) != s

    def test_clean_stats_shape(self):
        """clean_stats 的字段契约（上线后要打日志观测）。"""
        from src.rag.textclean import clean_stats
        s = "|---\n" * 5 + "这是一段足够长的正文内容，讲了很多有价值的事实。" * 6
        st = clean_stats(s)
        for k in ("raw_len", "clean_len", "removed_chars", "removed_ratio",
                  "raw_lines", "kept_lines", "fell_back"):
            assert k in st, f"缺少字段 {k}"
        assert st["removed_chars"] == st["raw_len"] - st["clean_len"]
        assert 0.0 <= st["removed_ratio"] <= 1.0

    def test_l4_layer_wired(self):
        """L4 层必须调用清洗（接线正确性）。"""
        src = open(_repo("src/rag/layers.py"), encoding="utf-8").read()
        assert "from .textclean import clean_snippet" in src
        assert "clean_snippet(r.get(\"snippet\", \"\"))" in src

    def test_l4_search_applies_clean(self, monkeypatch):
        """端到端：L4.search 产出的 Passage.text 必须已清洗。"""
        import src.rag.layers as L
        dirty = ("||  |  |\n4.5\nTWD 210 起\n© OpenStreetMap contributors\n"
                 + "上半年国内生产总值同比增长5.3%，增速比一季度加快0.1个百分点。" * 5)
        monkeypatch.setattr(L, "web_search", lambda q, top_k=5, **kw: [
            {"title": "GDP数据", "url": "https://a.com/1", "snippet": dirty},
        ])
        ps = L.L4WebLayer().search("GDP", top_k=1)
        assert len(ps) == 1
        assert "TWD 210 起" not in ps[0].text
        assert "OpenStreetMap" not in ps[0].text
        assert "同比增长5.3%" in ps[0].text, "正文被洗掉了"

    def test_real_cache_no_fallback_storm(self):
        """真实缓存全量跑一遍：兜底触发率必须很低。

        兜底率高说明规则过激 —— 那样清洗就等于没做（全部原样返回），
        白付了 CPU 还给人"已经优化了"的错觉。
        """
        from src.configs import config
        from src.rag.textclean import clean_stats
        try:
            import diskcache
            c = diskcache.Cache(config.SEARCH_CACHE_DIR)
        except Exception:
            pytest.skip("检索缓存不可用")
        n, n_fb = 0, 0
        for k in list(c):
            v = c.get(k)
            if not isinstance(v, list):
                continue
            for r in v:
                sn = r.get("snippet") or ""
                if len(sn) < 200:
                    continue
                n += 1
                if clean_stats(sn)["fell_back"]:
                    n_fb += 1
        if n == 0:
            pytest.skip("缓存里没有长 snippet")
        assert n_fb / n < 0.2, f"兜底率过高 {n_fb}/{n}，规则可能过激"


# ══════════════════════════════════════════════════════════════════════════
# 数据路径锚定
# ══════════════════════════════════════════════════════════════════════════
class TestDataPathAnchoring:
    """三处独立的"从 __file__ 反推仓库根"必须都指向真正的仓库根。

    为什么需要这组测试：源码搬进 `src/` 后，仓库根比原来深了一级。三个
    模块各自独立地用 `__file__` 反推根目录来定位 `data/`：

        src/configs/config.py        → DATA_DIR / QA_CACHE_DIR
        src/rag/config.py            → RAG_DATA_DIR
        src/rag/wiki_rag/config.py   → YAML 里 16 个 paths 的锚点

    漏改任何一处，路径就会变成 `<ROOT>/src/data/...`。**最危险的是这种
    失败不会报错**：retriever 把"文件打不开"当成"离线索引尚未构建"的正常
    降级处理 —— 只打一行日志、该层返回空结果，其余层照常融合。系统看起来
    一切正常，实际上 L2/L5 已经永久失效。

    实际发生过：重构时改了前两处、漏了 wiki_rag 那处，全量测试 349 项
    照样全绿，只有真实跑一次问答、盯着 stderr 才看见
    `KG db not found: .../src/data/rag_data/wikidata_zh_kg.db`。
    """

    @staticmethod
    def _repo_root() -> str:
        return _REPO_ROOT

    def test_configs_data_dir_not_under_src(self):
        from src.configs import config
        assert config.PROJECT_ROOT == self._repo_root(), (
            f"configs PROJECT_ROOT 错位: {config.PROJECT_ROOT}")
        assert "/src/data" not in config.DATA_DIR, config.DATA_DIR
        assert "/src/data" not in config.QA_CACHE_DIR, config.QA_CACHE_DIR

    def test_rag_config_data_dir_not_under_src(self):
        from src.rag import config as rc
        assert str(rc.PROJECT_ROOT) == self._repo_root(), (
            f"rag PROJECT_ROOT 错位: {rc.PROJECT_ROOT}")
        assert "/src/data" not in str(rc.RAG_DATA_DIR), rc.RAG_DATA_DIR

    def test_wiki_rag_yaml_paths_not_under_src(self):
        """这条就是当初漏掉的那处。"""
        from src.rag.wiki_rag.config import PROJECT_ROOT, load_config
        assert str(PROJECT_ROOT) == self._repo_root(), (
            f"wiki_rag PROJECT_ROOT 错位: {PROJECT_ROOT}")
        paths = load_config()["paths"]
        assert paths, "YAML paths 为空，配置没读到"
        bad = {k: str(v) for k, v in paths.items() if "/src/data" in str(v)}
        assert not bad, f"这些路径错误地落在 src/ 下: {bad}"

    def test_all_yaml_paths_anchored_to_repo_data(self):
        """所有相对路径都应落在 <repo>/data/ 下（绝对路径注入的除外）。"""
        from src.rag.wiki_rag.config import load_config
        expect = _os.path.join(self._repo_root(), "data")
        for k, v in load_config()["paths"].items():
            assert str(v).startswith(expect), f"{k} 未锚定到 data/: {v}"


# ══════════════════════════════════════════════════════════════════════════
# warmup 契约
# ══════════════════════════════════════════════════════════════════════════
class TestWarmupContract:
    """`warmup_all` 必须对**第一个** stage 做回扫补偿。

    背景（实测）：router 与 rewriter 共用同一模型但 temperature 不同，
    ollama 只能让其中一个处于热态 —— 逐个预热时后者会顶掉前者。
    router 是每条 query 的第一个环节，若它被顶掉，约 1.6s 毛刺就
    100% 落在用户感知最强的位置。

    这里用「打桩记录调用序列」而不是测真实耗时：耗时断言在 CI 上
    必然不稳定，而"是否回扫"是确定性的结构行为，才是该锁住的契约。
    """

    def _record(self, monkeypatch, stages):
        from src.core import llm_client
        calls: list[str] = []
        monkeypatch.setattr(llm_client, "local_stages", lambda: list(stages))
        monkeypatch.setattr(
            llm_client, "get_stage_config",
            lambda s: {"provider": "ollama", "model": "m", "temperature": 0.0},
        )
        monkeypatch.setattr(
            llm_client, "chat",
            lambda stage, messages, **kw: calls.append(stage) or "ok",
        )
        llm_client.warmup_all(verbose=False)
        return calls

    def test_multi_stage_rebounds_to_first(self, monkeypatch):
        calls = self._record(monkeypatch, ["router", "rewriter"])
        assert calls == ["router", "rewriter", "router"], (
            f"应在末尾回扫第一个 stage，实际调用序列: {calls}")

    def test_single_stage_no_redundant_rebound(self, monkeypatch):
        """只有一个 stage 时不存在互顶，不应浪费一次推理。"""
        calls = self._record(monkeypatch, ["router"])
        assert calls == ["router"], f"单 stage 不该回扫，实际: {calls}"

    def test_rebound_targets_first_local_stage(self, monkeypatch):
        """回扫对象必须是**首个**（用户最先感知的）stage，而非最后一个。"""
        calls = self._record(monkeypatch, ["router", "rewriter", "summary"])
        assert calls[-1] == "router", f"回扫对象应为 router，实际: {calls[-1]}"
        assert calls[:3] == ["router", "rewriter", "summary"]

    def test_warmup_never_raises(self, monkeypatch):
        """预热失败必须被吞掉：它不该阻止服务启动。"""
        from src.core import llm_client
        monkeypatch.setattr(llm_client, "local_stages", lambda: ["router"])
        monkeypatch.setattr(
            llm_client, "get_stage_config",
            lambda s: {"provider": "ollama", "model": "m", "temperature": 0.0},
        )

        def _boom(*a, **k):
            raise RuntimeError("ollama down")

        monkeypatch.setattr(llm_client, "chat", _boom)
        out = llm_client.warmup_all(verbose=False)   # 不应抛异常
        assert out.get("router") is None

    def test_remote_stage_skipped(self, monkeypatch):
        """远端 provider 不该被预热（白花 token 与配额）。"""
        from src.core import llm_client
        calls: list[str] = []
        monkeypatch.setattr(llm_client, "local_stages", lambda: ["summary"])
        monkeypatch.setattr(
            llm_client, "get_stage_config",
            lambda s: {"provider": "openai", "model": "deepseek-chat"},
        )
        monkeypatch.setattr(
            llm_client, "chat",
            lambda stage, messages, **kw: calls.append(stage) or "ok",
        )
        assert llm_client.warmup_stage("summary", verbose=False) is None
        assert calls == [], f"远端 stage 不应发起请求，实际: {calls}"


# ══════════════════════════════════════════════════════════════════════════
# 显式联网指令（web_intent）
# ══════════════════════════════════════════════════════════════════════════
class TestWebIntent:
    """L4 自动触发是启发式的，必须给用户一个确定性的逃生舱。

    实测盲区 case：「法国最低饮酒年龄」离线聚合置信度 0.98、实词覆盖率
    也达标（"法国/最低/饮酒/年龄" 在维基「合法飲酒年齡」条目里全都出现），
    于是 low_conf 与 insufficient 双双为 False → L4 不触发 —— 但那篇条目
    只讲欧洲通例、没点名法国 → 只能拒答。

    这类"主题相关、词面全覆盖，却恰好缺了用户要的那个具体事实"的情况，
    任何基于相似度或词面覆盖的指标都识别不了。所以本模块的价值在于
    **确定性**：用户说了就一定生效，不受采样波动影响。
    """

    # 应触发（用户在下指令）
    POSITIVE = [
        '请你出触发 网页搜索,进行"法国最低饮酒年龄"的资料搜索',
        "请联网搜索一下法国饮酒年龄",
        "帮我网上查一下特斯拉股价",
        "谷歌搜一下诺贝尔奖",
        "重新搜一下最新资料",
        "用 web 搜索确认一下",
        "上网查查这个型号",
    ]

    # 不应触发（正常提问，误判代价是白等 2~16s 联网）
    NEGATIVE = [
        "法国最低饮酒年龄",
        "网上购物的发展史",       # 含渠道词但无指令意图
        "如何联网打印机",         # 疑问句 + 渠道词
        "搜索引擎的排序原理",     # 含动作词但在讨论概念
        "查尔斯·达尔文的生平",   # "查"只是姓名的一部分
        "什么是网络搜索",         # 含强指令词，但疑问句必须优先排除
        "为什么谷歌搜索这么快",
        "",
    ]

    def test_positive_cases(self):
        from src.pipeline.web_intent import wants_web_search
        for q in self.POSITIVE:
            assert wants_web_search(q) is True, f"应识别为联网指令: {q!r}"

    def test_negative_cases(self):
        from src.pipeline.web_intent import wants_web_search
        for q in self.NEGATIVE:
            assert wants_web_search(q) is False, f"不应误判为联网指令: {q!r}"

    def test_interrogative_beats_strong_directive(self):
        """疑问句排除必须优先于强指令匹配。

        「什么是网络搜索」含强指令词"网络搜索"，若先判强指令就会误触发。
        这个顺序是实测踩出来的，用一条独立测试锁死，防止后人调换顺序。
        """
        from src.pipeline.web_intent import wants_web_search
        assert wants_web_search("什么是网络搜索") is False
        assert wants_web_search("请用网络搜索查一下") is True

    def test_strip_extracts_quoted_target(self):
        """带引号时应精确提取引号内容——那是用户明示的检索对象。"""
        from src.pipeline.web_intent import strip_web_directive
        got = strip_web_directive('请你出触发 网页搜索,进行"法国最低饮酒年龄"的资料搜索')
        assert got == "法国最低饮酒年龄", got

    def test_strip_removes_directive_words(self):
        from src.pipeline.web_intent import strip_web_directive
        got = strip_web_directive("请联网搜索一下法国饮酒年龄")
        assert "联网" not in got and "搜索" not in got
        assert "法国" in got and "饮酒年龄" in got

    def test_strip_keeps_original_when_all_directive(self):
        """整句都是指令时必须返回原句，不能变成空串让检索崩掉。"""
        from src.pipeline.web_intent import strip_web_directive
        for q in ("联网搜一下", "帮我搜搜"):
            assert len(strip_web_directive(q)) >= 3, q

    def test_retriever_force_web_bypasses_heuristics(self, monkeypatch):
        """force_web=True 时，即使置信度很高也必须激活 L4。"""
        from src.rag.retriever import LayeredRetriever
        from src.rag.types import Passage

        r = LayeredRetriever.__new__(LayeredRetriever)   # 不跑 __init__（避免加载模型）
        called = {"l4": False}

        class _L4:
            name = "L4_web"
            def search(self, q, top_k=5):
                called["l4"] = True
                return [Passage(text="web 证据", layer="L4_web", score=0.9, title="w")]

        # 离线层给一条高分证据，让 low_conf / insufficient 都为 False
        hi = Passage(text="法国 最低 饮酒 年龄 都在这段里", layer="L2_wiki",
                     score=0.99, title="t")
        r.l1 = None
        r.l2 = r.l3 = r.l5 = None
        r.l4 = _L4()
        r.strategy = "offline_only"
        r.fusion_top_k = 6
        r.embedder = None
        r._incr = None
        from concurrent.futures import ThreadPoolExecutor
        r._pool = ThreadPoolExecutor(max_workers=2)
        monkeypatch.setattr(
            r, "_parallel_search", lambda q, a, namespace=None: {"L2_wiki": [hi]},
        )
        try:
            res = r.retrieve("法国 最低 饮酒 年龄", force_web=True)
            assert called["l4"] is True, "force_web=True 必须触发 L4"
            assert res.web_fallback is True
        finally:
            r._pool.shutdown(wait=False)

    def test_retriever_no_force_respects_heuristics(self, monkeypatch):
        """对照组：force_web=False 且证据充分时不应联网（避免白等）。"""
        from src.rag.retriever import LayeredRetriever
        from src.rag.types import Passage

        r = LayeredRetriever.__new__(LayeredRetriever)
        called = {"l4": False}

        class _L4:
            name = "L4_web"
            def search(self, q, top_k=5):
                called["l4"] = True
                return []

        hi = Passage(text="法国 最低 饮酒 年龄 都在这段里", layer="L2_wiki",
                     score=0.99, title="t")
        r.l1 = None
        r.l2 = r.l3 = r.l5 = None
        r.l4 = _L4()
        r.strategy = "offline_only"
        r.fusion_top_k = 6
        r.embedder = None
        r._incr = None
        from concurrent.futures import ThreadPoolExecutor
        r._pool = ThreadPoolExecutor(max_workers=2)
        monkeypatch.setattr(
            r, "_parallel_search", lambda q, a, namespace=None: {"L2_wiki": [hi]},
        )
        try:
            r.retrieve("法国 最低 饮酒 年龄", force_web=False)
            assert called["l4"] is False, "证据充分时不该联网"
        finally:
            r._pool.shutdown(wait=False)

    def test_search_note_injected_when_forced(self):
        """web_forced=True 必须告知模型"联网已完成"。

        实测：不加这句时，模型看到用户说"请触发网页搜索"，会先声明
        「我无法主动触发网页搜索」再作答 —— 它把指令理解成"要求我自己
        去联网"，而它确实看不到检索是怎么发生的。这不是模型的错，是
        缺了"动作已完成"这个事实的回传。
        """
        from src.pipeline.evidence import build_user_message
        msg = build_user_message("法国饮酒年龄", "<evidence></evidence>",
                                 web_forced=True)
        assert "<search_note>" in msg
        assert "已经完成" in msg
        # 必须出现在 <question> 之前（紧邻提问，避免被长证据稀释）
        assert msg.index("<search_note>") < msg.index("<question>")

    def test_search_note_absent_by_default(self):
        """未强制联网时不应出现这段说明（避免误导模型）。"""
        from src.pipeline.evidence import build_user_message
        msg = build_user_message("法国饮酒年龄", "<evidence></evidence>")
        assert "<search_note>" not in msg

    def test_head_partial_refusal_rejected(self):
        """开门就说"未能直接确认"的拒答必须拦住（真实 case）。

        原来只扫尾部 200 字，这条答案结尾是正常的"建议以官方法律为准"，
        于是被判 tier=stable 写进 L1 冻结 30 天 —— 用户再问同一问题时
        毫秒命中这条拒答，**永远不会再检索**。
        """
        from src.cache.cache_policy import decide_cacheability
        ans = ("根据现有检索资料，**未能直接确认**法国的法定最低饮酒年龄："
               "资料提到大多数欧洲国家把 18 岁定为最低年龄，但并未直接点名法国。"
               "因此很可能是 18 岁，但建议以法国现行法律为准。")
        d = decide_cacheability("法国最低饮酒年龄", ans)
        assert d.cacheable is False
        assert d.tier == "reject_partial_refusal"

    def test_head_scan_does_not_kill_good_answer(self):
        """对照组：头部扫描不能误杀"交代某条证据局限但结论完整"的好答案。

        误杀的代价是丢掉"越用越强"的积累，所以这条必须放行。
        判据的分界线：整个问题没答出来（拒答）vs 某条证据不含某细节（严谨）。
        """
        from src.cache.cache_policy import decide_cacheability
        ans = ("Transformer 的核心是自注意力机制。虽然原论文未直接说明具体的"
               "初始化细节，但完整结构包含多头注意力、前馈网络、残差连接与"
               "层归一化四部分，编码器与解码器各堆叠 6 层。")
        d = decide_cacheability("Transformer 结构", ans)
        assert d.cacheable is True

    def test_retriever_force_web_skips_inner_l1(self, monkeypatch):
        """force_web 必须同时跳过 retriever **内部**的 L1。

        ⚠️ L1 在本系统里有两个入口：agent 的前置缓存 + retriever 的第一层。
        只堵 agent 那处时，实测 retriever 仍会毫秒级把上一轮的拒答当命中
        证据返回（日志显示 `[L1_qa:1], 融合 1 段`），L4 根本没机会跑。
        """
        import src.rag.retriever as R
        r = R.LayeredRetriever.__new__(R.LayeredRetriever)
        hit = {"used": False}

        class _L1:
            def get(self, *a, **k):
                hit["used"] = True
                return {"answer": "上一轮的拒答"}

        r.l1 = _L1()
        # force_web=True → 绝不能碰 l1
        assert R.LayeredRetriever.retrieve.__doc__ is not None
        import inspect
        src = inspect.getsource(R.LayeredRetriever.retrieve)
        assert "not force_web" in src, "retriever 内部 L1 必须受 force_web 约束"

    # ================================================================== #
    # L1 读写两侧的一致性（本轮修复：判据升级不回溯 + force_web 只旁路读）
    # ================================================================== #
    @staticmethod
    def _bare_agent(cache):
        """造一个不带 RAG / 工具的最小 agent，用于隔离测 L1 读写。"""
        from src.core.agent import AgenticSearchAgent
        return AgenticSearchAgent(
            qa_cache=cache, enable_rag=False, enable_tools=False,
        )

    def test_injected_empty_cache_is_not_discarded(self):
        """注入的**空** QACache 不能被静默丢弃。

        ⚠️ 原实现是 `qa_cache or QACache(...)`。QACache 定义了 `__len__`，
        于是空缓存的布尔值是 False，`or` 会把它丢掉、转而新建一个**读磁盘**
        的实例。后果极隐蔽：单测以为拿到了干净的内存缓存，实际 agent 用的
        是 data/qa_cache 里的存量条目（实测 72 条），测试会莫名命中生产
        数据且不报任何错。本条锁定 `is not None` 的语义。
        """
        from src.cache.qa_cache import QACache
        c = QACache(backend="memory", enable_fuzzy=False)
        assert len(c) == 0            # 前提：它是空的（布尔值为 False）
        assert self._bare_agent(c).qa_cache is c

    def test_l1_read_rechecks_admission(self):
        """L1 命中必须再过一次准入判据（读取侧复核）。

        这是用户报的 bug 的直接修复点：判据收紧只作用于**写入**，
        存量条目照旧毫秒命中。实测 72 条里有 7 条属于这种情况，
        其中就包括「美国可饮酒的年龄是多少,法国和日本呢」——
        用户刚联网查到"法国 18 岁"，下一轮仍拿到旧的推断式拒答，
        因为 L1 短路发生在任何检索之前，新入库的 L3 知识没机会参与。
        """
        from src.cache.qa_cache import QACache
        c = QACache(backend="memory", enable_fuzzy=False)
        agent = self._bare_agent(c)
        q = "美国可饮酒的年龄是多少,法国和日本呢"
        stale = (
            "根据现有检索资料，未能直接确认法国的法定最低饮酒年龄："
            "美国为 21 岁，日本为 20 岁。关于法国，资料提到欧洲多国普遍为"
            " 18 岁，但并未直接点名法国，这只是基于欧洲通行标准的推断。"
        )
        c.add(q, stale)
        assert c.get(q) is not None            # 条目确实在 L1 里
        assert agent._l1_get(q, None) is None  # 但读取侧把它当 miss

    def test_l1_read_recheck_keeps_good_answer(self):
        """对照组：读取侧复核不能误杀正常答案（否则缓存全废）。"""
        from src.cache.qa_cache import QACache
        c = QACache(backend="memory", enable_fuzzy=False)
        agent = self._bare_agent(c)
        q = "美国可饮酒的年龄是多少,法国和日本呢"
        good = ("美国的最低法定饮酒年龄为 21 岁；法国为 18 岁；日本为 20 岁。"
                "三国均对未成年人购买酒类有明确限制。")
        c.add(q, good)
        assert agent._l1_get(q, None) == good

    def test_l1_read_recheck_off_when_policy_disabled(self):
        """`enable_cache_policy=False`（A/B 对照组）必须退回旧行为。"""
        from src.core.agent import AgenticSearchAgent
        from src.cache.qa_cache import QACache
        c = QACache(backend="memory", enable_fuzzy=False)
        agent = AgenticSearchAgent(
            qa_cache=c, enable_rag=False, enable_tools=False,
            enable_cache_policy=False,
        )
        q = "你好"
        c.add(q, "hi")            # 过短，开门禁时会被拒
        assert agent._l1_get(q, None) == "hi"

    def test_force_web_answer_not_written_to_l1(self):
        """显式联网指令的答案禁止写 L1，但仍要写 L3。

        ⚠️ force_web 原先只旁路了 L1 的**读**、没禁止**写**。实测在
        data/qa_cache 里捞到两条 key 本身就是联网指令的条目：
            '请你触发网页搜索进行法国最低饮酒年龄的资料搜索'
        这类 key 的 `wants_web_search()` 恒为 True —— 每次都判为
        "用户要求重新联网"，但它自己的答案却躺在缓存里。
        另一层理由：联网结果天然带时效性，按 stable(30d) 冻结等于把
        "实时"退化成"上个月的实时"。
        """
        from src.cache.qa_cache import QACache
        c = QACache(backend="memory", enable_fuzzy=False)
        agent = self._bare_agent(c)

        archived: list[str] = []

        class _FakeRetriever:
            def archive(self, query, answer, sources=None, namespace=None):
                archived.append(query)

        agent.retriever = _FakeRetriever()
        directive = "请你触发网页搜索，进行法国最低饮酒年龄的资料搜索"
        ans = "根据本次联网检索到的实时资料，法国最低饮酒年龄为 18 岁。"
        agent._archive_if_enabled(directive, ans, None, namespace=None,
                                  skip_l1=True)
        assert c.get(directive) is None       # L1 不写
        assert archived == [directive]        # L3 照写

    def test_normal_answer_still_written_to_l1(self):
        """对照组：skip_l1=False 时 L1 写入路径不受影响。"""
        from src.cache.qa_cache import QACache
        c = QACache(backend="memory", enable_fuzzy=False)
        agent = self._bare_agent(c)

        class _FakeRetriever:
            def archive(self, *a, **k):
                pass

        agent.retriever = _FakeRetriever()
        q = "美国可饮酒的年龄是多少,法国和日本呢"
        good = ("美国的最低法定饮酒年龄为 21 岁；法国为 18 岁；日本为 20 岁。"
                "三国均对未成年人购买酒类有明确限制。")
        agent._archive_if_enabled(q, good, None, namespace=None, skip_l1=False)
        assert c.get(q) == good

    def test_chat_passes_force_web_to_archive(self):
        """chat() 必须把 force_web 透传给归档（防止将来重构时漏掉）。

        这是"同一机制有多个入口"类疏漏的防线：force_web 现在要作用于
        三个地方 —— agent 读侧、retriever 读侧、归档写侧。少一处即失效。
        """
        import inspect
        from src.core.agent import AgenticSearchAgent
        src = inspect.getsource(AgenticSearchAgent.chat)
        assert "skip_l1=force_web" in src

    def test_retriever_l1_layer_also_rechecks_admission(self):
        """retriever 内部的 L1 层**也**必须做读取侧复核。

        ⚠️ 只在 `agent._l1_get()` 加复核时，实测日志出现自相矛盾的一幕：
            [agent] L1 命中被读取侧复核拒绝 (tier=reject_partial_refusal)
            …
                  RAG 检索完成 [L1_qa:1], 融合 1 段, conf=1.000
        agent 前门拦住了，retriever 后门又把同一条拒答放进来，还打上
        conf=1.0 的最高可信标签 —— 比不拦更糟（模型会更信它）。
        「同一机制有多个入口」是本仓库反复踩到的疏漏模式。
        """
        from src.rag.layers import L1QACacheLayer
        from src.cache.qa_cache import QACache
        c = QACache(backend="memory", enable_fuzzy=False)
        q = "法国最低饮酒年龄"
        stale = ("根据现有检索资料，未能直接确认法国的法定最低饮酒年龄："
                 "资料提到欧洲多国普遍为 18 岁，但并未直接点名法国。")
        c.add(q, stale)
        assert L1QACacheLayer(c).lookup(q) is None
        # A/B 对照组仍可退回旧行为
        assert L1QACacheLayer(
            c, enable_admission_recheck=False,
        ).lookup(q) == stale

    def test_l1_layer_recheck_keeps_good_answer(self):
        """对照组：retriever 侧复核同样不能误杀正常答案。"""
        from src.rag.layers import L1QACacheLayer
        from src.cache.qa_cache import QACache
        c = QACache(backend="memory", enable_fuzzy=False)
        q = "法国最低饮酒年龄"
        good = ("法国的最低法定饮酒年龄为 18 岁。2009 年法国将该年龄从 16 岁"
                "提高到 18 岁，禁止向未成年人销售任何含酒精饮料。")
        c.add(q, good)
        assert L1QACacheLayer(c).lookup(q) == good

    def test_hit_admissible_is_single_source_of_truth(self):
        """所有 L1 读取入口必须共用 `cache_policy.hit_admissible`。

        这条测试的目的不是验证行为，而是**锁定架构约束**：判据一旦被复制
        成多份，将来只改一处就会重现"前门拦住、后门放进来"的 bug。
        """
        import inspect
        from src.core.agent import AgenticSearchAgent
        from src.rag.layers import L1QACacheLayer
        for fn in (AgenticSearchAgent._l1_admissible, L1QACacheLayer.lookup):
            assert "hit_admissible" in inspect.getsource(fn), (
                f"{fn.__qualname__} 必须调用共享判据，不得自行实现"
            )

    def test_l3_search_also_rechecks_admission(self, tmp_path):
        """L3 召回**也**必须剔除拒答类历史条目。

        ⚠️ 这是「判据升级不回溯」在 L3 的重演，而且比 L1 更隐蔽：
        L1 命中直接返回给用户，错了一眼可见；L3 是作为**证据**进 prompt，
        一条旧拒答会安静地把模型带偏。

        实测（L1 已修好之后）用户第 4 轮的 6 段证据里有 3 段是 L3 旧拒答，
        而第 1/2 轮刚生成的正确答案根本没被召回 —— 于是模型照抄旧措辞，
        用户看到的仍是"基于欧洲通行标准的推断"，L1 的修复看起来毫无效果。
        """
        from src.rag.layers import L3HistoryLayer
        from src.rag.embedder import Embedder
        layer = L3HistoryLayer(Embedder(), index_dir=str(tmp_path / "l3"))
        q = "法国最低饮酒年龄"
        layer.add(q, "根据现有检索资料，未能直接确认法国的法定最低饮酒年龄："
                     "资料提到欧洲多国普遍为 18 岁，但并未直接点名法国。")
        layer.add("法国饮酒年龄规定",
                  "法国最低饮酒年龄为 18 岁。2009 年从 16 岁提高到 18 岁，"
                  "禁止向未成年人销售任何含酒精饮料。")
        texts = [p.text for p in layer.search(q, top_k=5)]
        assert not any("未能直接确认" in t for t in texts), "拒答仍被当证据召回"
        assert any("18 岁" in t and "未能直接确认" not in t for t in texts), \
            "正确答案必须仍能召回（不能把 L3 整层废掉）"

    def test_l3_recheck_can_be_disabled(self, tmp_path):
        """A/B 对照组：L3 复核可关闭，用于耗时基线测量。"""
        from src.rag.layers import L3HistoryLayer
        from src.rag.embedder import Embedder
        layer = L3HistoryLayer(Embedder(), index_dir=str(tmp_path / "l3b"),
                               enable_admission_recheck=False)
        q = "法国最低饮酒年龄"
        layer.add(q, "根据现有检索资料，未能直接确认法国的法定最低饮酒年龄："
                     "资料提到欧洲多国普遍为 18 岁，但并未直接点名法国。")
        assert any("未能直接确认" in p.text for p in layer.search(q, top_k=5))

    def test_l3_overfetches_when_recheck_enabled(self):
        """开启复核时必须超取，否则剔除脏条目会变成"减少证据量"。

        不超取的话，top_k 条里剔掉几条就真的少几条 —— 可能反过来触发
        low_evidence 判据，把"清理脏数据"变成"证据不足"。
        """
        import inspect
        from src.rag.layers import L3HistoryLayer
        src = inspect.getsource(L3HistoryLayer.search)
        assert "enable_admission_recheck" in src.split("fetch_k")[1][:200], \
            "fetch_k 的计算必须把复核开关考虑进去"

    def test_web_directive_key_is_never_cacheable(self):
        """key 本身是联网指令的条目属于「永久死条目」，必须能被检出。

        `chat()` 对这类 query 恒有 force_web=True → L1 的读被永久旁路，
        这条记录再也不可能被命中，纯粹占磁盘 + 污染 fuzzy 候选集。
        语义上也本就不该缓存：「请帮我重新联网搜一次 X」的正确语义是
        "每次都要重新搜"，给它冻结 30 天的答案与用户本意完全相反。
        """
        from src.pipeline.web_intent import wants_web_search
        for q in (
            "请你出触发网页搜索进行法国最低饮酒年龄的资料搜索",
            "请你触发 网页搜索,进行“法国最低饮酒年龄”的资料搜索",
        ):
            assert wants_web_search(q), f"应识别为联网指令: {q}"


class TestParallelEntityExtraction:
    """并列实体识别的**精度**护栏。

    这组测试锁定的是一个实测故障：判据原本是「句子里有连接词 且
    抽到 ≥2 个专名」，在 BrowseComp-ZH 的混淆式长题干上误报率
    **21.8%（63/289）**。误报不是无害的 —— `quota_fuse` 会给伪实体
    预留证据席位挤掉真正相关的段落，并发子检索还会为每个伪实体多发
    一路检索（实测白付约 3s）。

    修法是把判据从"共现"升级为"**结构相邻**"，同时补上被漏掉的词性
    （nrt 音译人名 / eng 英文词）。改完实测：
        BCZ 误报   63/289 (21.8%)  →  6/289 (2.1%)
        真并列召回      8/10       →  10/10
    """

    def test_true_parallel_still_recognized(self):
        """真并列题必须抽到 ≥2 个实体，否则配额保护根本不会启动。"""
        from src.rag.entities import extract_parallel_entities as E
        cases = [
            ("美国、法国和日本的法定最低饮酒年龄分别是多少？", 3),
            ("长江和黄河的全长分别是多少公里？", 2),
            ("国庆期间，俄罗斯、希腊、巴厘岛的气候和景色分别如何", 3),
        ]
        for q, n in cases:
            ents = E(q)
            assert len(ents) >= n, f"{q} → {ents}（应 ≥{n} 个）"

    def test_translit_and_english_entities(self):
        """补 nrt / eng 两个词性修掉的**漏报**。

        原实现在这两类 query 上抽到 0 个实体、完全不触发配额保护：
            "特斯拉"   被 jieba 标成 nrt（音译人名），不在 _PROPER_POS 里
            "爱因斯坦" 同上
            "Python/Java/Go" 全是 eng
        实测 PAR-08 类的题正是因为没触发而只答出一半（🟡 1/2）。
        """
        from src.rag.entities import extract_parallel_entities as E
        assert len(E("特斯拉和比亚迪 2024 年的全球销量分别是多少？")) >= 2
        assert len(E("爱因斯坦和牛顿谁贡献大")) >= 2
        assert len(E("Python、Java 和 Go 分别由谁创造？")) >= 3

    def test_cross_clause_nouns_are_not_parallel(self):
        """⚠️ 核心回归点：跨从句的专名**不是**并列关系。

        这条就是 BCZ 上误报的原型。句中那个"和"字连的是
        「音乐制作方法和理念」两个普通名词，跟"北京""美国"
        这两个分处不同从句的地点状语毫无关系；这题实际是问
        "这个音乐人是谁"，是**单实体**反向查找。
        原实现给出 ents=['北京','美国']。
        """
        from src.rag.entities import extract_parallel_entities as E
        q = ("一位出生于上世纪 80 年代，毕业于北京著名音乐院校的音乐人，"
             "不仅会弹钢琴，而且还会吹小号，曾前往美国学习先进的"
             "音乐制作方法和理念。这位音乐人 2025 年发布的专辑名字是什么？")
        assert E(q) == [], f"跨从句专名不应判为并列，实际 {E(q)}"

    def test_generic_nouns_are_not_entities(self):
        """通用名词被 jieba 误标成专名时不能进实体列表。

        实测误报来源：城市(ns) / 论文(nz) / 青少年(nr) / 奇特(nz)
        —— "奇特"甚至是个形容词。它们是**问句的骨架**，不是并列对象。
        """
        from src.rag.entities import extract_parallel_entities as E
        ents = E("法国、德国和英国的首都分别是哪座城市？")
        assert "城市" not in ents
        assert set(ents) == {"法国", "德国", "英国"}, ents

    def test_single_intent_no_false_positive(self):
        """单一意图 query 必须返回 [] —— 误报会**主动制造**回归。"""
        from src.rag.entities import extract_parallel_entities as E
        for q in ("俄罗斯的十月革命是怎么回事",
                  "美国一共多少位副总统 历史上",
                  "量子计算是什么",
                  "法国的最低饮酒年龄"):
            assert E(q) == [], f"{q} 不应判为并列，实际 {E(q)}"

    def test_bcz_false_positive_rate_bounded(self):
        """在 BCZ 真实分布上给误报率**上界**，防止后续放宽判据时悄悄退化。

        跳过而非失败：BCZ 数据集需要单独下载，不该让没有数据的
        开发环境跑不过测试。
        """
        import pytest as _pt
        try:
            from evals.datasets import load_bcz
            cases = load_bcz()
        except Exception as e:
            _pt.skip(f"BCZ 数据集不可用: {e}")
        if not cases:
            _pt.skip("BCZ 数据集为空")
        from src.rag.entities import extract_parallel_entities as E
        n_fp = sum(1 for c in cases if len(E(c.question)) >= 2)
        rate = n_fp / len(cases)
        # 实测 2.1%；留到 8% 作为上界（原实现 21.8% 会被这条挡住）
        assert rate <= 0.08, f"BCZ 并列判定率 {rate:.1%} 过高（{n_fp}/{len(cases)}）"


class TestKGStoreThreadSafety:
    """L5 KG 的 SQLite 连接必须是**每线程一条**。

    【实测故障】BrowseComp-ZH 评测时每题都刷：
        [retriever] L5_kg search 异常: bad parameter or other API misuse
    即 sqlite3.InterfaceError（SQLITE_MISUSE）。后果是 **L5 整层静默
    返回空列表** —— 异常被 `_safe_search` 吞掉，检索照常降级继续，
    功能上看不出坏，只是知识图谱这一路的召回一直是 0。
    这类"降级成功但收益归零"的故障最难发现，所以必须有测试盯着。

    【根因】`check_same_thread=False` 只是关掉 Python 层的线程检查，
    并不让连接变成并发安全。而 `retriever._parallel_search` 用线程池
    并行调各层，必然踩中。
    """

    def test_conn_is_thread_local(self):
        """不同线程拿到的必须是**不同**的连接对象。

        直接断言"对象不同"而不是"能跑通"：单连接并发是**竞态**，
        跑通只是这次没撞上，测不出真正的隐患。
        """
        import pytest as _pt
        from concurrent.futures import ThreadPoolExecutor
        try:
            from src.rag.wiki_rag.kg_store import KGStore
            kg = KGStore()
        except Exception as e:
            _pt.skip(f"KG 库不可用: {e}")
        with ThreadPoolExecutor(max_workers=4) as ex:
            ids = list(ex.map(lambda _: id(kg.conn), range(4)))
        assert len(set(ids)) > 1, "各线程拿到同一条连接 → 会触发 SQLITE_MISUSE"

    def test_concurrent_link_no_misuse(self):
        """并发调 link() 不应抛 InterfaceError。"""
        import pytest as _pt
        from concurrent.futures import ThreadPoolExecutor
        try:
            from src.rag.wiki_rag.kg_store import KGStore
            kg = KGStore()
        except Exception as e:
            _pt.skip(f"KG 库不可用: {e}")

        errs: list[str] = []

        def _q(_i):
            try:
                for m in ("中国", "法国", "长江"):
                    kg.link(m)
            except Exception as exc:      # noqa: BLE001
                errs.append(f"{type(exc).__name__}: {exc}")

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(_q, range(8)))
        assert not errs, f"并发 link 报错: {errs[:3]}"


# ══════════════════════════════════════════════════════════════════════════
class TestNTriplesUnescape:
    """Wikidata dump 的 `\\uXXXX` 转义必须在**入库前**还原成真中文。

    【实测故障】KG 库里 96.7% 的 label_zh、70.1% 的 mention 存的都不是
    中文，而是 '\\','u','4','E','2','D' 这些 ASCII 字面字符：

        SELECT label_zh FROM entities WHERE qid='Q148'
        →  '\\u4E2D\\u83EF\\u4EBA\\u6C11\\u5171\\u548C\\u570B'

    后果是 L5 知识图谱层**静默失效**：mention 抽取靠拿 query 的切词去
    `mentions` 表精确查表，表里既然存的是转义串，用户输入的真中文一条
    都对不上 —— L5 每次照常被激活、付 ~434ms，召回却恒为 0。
    它不报任何错，所以光看日志永远发现不了。

    根因是 `09_filter_wikidata_zh.py` 的字面量解析只做了 `m.group(1)`
    取值，漏了 N-Triples 反转义（RDF 1.1 §7 允许 dump 用 \\uXXXX 写非
    ASCII 字符，Wikidata 确实大量这么写）。
    """

    @staticmethod
    def _u():
        """按路径加载 09 脚本（文件名以数字开头，没法 import）。"""
        import importlib.util
        import sys
        from pathlib import Path
        p = (Path(__file__).resolve().parent.parent
             / "src" / "rag" / "scripts" / "09_filter_wikidata_zh.py")
        spec = importlib.util.spec_from_file_location("_m09", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules["_m09"] = m
        spec.loader.exec_module(m)
        return m

    def test_decodes_escaped_chinese(self):
        """核心用例：Q148 在 dump 里的原始写法必须还原成中文。"""
        u = self._u()._nt_unescape
        assert u(r"\u4E2D\u83EF\u4EBA\u6C11\u5171\u548C\u570B") == "中華人民共和國"

    def test_idempotent_on_real_chinese(self):
        """⚠️ 幂等性：已经是真中文的串必须原样返回。

        修复脚本会对同一批数据反复扫描（每轮重查仍含转义的行），
        若不幂等就会在第二次跑时把正常数据二次损坏。
        """
        u = self._u()._nt_unescape
        for s in ["中华人民共和国", "Python", "北京市", ""]:
            assert u(s) == s

    def test_mixed_content_and_backslash_preserved(self):
        """混合内容要正确解码；非 \\u 的反斜杠必须**保留**。

        这正是不能用 codecs.decode(s,'unicode_escape') 的原因：
        它会把 \\n \\t \\\\ 一并吃掉，而实体名里出现反斜杠是合法的。
        """
        u = self._u()._nt_unescape
        assert u(r"1979\u5E74\u7EAA\u5FF5") == "1979年纪念"
        assert u(r"AC\220V") == r"AC\220V"      # \2 不是合法 \uXXXX，原样留

    def test_illegal_surrogate_kept_as_is(self):
        """孤立代理区码点（D800-DFFF）不能解码。

        chr(0xD800) 本身不报错，但这种字符串写进 SQLite 时会抛
        「surrogates not allowed」。宁可留一条脏数据，也不能让整批
        2000 万行的修复中断 —— 所以策略是原样保留而非抛异常。
        """
        u = self._u()._nt_unescape
        assert u(r"\uD800abc") == r"\uD800abc"

    def test_parse_object_applies_unescape(self):
        """反转义必须接在**解析出口**上，而不是只提供一个工具函数。

        这条防的是"函数写了但没接上"——最容易发生的退化。
        """
        m = self._u()
        assert m._parse_object(r'"\u4E2D\u83EF"@zh') == ("中華", "string")
        assert m._parse_object(r'"\u897F\u6B50"') == ("西歐", "string")

    def test_no_raw_group1_left_in_literal_parsing(self):
        """静态兜底：字面量解析处不许再出现裸 `m.group(1)`。

        06 处解析点分散在 _parse_object 与 label/alias/description 三个
        分支里，漏接任何一处都会让对应字段继续被污染，而且**不报错**。
        用源码扫描把这个约束固化下来。
        """
        from pathlib import Path
        p = (Path(__file__).resolve().parent.parent
             / "src" / "rag" / "scripts" / "09_filter_wikidata_zh.py")
        bad = []
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or "_nt_unescape" in s:
                continue
            # _extract_qid/_extract_pid 取的是纯 ASCII 的 QID/PID，无需解码；
            # _nt_unescape 内部的 m.group(1) 是取十六进制数，也要排除。
            if "m.group(1)" in s and "if m else None" not in s and "int(" not in s:
                bad.append(f"{i}: {s[:60]}")
        assert not bad, "字面量解析漏接 _nt_unescape:\n" + "\n".join(bad)


class TestEntityDisambiguationRanking:
    """实体消歧排序质量（L5 KG）。

    背景：修复前 link() 的排序是 `weight DESC, article_rank ...`，
    而 popularity 恒为 0、article_rank 覆盖率仅 0.03%，导致同名候选
    之间**完全没有排序依据**，主实体排不进前列：
        link("北京") → 美國伊利諾伊州的縣城(入度 11)
        link("长江") → 中国歌手与男演员(入度 0)
    这比"召回为空"更危险 —— 它把张冠李戴的事实喂给 LLM 且不报错。
    """

    def test_safe_log_never_raises(self):
        """_safe_log 对 None/0/负数/脏字符串都必须返回有限值。

        它挂在 SQL 的 ORDER BY 里，一旦抛异常整个 link() 就废了；
        返回 None/NaN 则会让排序顺序变得不可预测。
        """
        from src.rag.wiki_rag.kg_store import _safe_log
        for bad in (None, 0, -1, "", "abc", [], 0.0):
            v = _safe_log(bad)
            assert isinstance(v, float) and v == v, f"{bad!r} → {v!r}"
        assert _safe_log(1) == 0.0
        assert _safe_log(_math.e) == pytest.approx(1.0)

    def test_wiki_meta_detected(self):
        """消歧义页/分类页必须被识别，正常实体描述不能误伤。"""
        from src.rag.wiki_rag.kg_store import _is_wiki_meta
        assert _is_wiki_meta("维基媒体消歧义页")
        assert _is_wiki_meta("維基媒體消歧義頁")
        assert _is_wiki_meta("Wikimedia category")
        assert not _is_wiki_meta("中华人民共和国首都")
        assert not _is_wiki_meta(None)
        assert not _is_wiki_meta("")

    def test_to_simplified_is_safe(self):
        """繁简归一：缺 opencc 时降级为恒等，绝不抛异常。"""
        from src.rag.wiki_rag.kg_store import _to_simplified
        assert _to_simplified("") == ""
        out = _to_simplified("蘋果公司")
        # opencc 装了就该转简；没装则原样返回 —— 两种都算通过
        assert out in ("苹果公司", "蘋果公司")

    def test_ranking_uses_combined_score(self):
        """排序必须是 weight×log(popularity) 组合分，不是字典序。

        若退化成 `ORDER BY weight DESC, popularity DESC`，weight 会成为
        第一优先级，「冷门实体的 label(w=1.0)」将永远压过
        「主实体的 alias(w=0.6)」—— 而 Wikidata 主实体的 label 常是繁体，
        简体串恰恰只能作为 alias 命中。用源码扫描把这个约束固化。
        """
        src = _Path(_REPO_ROOT, "src", "rag", "wiki_rag",
                    "kg_store.py").read_text(encoding="utf-8")
        assert "LOG(1 + e.popularity)" in src, "组合分公式被改掉了"
        assert "ORDER BY score DESC" in src, "没有按组合分排序"

    def test_fts_pool_is_bounded(self):
        """FTS 兜底必须**截断候选池**，否则会慢到不可用。

        '\"中国\"*' 前缀命中 156,306 行，不截断直接 ORDER BY 实测 9.1 秒，
        而 L5 整层预算只有几百毫秒。
        """
        src = _Path(_REPO_ROOT, "src", "rag", "wiki_rag",
                    "kg_store.py").read_text(encoding="utf-8")
        assert "LIMIT ?) f" in src, "FTS 子查询没有截断候选池"

    def test_build_script_backfills_popularity(self):
        """建库脚本必须回填 popularity，否则定期重建后排序会再次失效。"""
        src = _Path(_REPO_ROOT, "src", "rag", "scripts",
                    "10_build_kg_sqlite.py").read_text(encoding="utf-8")
        assert "popularity = COALESCE((" in src, "10 脚本没有回填 popularity"
        assert "idx_entities_pop" in src, "popularity 缺索引"
        assert "_gen_suffix_aliases" in src, "10 脚本没有补后缀别名"

    def test_suffix_alias_excludes_university(self):
        """后缀剥离**不得**包含「大学」「公司」。

        「东京大学→东京」「北京大学→北京」会与真正的城市实体撞车。
        这是干跑时实际踩到的坑，必须固化成约束。
        """
        for f in ("10_build_kg_sqlite.py", "15_gen_suffix_aliases.py"):
            src = _Path(_REPO_ROOT, "src", "rag", "scripts", f).read_text(
                encoding="utf-8")
            # 只看 _SUFFIXES/suffixes 元组定义那一段，注释里提到是允许的
            body = src.split("suffixes")[-1].split(")")[0]
            assert '"大学"' not in body, f"{f} 的后缀表里混入了「大学」"
            assert '"公司"' not in body, f"{f} 的后缀表里混入了「公司」"
