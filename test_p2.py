# test_p2.py
"""P2 改造的回归测试。

════════════════════════════════════════════════════════════════════════
覆盖范围
════════════════════════════════════════════════════════════════════════
P2-2c 近重去重 + MMR（rag/dedup.py）
    TestNearDupDedup     —— 转载被删、不同答案绝不误删、降级路径
    TestMMRRerank        —— 相关性量纲归一化、λ=1 退化、首段取最相关
    TestDiversifyE2E     —— 两阶段串联（MMR 单独不够，这是设计核心）
    TestLayeredDedup     —— 只对 L4 编码（延迟优化）、向量只编一次、metadata 不污染
    TestRetrieverWiring  —— 候选池放大、开关全关时零行为差异

P2-3 追问推荐（followup.py）
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
（P2-2c 需要真实 BGE-M3 向量，P0 大量用 mock），混在一起会让
"只想跑某一块"变得困难。分文件后可以 `pytest test_p2.py -k NearDup`
精准定位。

⚠️ 部分测试需要真实 BGE-M3（首次加载约 5s，之后进程内复用）。
用 module 级 fixture 保证只加载一次。
"""
from __future__ import annotations

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
    from rag.embedder import Embedder
    return Embedder()


def _P(title, text, url="", score=1.0, layer="L4_web"):
    """构造 Passage 的简写。"""
    from rag.types import Passage
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
#                    P2-2c ①：近重去重（硬删，事实判断）
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
        from rag.dedup import drop_near_duplicates
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
        from rag.dedup import drop_near_duplicates
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
        from rag.dedup import drop_near_duplicates
        out = drop_near_duplicates(reposts, embedder=None, verbose=False)
        assert len(out) == len(reposts)
        assert out is reposts or [p.url for p in out] == [p.url for p in reposts]

    def test_single_or_empty_input(self, emb):
        """0/1 条输入直接短路，不该触发编码。"""
        from rag.dedup import drop_near_duplicates
        assert drop_near_duplicates([], embedder=emb) == []
        one = [_P("t", "x")]
        assert drop_near_duplicates(one, embedder=emb) == one

    def test_reuses_metadata_embedding(self, reposts):
        """metadata 里已有向量时必须**复用**，不再调编码器（零成本路径）。

        这是性能上的关键优化：L2/L3 检索时本来就编码过 passage，
        若它们把向量写进 metadata，去重就完全免费。
        """
        from rag.dedup import drop_near_duplicates

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
        from rag.dedup import drop_near_duplicates
        reposts[0].metadata["embedding"] = [1.0, 0.0]
        reposts[1].metadata["embedding"] = [1.0, 0.0, 0.0]   # 维度不同
        reposts[2].metadata["embedding"] = [0.0, 1.0]
        out = drop_near_duplicates(reposts, embedder=None, verbose=False)
        assert len(out) == 3, "维度不一致应整体降级为不去重"


# ════════════════════════════════════════════════════════════════════════
#                    P2-2c ②：MMR（软重排，偏好判断）
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
        from rag.dedup import mmr_rerank
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
        from rag.dedup import mmr_rerank
        out = mmr_rerank(self._cands(), embedder=emb, top_k=3)
        titles = [p.title for p in out]
        assert titles[1] == "GDP总量重复", (
            f"相关性项被量纲压倒：第 2 段是 {titles[1]!r}"
        )

    def test_lambda_one_is_identity(self, emb):
        """λ=1.0 → 多样性项系数为 0 → 逐段等价于原排序。"""
        from rag.dedup import mmr_rerank
        cands = self._cands()
        out = mmr_rerank(cands, embedder=emb, top_k=4, lambda_=1.0)
        assert [p.title for p in out] == [p.title for p in cands]

    def test_diversity_wins_at_low_lambda(self, emb):
        """λ 很小时多样性主导，重复项应被压到后面（验证 λ 真的是旋钮）。"""
        from rag.dedup import mmr_rerank
        out = mmr_rerank(self._cands(), embedder=emb, top_k=2, lambda_=0.1)
        assert out[1].title != "GDP总量重复", (
            "λ=0.1 时多样性应主导，重复项不该排第 2"
        )

    def test_graceful_degradation(self):
        """无 embedder → 原序截断，不抛异常。"""
        from rag.dedup import mmr_rerank
        cands = self._cands()
        out = mmr_rerank(cands, embedder=None, top_k=2)
        assert [p.title for p in out] == [p.title for p in cands[:2]]

    def test_identical_scores_no_crash(self, emb):
        """所有分数相同时（min==max）不能除零。"""
        from rag.dedup import mmr_rerank
        cands = self._cands()
        for p in cands:
            p.score = 0.02
        out = mmr_rerank(cands, embedder=emb, top_k=3)
        assert len(out) == 3


# ════════════════════════════════════════════════════════════════════════
#              P2-2c ③：两阶段串联（本设计的核心价值）
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
        from rag.dedup import mmr_rerank
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
        from rag.dedup import diversify
        cands = reposts + complements
        out = diversify(cands, embedder=emb, top_k=3, verbose=False)
        repost_urls = {p.url for p in reposts}
        n = sum(1 for p in out if p.url in repost_urls)
        assert n == 1, f"转载仍占 {n} 席（应恰好 1 席代表）"
        assert len(out) == 3, "席位应被填满（去重后仍有足够候选）"

    def test_both_disabled_is_identity(self, emb, reposts, complements):
        """两个开关都关时必须**逐段等价**于原输入（零回归保证）。"""
        from rag.dedup import diversify
        cands = reposts + complements
        out = diversify(cands, embedder=emb, top_k=3,
                        enable_near_dup=False, enable_mmr=False, verbose=False)
        assert [p.url for p in out] == [p.url for p in cands[:3]]

    def test_empty_input(self, emb):
        from rag.dedup import diversify
        assert diversify([], embedder=emb, top_k=3) == []


# ════════════════════════════════════════════════════════════════════════
#          P2-2c ⑤：分层去重（方案 B —— 只对 L4 编码，降延迟）
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
        from rag.embedder import Embedder

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
        from rag.dedup import diversify
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
        from rag.dedup import diversify
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
        from rag.dedup import diversify
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
        from rag.dedup import diversify
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
        from rag.dedup import diversify
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
        from rag.dedup import diversify, EMBEDDING_META_KEY
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
        from rag.dedup import diversify, EMBEDDING_META_KEY
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
        from rag.dedup import diversify
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
        from rag.dedup import diversify, mmr_rerank
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
        from rag.dedup import diversify
        offline = [
            _P(f"离线{i}", f"这是第{i}条维基材料，内容各不相同。",
               "", 0.02 - i * 0.001, layer="L2_wiki")
            for i in range(5)
        ]
        out = diversify(reposts + offline, embedder=self._spy(),
                        top_k=6, verbose=False)
        assert len(out) == 6, f"席位未填满，只剩 {len(out)} 段"

    def test_config_parses_layers(self):
        from rag import config as rc
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
        from rag import config as rc
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
        from rag import config as rc
        monkeypatch.setenv("RAG_DEDUP_LAYERS", "L4_web, L3_history")
        importlib.reload(rc)
        try:
            assert rc.DEDUP_LAYERS == frozenset({"L4_web", "L3_history"})
        finally:
            monkeypatch.delenv("RAG_DEDUP_LAYERS", raising=False)
            importlib.reload(rc)

    def test_switches_off_still_zero_encoding(self, reposts):
        """两开关全关时不该编码（零回归 + 零成本）。"""
        from rag.dedup import diversify
        spy = self._spy()
        out = diversify(reposts, embedder=spy, top_k=3,
                        enable_near_dup=False, enable_mmr=False, verbose=False)
        assert spy.calls == 0
        assert [p.url for p in out] == [p.url for p in reposts[:3]]


# ════════════════════════════════════════════════════════════════════════
#                 P2-2c ④：retriever 接线（候选池放大）
# ════════════════════════════════════════════════════════════════════════
class TestRetrieverWiring:
    def test_config_switches_exist(self):
        from rag import config as rc
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
        src = open("rag/retriever.py", encoding="utf-8").read()
        assert "FUSION_CANDIDATE_MULTIPLIER" in src
        assert "_cand_k" in src
        # 必须传放大后的 _cand_k 给 quota_fuse，而不是 self.fusion_top_k
        assert "top_k=_cand_k" in src
        # diversify 必须收窄回 fusion_top_k
        assert "top_k=self.fusion_top_k" in src

    def test_dedup_runs_before_rerank(self):
        """去重必须在 rerank **之前**：rerank 是最贵的一步，先减候选省钱。"""
        src = open("rag/retriever.py", encoding="utf-8").read()
        i_dedup = src.index("rag_dedup.diversify")
        i_rerank = src.index("fused = rerank(query")
        assert i_dedup < i_rerank, "diversify 必须在 rerank 之前"

    def test_dedup_after_quota_fuse(self):
        """去重必须在配额融合**之后**：否则弱实体的证据可能先被当近重删掉。"""
        src = open("rag/retriever.py", encoding="utf-8").read()
        i_quota = src.index("fused = quota_fuse")
        i_dedup = src.index("rag_dedup.diversify")
        assert i_quota < i_dedup, "quota_fuse 必须在 diversify 之前"


# ════════════════════════════════════════════════════════════════════════
#                     P2-3 ①：追问解析与过滤
# ════════════════════════════════════════════════════════════════════════
class TestFollowupParse:
    def test_normal_parse(self):
        from followup import parse_followups, FOLLOWUP_MARKER
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
        from followup import parse_followups
        plain = "这是一个没有追问区的普通答案。"
        r = parse_followups(plain, "问题")
        assert r.body == plain
        assert r.followups == []

    def test_empty_input(self):
        from followup import parse_followups
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
        from followup import parse_followups, FOLLOWUP_MARKER
        raw = f"答案内容。\n{FOLLOWUP_MARKER}\n{bad}\n"
        r = parse_followups(raw, "上半年GDP增长多少")
        assert r.followups == [], f"未过滤掉「{why}」的输出: {r.followups}"

    def test_dedup_similar_followups(self):
        """归一化后相同的追问只保留一条。"""
        from followup import parse_followups, FOLLOWUP_MARKER
        raw = (f"答案。\n{FOLLOWUP_MARKER}\n"
               "三次产业增速是多少？\n三次产业增速是多少\n")
        r = parse_followups(raw, "GDP")
        assert len(r.followups) == 1

    def test_count_capped(self):
        """最多返回 FOLLOWUP_COUNT 条（防止模型输出一长串）。"""
        from followup import parse_followups, FOLLOWUP_MARKER, FOLLOWUP_COUNT
        qs = "\n".join(f"第{i}个问题是什么？" for i in range(1, 10))
        r = parse_followups(f"答案。\n{FOLLOWUP_MARKER}\n{qs}", "q")
        assert len(r.followups) <= FOLLOWUP_COUNT


# ════════════════════════════════════════════════════════════════════════
#            P2-3 ②：流式分隔符抑制（最容易漏的坑）
# ════════════════════════════════════════════════════════════════════════
class TestStreamFilter:
    """流式下分隔符会被**切碎**在多个 chunk 里。

    逐 chunk 做 `in` 判断必然漏检，碎片会漏给用户。
    `StreamFilter` 用"滞后输出 len(marker)-1 个字符"解决。
    """

    def test_fragmented_marker_blocked(self):
        """⚠️ 核心：分隔符被切成碎片也不能漏给用户。"""
        from followup import StreamFilter, FOLLOWUP_MARKER
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
        from followup import StreamFilter
        f = StreamFilter()
        chunks = ["这是", "一个普通", "答案，没有追问区。"]
        got = "".join(f.feed(c) for c in chunks) + f.flush()
        assert got == "".join(chunks)

    def test_single_char_chunks(self):
        """极端情况：每个 chunk 只有 1 个字符。"""
        from followup import StreamFilter, FOLLOWUP_MARKER
        raw = f"答案文本。\n{FOLLOWUP_MARKER}\n追问是什么？\n"
        f = StreamFilter()
        visible = "".join(f.feed(c) for c in raw) + f.flush()
        assert FOLLOWUP_MARKER not in visible and "#" not in visible
        assert visible == "答案文本。\n"

    def test_raw_preserves_everything(self):
        """raw() 必须保留完整原始文本（含追问区），供解析用。"""
        from followup import StreamFilter, FOLLOWUP_MARKER
        raw = f"答案。\n{FOLLOWUP_MARKER}\n问题是什么？\n"
        f = StreamFilter()
        for c in raw:
            f.feed(c)
        f.flush()
        assert f.raw() == raw

    def test_interrupted_mid_stream(self):
        """流被打断（未 flush）时仍能拿到已生成部分的正文。"""
        from followup import StreamFilter
        f = StreamFilter()
        f.feed("上半年GDP同比增长")
        r = f.result("GDP")
        # 没有分隔符 → parse_followups 原样返回，正文完整、追问为空
        assert "上半年GDP" in r.body
        assert r.followups == []

    def test_empty_feed(self):
        from followup import StreamFilter
        f = StreamFilter()
        assert f.feed("") == ""
        assert f.flush() == ""


# ════════════════════════════════════════════════════════════════════════
#                   P2-3 ③：澄清提问（保守判定）
# ════════════════════════════════════════════════════════════════════════
class TestClarify:
    """澄清提问的**代价不对称**：误触发会打断正常问答（负体验），
    而漏判只是少了个功能。所以默认关闭 + 三重保守约束。
    """

    def test_disabled_by_default(self):
        """⚠️ 必须默认关闭 —— 精度未经线上验证前不该打断用户。"""
        from followup import ENABLE_CLARIFY, should_clarify
        assert ENABLE_CLARIFY is False, "澄清提问不应默认开启"
        d = should_clarify("苹果多少钱", [], None)
        assert d.need is False
        assert "未启用" in d.reason

    def test_too_few_passages(self, emb):
        """证据太少时"分裂"没有统计意义 → 不反问。"""
        from followup import should_clarify
        ps = [_P("a", "内容一"), _P("b", "内容二")]
        d = should_clarify("苹果", ps, emb, enable=True)
        assert d.need is False
        assert "统计意义" in d.reason

    def test_no_embedder(self):
        from followup import should_clarify
        ps = [_P(f"t{i}", f"内容{i}") for i in range(5)]
        d = should_clarify("q", ps, None, enable=True)
        assert d.need is False

    def test_coherent_evidence_no_clarify(self, emb):
        """证据同源（都在讲一件事）时**绝不能**反问。"""
        from followup import should_clarify
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
        from followup import should_clarify
        d = should_clarify("q", [], None, enable=True)
        assert d.reason, "reason 为空会让线上无法调参"


# ════════════════════════════════════════════════════════════════════════
#         P2-3 ④：agent 端到端（追问区绝不能污染 memory / 缓存）
# ════════════════════════════════════════════════════════════════════════
_RAW_ANSWER = None      # 在 fixture 里按 marker 组装


@pytest.fixture()
def followup_agent(monkeypatch, tmp_path):
    """mock 掉 LLM / 改写的 agent，避免联网与付费调用。"""
    import agent as agent_mod
    from agent import AgenticSearchAgent
    from followup import FOLLOWUP_MARKER

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
        from followup import FOLLOWUP_MARKER
        assert FOLLOWUP_MARKER not in res.text
        assert res.text == _BODY
        assert len(res.followups) == 3

    def test_memory_never_polluted(self, followup_agent):
        """⚠️ 追问区绝不能进 memory —— 会污染后续所有轮次的对话历史。"""
        from followup import FOLLOWUP_MARKER
        followup_agent.chat("上半年GDP增长多少", verbose=False, session_id="s2")
        last = followup_agent._get_memory("s2").get_messages()[-1]["content"]
        assert FOLLOWUP_MARKER not in last
        assert "三次产业" not in last
        assert last == _BODY

    def test_l1_cache_never_polluted(self, followup_agent):
        """⚠️ 最严重：追问区若进 L1，下次命中会**直接吐给用户**并永久固化。"""
        from followup import FOLLOWUP_MARKER
        followup_agent.chat("量子计算的基本原理是什么", verbose=False,
                            session_id="s3")
        cached = followup_agent.qa_cache.get("量子计算的基本原理是什么")
        if cached is not None:
            assert FOLLOWUP_MARKER not in cached
            assert "三次产业" not in cached

    def test_stream_hides_marker(self, followup_agent):
        """流式下分隔符（被切碎）不能漏给用户。"""
        from followup import FOLLOWUP_MARKER
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
        from followup import FOLLOWUP_MARKER
        stream = followup_agent.chat("再问GDP", verbose=False, is_stream=True,
                                     return_result=True, session_id="s5")
        list(stream)
        last = followup_agent._get_memory("s5").get_messages()[-1]["content"]
        assert FOLLOWUP_MARKER not in last and "三次产业" not in last

    def test_backward_compat_plain_str(self, followup_agent):
        """不传 return_result 时仍返回裸 str（且已剥离）。"""
        from followup import FOLLOWUP_MARKER
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
#                   P2-3 ⑤：前端渲染契约
# ════════════════════════════════════════════════════════════════════════
class TestFollowupFrontend:
    def test_prompt_contains_instruction(self):
        """summary system prompt 必须同时含追问指令**与**安全规则。

        后半句是防回归：组装逻辑改成 list+join 后，若不小心漏掉
        EVIDENCE_GUARD_PROMPT，injection 防护会静默失效。
        """
        from configs.prompts import build_summary_system
        from followup import FOLLOWUP_MARKER
        p = build_summary_system(context="[环境信息] test")
        assert FOLLOWUP_MARKER in p
        assert "外部资料安全规则" in p, "安全规则被挤掉了（P0-4 回归）"

    def test_cli_icon_and_render(self):
        import main as cli
        assert "followup" in cli.STAGE_ICON

    def test_cli_silent_when_empty(self):
        """无追问时 CLI 必须零输出（不能打"无追问推荐"之类的废话）。"""
        import io
        import contextlib
        import main as cli
        from answer_types import AnswerResult
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli._print_followups(AnswerResult(text="闲聊"))
            cli._print_followups(None)
        assert buf.getvalue() == ""

    def test_cli_renders_followups(self):
        import io
        import contextlib
        import main as cli
        from answer_types import AnswerResult
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli._print_followups(AnswerResult(text="a", followups=["问题一？"]))
        out = buf.getvalue()
        assert "你可能还想问" in out and "问题一？" in out

    def test_answer_result_render(self):
        from answer_types import AnswerResult
        r = AnswerResult(text="a", followups=["问题一？", "问题二？"])
        md = r.render_followups_markdown()
        assert "你可能还想问" in md and "问题一？" in md
        assert AnswerResult(text="a").render_followups_markdown() == ""

    def test_web_wiring(self):
        """Web 端必须有独立渲染函数，且 followup 事件不重复渲染成步骤块。"""
        src = open("main_web.py", encoding="utf-8").read()
        assert "_render_followups_md" in src
        assert '("sources", "followup")' in src, (
            "followup 事件未被步骤块跳过，会与独立区块重复展示"
        )

    def test_scripts_contract(self):
        src = open("scripts/search.py", encoding="utf-8").read()
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
        from rag.textclean import _is_noise_line
        assert _is_noise_line(line), f"漏删噪声: {line!r}"

    @pytest.mark.parametrize("line", CONTENT)
    def test_content_preserved(self, line):
        from rag.textclean import _is_noise_line
        assert not _is_noise_line(line), f"误删正文: {line!r}"

    def test_markdown_heading_whitelist(self):
        """⚠️ Markdown 标题必须**最优先**放行。

        `### 七月` 只有 4 个字符、没有任何句子标记，会被"孤立短标签"
        规则判为噪声。但它是**结构信息**，告诉模型"接下来讲七月"，
        对多月份对比类问题价值很高。所以白名单必须在判定链最前面。
        """
        from rag.textclean import _is_noise_line, SHORT_LABEL_MAX
        assert len("### 七月") < SHORT_LABEL_MAX      # 确认它真的会命中短标签规则
        assert not _is_noise_line("### 七月")
        assert not _is_noise_line("# 概述")

    def test_colon_bullet_preserved(self):
        """⚠️ 冒号式要点是高价值结构化事实，不能被短标签规则删掉。

        `降雨高峰：通常集中在 12 月至 2 月` 这类是直接的答案素材。
        它们靠"冒号在句子标记里"被放行 —— 若哪天把冒号从
        `_RE_SENTENCE_MARK` 移除，这类要点会被大批误删。
        """
        from rag.textclean import _is_noise_line
        for s in ["降雨高峰：通常集中在 12 月至 2 月",
                  "天氣特色：午後陣雨較多",
                  "注意事項：建議攜帶輕便雨衣"]:
            assert not _is_noise_line(s), f"冒号要点被误删: {s!r}"

    def test_digit_run_needs_both_conditions(self):
        """连续数字串必须同时满足"有长数字段"和"整行只有数字"。

        只看"有 8 位连续数字"会误删
        `订单号 20260807123456 已生成，请查收。` 这类正文。
        """
        from rag.textclean import _is_noise_line
        assert _is_noise_line("12345678910月1112")
        assert not _is_noise_line("订单号 20260807123456 已生成，请查收。")

    def test_short_text_untouched(self):
        """短于 MIN_LEN_TO_CLEAN 的文本原样返回。

        DDG/Serper/Bing 的摘要实测 40~100 字（DDG 中位数 61），
        本身就是搜索引擎抽好的句子，清洗只有风险没有收益。
        """
        from rag.textclean import clean_snippet, MIN_LEN_TO_CLEAN
        s = "雅典, 希腊 - 秋季. 十月份的平均海水温度: 22.3°摄氏度."
        assert len(s) < MIN_LEN_TO_CLEAN
        assert clean_snippet(s) == s

    def test_never_empties_evidence(self):
        """⚠️ 本组最重要：清洗**绝不能**把证据洗空。

        清洗是有损变换。若规则过激（或遇到极端页面结构），
        必须放弃清洗、返回原文 —— 带噪声的证据仍然可用，
        被洗空的证据等于丢了这一路召回。
        """
        from rag.textclean import clean_snippet
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
        from rag.textclean import clean_snippet, MIN_KEEP_RATIO
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
        from rag.textclean import clean_snippet
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
        from rag.textclean import clean_snippet
        s = ("這是第一段正文，講述乾季的天氣狀況與適合的活動安排。" * 3 +
             "[...]4.5[...]" +
             "這是第二段正文，講述雨季的降雨量與注意事項。" * 3)
        out = clean_snippet(s)
        assert "4.5" not in out.split("\n"), "片段间的噪声未被清掉"
        assert "第一段正文" in out and "第二段正文" in out

    def test_switch_off_is_identity(self):
        """开关关闭时逐字原样返回（零回归保证）。"""
        from rag.textclean import clean_snippet
        s = "||  |  |\n4.5\nTWD 0 起\n" + "真实正文内容，包含足够多的中文字符。" * 10
        assert clean_snippet(s, enable=False) == s
        assert clean_snippet(s, enable=True) != s

    def test_clean_stats_shape(self):
        """clean_stats 的字段契约（上线后要打日志观测）。"""
        from rag.textclean import clean_stats
        s = "|---\n" * 5 + "这是一段足够长的正文内容，讲了很多有价值的事实。" * 6
        st = clean_stats(s)
        for k in ("raw_len", "clean_len", "removed_chars", "removed_ratio",
                  "raw_lines", "kept_lines", "fell_back"):
            assert k in st, f"缺少字段 {k}"
        assert st["removed_chars"] == st["raw_len"] - st["clean_len"]
        assert 0.0 <= st["removed_ratio"] <= 1.0

    def test_l4_layer_wired(self):
        """L4 层必须调用清洗（接线正确性）。"""
        src = open("rag/layers.py", encoding="utf-8").read()
        assert "from .textclean import clean_snippet" in src
        assert "clean_snippet(r.get(\"snippet\", \"\"))" in src

    def test_l4_search_applies_clean(self, monkeypatch):
        """端到端：L4.search 产出的 Passage.text 必须已清洗。"""
        import rag.layers as L
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
        from configs import config
        from rag.textclean import clean_stats
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
