# test_kg_linker.py
"""L5 实体链接：上下文消歧的回归测试。

════════════════════════════════════════════════════════════════════════
本文件覆盖什么
════════════════════════════════════════════════════════════════════════
    context_lexical_score   —— description ↔ query 的词面重叠打分
    Linker.link             —— 上下文能真正影响排序（核心修复）
    退化路径                —— 无上下文 / 无 description / 无热门向量库

════════════════════════════════════════════════════════════════════════
为什么要有这些测试（query_context 传了却毫无作用）
════════════════════════════════════════════════════════════════════════
修复前实测：同一 mention、语义完全相反的两个上下文，返回结果和分数
**完全一致** ——
    "华盛顿是美国第一任总统"  → 華盛頓哥倫比亞特區(0.18)   ✗
    "美国首都华盛顿的人口"    → 華盛頓哥倫比亞特區(0.18)   ✓
    "特斯拉发明了交流电"      → 特斯拉公司(0.07)           ✗
    "特斯拉发布了新款电动车"  → 特斯拉公司(0.07)           ✓
即消歧退化成"按热度返回固定答案"，碰巧对一半。

根因：`link()` 里的 cosine 只对**命中热门实体向量库**的候选计算，而该库
仅 2871 条（KG 有 1038 万，覆盖率 0.03%）。绝大多数候选走 cold 分支
`score = 0.5 * prior`，与上下文无关。

修复：引入 description 词面重叠分，对**所有**候选可用。
实测 9 个消歧 case：现状 5/9 → 词面分 8/9；
而"给 cold 候选现算 description embedding"反而是 4/9 且 673ms/题
（description 太短，BGE-M3 不稳定）—— 所以选了词面方案。

设计原则：**不加载 10.9GB 真实 KG、不调 BGE-M3**。用假候选直接驱动
打分逻辑，测试毫秒级完成且结果确定。
运行：
    pytest tests/test_kg_linker.py -v
"""
from __future__ import annotations

import math

import pytest

from src.rag.wiki_rag import linker as lk_mod
from src.rag.wiki_rag.linker import Linker, context_lexical_score


# =============================================================================
# context_lexical_score
# =============================================================================

def test_lexical_score_hits_overlapping_description():
    """query 提到了 description 的内容 → 分数应显著为正。"""
    s = context_lexical_score("华盛顿是美国第一任总统", "第1任美国总统")
    assert s > 0.3, f"重叠明显却只有 {s}"


def test_lexical_score_zero_for_unrelated_description():
    """无关 description → 0 分，不干扰先验排序。

    这是"有信号才发声"特性的基础：正因为无关时恒为 0，才敢把它的权重
    给到和先验分等权（_CTX_W=0.5）。
    """
    assert context_lexical_score("苹果这种水果的营养价值",
                                 "美國電子產品公司") == 0.0


def test_lexical_score_empty_description():
    assert context_lexical_score("任意 query", "") == 0.0


def test_lexical_score_bounded():
    """取值必须落在 0~1，否则会破坏与先验分的加权尺度。"""
    for q, d in [("完全一样的描述", "完全一样的描述"),
                 ("abc", "xyz"), ("短", "也短"),
                 ("很长的一段查询语句用来测试边界", "描述")]:
        s = context_lexical_score(q, d)
        assert 0.0 <= s <= 1.0, f"({q!r},{d!r}) → {s}"


def test_lexical_score_identical_is_one():
    """description 完全被 query 包含时应为 1.0（分母是 description 侧）。"""
    assert context_lexical_score("他是第1任美国总统啊", "第1任美国总统") == 1.0


def test_lexical_score_uses_description_as_denominator():
    """长 query 不应压低分数。

    若用并集或 query 侧作分母，query 越长所有候选分数一起趋 0，
    候选间的相对差异被抹平 → 判别力丧失。
    """
    desc = "第1任美国总统"
    short = context_lexical_score("第1任美国总统", desc)
    long_q = context_lexical_score(
        "请问历史上那位被称为第1任美国总统的人物究竟是谁呢我很好奇", desc)
    assert short == pytest.approx(long_q), (short, long_q)


def test_single_char_description_no_crash():
    """单字 description 不能除零崩溃。"""
    assert 0.0 <= context_lexical_score("测试查询", "人") <= 1.0


# =============================================================================
# Linker.link —— 上下文真正影响排序
# =============================================================================

def _make_linker(monkeypatch, candidates):
    """构造一个不依赖真实 KG / 向量库 / BGE-M3 的 Linker。

    绕过 __init__（它会 load_config + mmap 2.6GB 向量），直接塞入
    link() 所需的最小状态。`emb=None` 走"无热门向量库"分支 —— 这正是
    线上绝大多数候选的实际路径（覆盖率仅 0.03%）。
    """
    lk = Linker.__new__(Linker)
    lk.kg_cfg = {"link_final_k": 5}
    lk.emb_cfg = {}
    lk.emb = None
    lk.qid_to_row = {}

    class _KG:
        # 假 conn：_is_human 会查 P31→Q5。这里返回 None 表示"不是人物"，
        # 让类型分恒为 0，从而隔离出词面分的效果单独测试。
        # 人物类型分的行为由 test_type_score_* 系列单独覆盖。
        class _Conn:
            @staticmethod
            def execute(*_a, **_k):
                class _R:
                    @staticmethod
                    def fetchone():
                        return None
                return _R()

        conn = _Conn()

        @staticmethod
        def link(mention):
            # 返回副本：link() 会往 dict 里写 prior/ctx/type_s/score
            return [dict(c) for c in candidates]

    lk.kg = _KG()
    return lk


# 「华盛顿」的真实候选结构：城市热度(4041)远高于人物(91)，
# 所以纯先验必然选城市 —— 这正是修复前的错误来源。
_WASHINGTON = [
    {"qid": "Q61", "label_zh": "華盛頓哥倫比亞特區",
     "description": "美利堅合眾國的首都", "score": 4.98, "weight": 0.6},
    {"qid": "Q23", "label_zh": "喬治·華盛頓",
     "description": "第1任美国总统", "score": 2.71, "weight": 0.6},
]


def test_context_selects_person(monkeypatch):
    """上下文指向人物 → 应选人物，尽管城市先验分高得多。"""
    lk = _make_linker(monkeypatch, _WASHINGTON)
    got = lk.link("华盛顿", query_context="华盛顿是美国第一任总统")
    assert got[0]["qid"] == "Q23", [c["qid"] for c in got]


def test_context_selects_city(monkeypatch):
    """上下文指向城市 → 应选城市。"""
    lk = _make_linker(monkeypatch, _WASHINGTON)
    got = lk.link("华盛顿", query_context="美国首都华盛顿的人口是多少")
    assert got[0]["qid"] == "Q61", [c["qid"] for c in got]


def test_opposite_contexts_give_different_results(monkeypatch):
    """【核心防回归】两个相反上下文必须给出不同结果。

    修复前两者返回完全相同的实体和分数 —— 这个断言直接锁死该 bug。
    """
    lk = _make_linker(monkeypatch, _WASHINGTON)
    a = lk.link("华盛顿", query_context="华盛顿是美国第一任总统")[0]
    lk = _make_linker(monkeypatch, _WASHINGTON)
    b = lk.link("华盛顿", query_context="美国首都华盛顿的人口是多少")[0]
    assert a["qid"] != b["qid"], "上下文未生效，消歧退化成固定答案"


def test_no_context_falls_back_to_prior(monkeypatch):
    """不传上下文 → 退回先验排序（热度最高者），行为与修复前一致。"""
    lk = _make_linker(monkeypatch, _WASHINGTON)
    assert lk.link("华盛顿")[0]["qid"] == "Q61"


def test_candidates_without_description(monkeypatch):
    """description 全为 None 时不能崩，且应退化为先验排序。

    实测库里确有大量 description 为 NULL 的实体（如 Q989456「华盛顿」）。
    """
    cands = [
        {"qid": "Q1", "label_zh": "甲", "description": None,
         "score": 1.0, "weight": 1.0},
        {"qid": "Q2", "label_zh": "乙", "description": None,
         "score": 3.0, "weight": 1.0},
    ]
    lk = _make_linker(monkeypatch, cands)
    got = lk.link("x", query_context="任意上下文")
    assert got[0]["qid"] == "Q2"


def test_empty_candidates(monkeypatch):
    lk = _make_linker(monkeypatch, [])
    assert lk.link("不存在", query_context="ctx") == []


def test_single_candidate_short_circuit(monkeypatch):
    """单候选走早退分支，也必须带上 ctx 字段与合法分数。"""
    cands = [{"qid": "Q1", "label_zh": "甲", "description": "描述",
              "score": 2.0, "weight": 1.0}]
    lk = _make_linker(monkeypatch, cands)
    got = lk.link("x", query_context="描述里的内容")
    assert len(got) == 1
    assert got[0]["qid"] == "Q1"
    assert math.isfinite(got[0]["score"])


def test_scores_are_finite_and_sorted(monkeypatch):
    """分数必须有限且降序 —— 上游 RRF/校准依赖这个前提。"""
    lk = _make_linker(monkeypatch, _WASHINGTON)
    got = lk.link("华盛顿", query_context="第1任美国总统")
    scores = [c["score"] for c in got]
    assert all(math.isfinite(s) for s in scores), scores
    assert scores == sorted(scores, reverse=True), scores


def test_top_k_respected(monkeypatch):
    cands = [
        {"qid": f"Q{i}", "label_zh": f"e{i}", "description": f"描述{i}",
         "score": float(i), "weight": 1.0} for i in range(1, 8)
    ]
    lk = _make_linker(monkeypatch, cands)
    assert len(lk.link("x", query_context="ctx", top_k=3)) == 3


def test_ctx_weight_is_configurable():
    """_CTX_W 应在 (0,1) 内，否则加权尺度失衡。"""
    assert 0.0 < lk_mod._CTX_W < 1.0


# =============================================================================
# 类型-意图消歧（query 问人还是问物）
# =============================================================================

@pytest.mark.parametrize("query", [
    "牛顿提出了万有引力定律",
    "华盛顿是美国第一任总统",
    "特斯拉发明了交流电",
])
def test_intent_detects_person(query):
    assert lk_mod.query_person_intent(query) > 0, query


@pytest.mark.parametrize("query", [
    "力的单位牛顿等于多少",
    "美国首都华盛顿的人口是多少",
    "苹果这种水果的营养价值",
])
def test_intent_detects_nonperson(query):
    assert lk_mod.query_person_intent(query) < 0, query


@pytest.mark.parametrize("query", [
    "水星是太阳系最内侧的行星",
    "长城是中国古代军事防御工程",
    "他在欧洲的一所大学学习",
])
def test_intent_returns_zero_without_cues(query):
    """无线索时必须返回 0 —— 这是"只修不坏"保证的基础。

    实测这三个 query 本来就排对了，类型分不介入正是期望行为。
    """
    assert lk_mod.query_person_intent(query) == 0, query


def test_intent_empty_query():
    assert lk_mod.query_person_intent("") == 0


def test_type_score_zero_when_no_intent(monkeypatch):
    """intent==0 → 类型分恒为 0，且**不应查库**（省一次 IO）。"""
    lk = _make_linker(monkeypatch, _WASHINGTON)
    assert lk._type_score("Q23", 0) == 0.0


def test_type_score_agrees_and_conflicts(monkeypatch):
    """人物实体：问人 → +1，问物 → -1。"""
    lk = _make_linker(monkeypatch, _WASHINGTON)
    # 让 _is_human 认为 Q23 是人
    monkeypatch.setattr(lk, "_is_human", lambda q: q == "Q23")
    assert lk._type_score("Q23", 1) == 1.0     # 问人，是人 → 一致
    assert lk._type_score("Q23", -1) == -1.0   # 问物，是人 → 冲突
    assert lk._type_score("Q61", 1) == -1.0    # 问人，非人 → 冲突
    assert lk._type_score("Q61", -1) == 1.0    # 问物，非人 → 一致


def test_type_score_survives_db_error(monkeypatch):
    """查库异常时不能抛 —— L5 异常会被上层吞掉导致整层静默返回空。"""
    lk = _make_linker(monkeypatch, _WASHINGTON)

    class _Boom:
        @staticmethod
        def execute(*_a, **_k):
            raise RuntimeError("db down")

    lk.kg.conn = _Boom()
    lk.__dict__.pop("_human_cache", None)
    assert lk._is_human("Q23") is False        # 容错为"非人物"
    assert lk._type_score("Q23", 1) == -1.0    # 不抛异常


def test_type_score_lifts_person_over_popular_homonym(monkeypatch):
    """【核心防回归】类型分应能翻盘先验热度差。

    真实场景：牛顿(城市, pop=135) vs 艾薩克·牛頓(pop=60)。
    修复前必选城市；query 说"提出了…定律"时应选人物。
    """
    cands = [
        {"qid": "Q49196", "label_zh": "牛顿", "description": "美国城市",
         "score": 4.91, "weight": 1.0},
        {"qid": "Q935", "label_zh": "艾薩克·牛頓",
         "description": "英格兰物理学家、数学家、天文学家和自然哲学家",
         "score": 2.47, "weight": 0.6},
    ]
    lk = _make_linker(monkeypatch, cands)
    monkeypatch.setattr(lk, "_is_human", lambda q: q == "Q935")
    got = lk.link("牛顿", query_context="牛顿提出了万有引力定律")
    assert got[0]["qid"] == "Q935", [(c["qid"], c["score"]) for c in got]


def test_type_score_keeps_nonperson_for_unit_query(monkeypatch):
    """反向：问计量单位时不能被类型分带偏到人物。"""
    cands = [
        {"qid": "Q49196", "label_zh": "牛顿", "description": "美国城市",
         "score": 4.91, "weight": 1.0},
        {"qid": "Q935", "label_zh": "艾薩克·牛頓",
         "description": "英格兰物理学家", "score": 2.47, "weight": 0.6},
    ]
    lk = _make_linker(monkeypatch, cands)
    monkeypatch.setattr(lk, "_is_human", lambda q: q == "Q935")
    got = lk.link("牛顿", query_context="力的单位牛顿等于多少")
    assert got[0]["qid"] == "Q49196"


def test_human_cache_is_lazy(monkeypatch):
    """_human_cache 必须惰性创建。

    Linker 存在绕过 __init__ 的构造路径（测试、反序列化等），
    若依赖 __init__ 里的赋值会直接 AttributeError。
    """
    lk = _make_linker(monkeypatch, _WASHINGTON)
    assert "_human_cache" not in lk.__dict__
    lk._is_human("Q23")
    assert "_human_cache" in lk.__dict__


def test_type_weight_is_configurable():
    """_TYPE_W 应小于 _CTX_W：description 明确命中时词面分仍占主导。"""
    assert 0.0 < lk_mod._TYPE_W < lk_mod._CTX_W
