# test_kg_mentions.py
"""L5 mention 抽取：泛实体（类实体）识别与重排的回归测试。

════════════════════════════════════════════════════════════════════════
本文件覆盖什么
════════════════════════════════════════════════════════════════════════
    is_generic_mention          —— 基于 P31/P279 入度判定"类实体"
    rerank_mentions_by_specificity —— 真实体前置（解决预算挤占）
    extract_mentions 集成       —— rerank / drop 开关行为
    健壮性                      —— 表缺失、空输入、缓存正确性

════════════════════════════════════════════════════════════════════════
为什么要有这些测试（一个真实的、收益归零的故障）
════════════════════════════════════════════════════════════════════════
L5 KG 层在 BrowseComp-ZH 这类"混淆式多跳"题上贡献长期为 **0/60**。
实测那道题：
    "哪个地方拥有AAAAA级景区、被称为成熟周期很长的水果之乡，并且有一位
     科学家曾在欧洲知名大学学习后回国奠定了一个学科基础？"
hybrid 抽出 7 个 mention，只有 '欧洲' 是真实体：
    ['AAA', '科学家', '地方', '景区', '水果', '欧洲', '学科']

而 `query_kg_end_to_end` 是**按 mention 顺序**消耗 `max_entities`(5) 预算的：
    AAA(+2) → 科学家(+2) → 地方(+1) → 预算耗尽，硬 break
于是 ['景区','水果','欧洲','学科'] 根本没机会被链接 ——
**唯一的真实体被泛实体挤掉了**。这不是"排序没排对"，是它没进候选。

修复后实际链接变成 ['AAA','欧洲','学科']，'欧洲' 成功进入预算。

设计原则：**不依赖那份 10.9GB 的真实 KG**。这里用内存 SQLite 造一份
微型 KG（几十行），把判定逻辑所依赖的 P31/P279 入度关系显式构造出来，
这样测试在 CI 里毫秒级完成且结果完全确定。
运行：
    pytest tests/test_kg_mentions.py -v
"""
from __future__ import annotations

import sqlite3

import pytest

from src.rag.wiki_rag import kg_retriever as kr


# =============================================================================
# 微型 KG 夹具
# =============================================================================

class _FakeKG:
    """只提供 `.conn` 的最小 KGStore 替身。

    `is_generic_mention` 只用到 `kg.conn.execute`，不碰 KGStore 的其他成员，
    所以这里不需要（也不应该）去实例化真正的 KGStore —— 那会加载 10.9GB 库。
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn


def _build_kg(with_triples: bool = True) -> _FakeKG:
    """构造一份微型 KG。

    实体设计（对应真实 Wikidata 的结构特征）：
      Q515  城市    —— 类：大量实体 instance-of 它（P31 入度高）
      Q901  科学家  —— 类：有子类 subclass-of 它（P279 入度高）
      Q937  爱因斯坦 —— 实例：入度 0
      Q215675 量子纠缠 —— 概念实例：**自身有 P279**（是"量子力学概念"的子类）
                          但没有入度。这条专门用来防回归：早期实现把
                          "自身有 P279" 当类信号，导致量子纠缠/相对论被误杀。
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE mentions (
            mention TEXT NOT NULL, qid TEXT NOT NULL,
            weight REAL DEFAULT 1.0, source TEXT,
            PRIMARY KEY (mention, qid)
        );
        CREATE TABLE entities (
            qid TEXT PRIMARY KEY, label_zh TEXT, label_en TEXT,
            description TEXT, article_rank INTEGER, popularity INTEGER DEFAULT 0
        );
        """
    )
    if with_triples:
        conn.executescript(
            """
            CREATE TABLE triples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_qid TEXT NOT NULL, predicate_pid TEXT NOT NULL,
                object_qid TEXT, object_value TEXT, object_type TEXT NOT NULL
            );
            CREATE INDEX idx_triples_obj ON triples(object_qid);
            """
        )

    ents = [
        ("Q515", "城市", 100), ("Q901", "科学家", 90),
        ("Q937", "爱因斯坦", 80), ("Q215675", "量子纠缠", 10),
        ("Q46", "欧洲", 95), ("Q3257686", "地方", 50),
        # "大学" 的 top-1 按 popularity 是一本"著作"（真实库里确实如此），
        # 真正的类实体 Q3918 排在后面 —— 用来验证必须探测多个候选。
        ("Q1069886", "著作", 60), ("Q3918", "大学", 5),
    ]
    conn.executemany(
        "INSERT INTO entities(qid,label_zh,popularity) VALUES(?,?,?)", ents)
    conn.executemany(
        "INSERT INTO mentions(mention,qid,weight,source) VALUES(?,?,1.0,'label')",
        [("城市", "Q515"), ("科学家", "Q901"), ("爱因斯坦", "Q937"),
         ("量子纠缠", "Q215675"), ("欧洲", "Q46"), ("地方", "Q3257686"),
         ("大学", "Q1069886"), ("大学", "Q3918")],
    )

    if with_triples:
        rows = []
        # Q515「城市」：20 个实体 instance-of 它 → 远超 _CLASS_IN31_TH
        for i in range(20):
            rows.append((f"Q90{i:03d}", "P31", "Q515", None, "entity"))
        # Q901「科学家」：10 个子类 subclass-of 它 → 超 _CLASS_IN279_TH
        for i in range(10):
            rows.append((f"Q91{i:03d}", "P279", "Q901", None, "entity"))
        # Q3918「大学」：也是类（但它不是 "大学" 的 top-1 候选）
        for i in range(12):
            rows.append((f"Q92{i:03d}", "P31", "Q3918", None, "entity"))
        # Q3257686「地方」：类
        for i in range(15):
            rows.append((f"Q93{i:03d}", "P31", "Q3257686", None, "entity"))
        # 量子纠缠：自身 subclass-of 某概念，但**无入度** → 必须判为非泛实体
        rows.append(("Q215675", "P279", "Q944", None, "entity"))
        # 爱因斯坦 / 欧洲：只有出边，无入度
        rows.append(("Q937", "P31", "Q5", None, "entity"))
        rows.append(("Q46", "P31", "Q5107", None, "entity"))
        conn.executemany(
            "INSERT INTO triples(subject_qid,predicate_pid,object_qid,"
            "object_value,object_type) VALUES(?,?,?,?,?)", rows)
    conn.commit()
    return _FakeKG(conn)


@pytest.fixture(autouse=True)
def _clear_generic_cache():
    """清空判定缓存。

    `_GENERIC_CACHE` 是**模块级**字典，按 mention 字符串缓存。不同测试用
    不同的微型 KG，但 mention 名字相同（"城市"/"大学"），若不清空，
    后一个测试会命中前一个测试的判定结果 —— 又是一个"顺序依赖"的坑
    （conftest.py 里记录过同类问题）。
    """
    kr._GENERIC_CACHE.clear()
    yield
    kr._GENERIC_CACHE.clear()


@pytest.fixture
def kg():
    return _build_kg()


# =============================================================================
# is_generic_mention
# =============================================================================

@pytest.mark.parametrize("mention", ["城市", "科学家", "地方"])
def test_generic_mentions_detected(kg, mention):
    """泛实体（被大量实体当类使用）应被识别。"""
    assert kr.is_generic_mention(mention, kg) is True


@pytest.mark.parametrize("mention", ["爱因斯坦", "欧洲"])
def test_specific_entities_not_flagged(kg, mention):
    """具体实例不应被判为泛实体。"""
    assert kr.is_generic_mention(mention, kg) is False


def test_concept_with_own_p279_not_flagged(kg):
    """【防回归】自身有 P279 的概念实体不能被误杀。

    早期实现把"自身有 P279"当作类信号，导致 量子纠缠、相对论 被判为泛实体
    —— 因为概念天然是某个上位概念的子类。判定必须**只看入度**。
    """
    assert kr.is_generic_mention("量子纠缠", kg) is False


def test_probes_multiple_candidates(kg):
    """【防回归】不能只看 popularity 最高的那个候选。

    "大学" 的 top-1 候选是 Q1069886「著作」（入度 0）；真正的类 Q3918
    排在后面。只探 top-1 会漏判，必须探测多个候选。
    """
    assert kr.is_generic_mention("大学", kg) is True


def test_unknown_mention_is_not_generic(kg):
    """KG 里查不到的 mention 视为非泛实体（不应误伤）。"""
    assert kr.is_generic_mention("不存在的实体xyz", kg) is False


def test_missing_triples_table_degrades_gracefully():
    """triples 表缺失时不能抛异常，应退化为"全部非泛实体"。

    L5 的异常会被 `_safe_search` 吞掉导致整层静默返回空（kg_store.py 里
    记录过这类"降级成功但收益归零"的故障）。所以这里必须容错而非抛错。
    """
    kg_no_triples = _build_kg(with_triples=False)
    assert kr.is_generic_mention("城市", kg_no_triples) is False


def test_cache_avoids_repeat_queries(kg):
    """判定结果应被缓存（泛实体高频复现，未缓存时实测单 query 660ms）。"""
    assert kr.is_generic_mention("城市", kg) is True
    assert "城市" in kr._GENERIC_CACHE

    # 把连接换成会爆的替身：若仍走查询就会抛错，命中缓存则安然返回
    class _Boom:
        def execute(self, *a, **k):
            raise AssertionError("应命中缓存，不该再查库")

    assert kr.is_generic_mention("城市", _FakeKG(_Boom())) is True


# =============================================================================
# rerank_mentions_by_specificity
# =============================================================================

def test_rerank_moves_specific_entities_first(kg):
    """真实体前置 —— 这正是修复预算挤占的关键。"""
    got = kr.rerank_mentions_by_specificity(
        ["科学家", "地方", "欧洲", "城市", "爱因斯坦"], kg)
    assert got[:2] == ["欧洲", "爱因斯坦"], f"真实体未前置: {got}"
    assert set(got[2:]) == {"科学家", "地方", "城市"}


def test_rerank_is_stable_within_groups(kg):
    """组内保持原有相对顺序（稳定排序），避免引入随机性。"""
    got = kr.rerank_mentions_by_specificity(
        ["城市", "欧洲", "科学家", "爱因斯坦", "地方"], kg)
    assert got == ["欧洲", "爱因斯坦", "城市", "科学家", "地方"]


def test_rerank_preserves_all_mentions_by_default(kg):
    """默认只重排、不丢弃 —— 判别器误判时最坏也只是顺序变化。"""
    src = ["城市", "欧洲", "科学家"]
    assert sorted(kr.rerank_mentions_by_specificity(src, kg)) == sorted(src)


def test_drop_generic_removes_generic(kg):
    got = kr.rerank_mentions_by_specificity(
        ["科学家", "欧洲", "城市"], kg, drop_generic=True)
    assert got == ["欧洲"]


def test_drop_generic_falls_back_when_all_generic(kg):
    """全是泛实体时必须回退到原列表。

    宁可带噪声检索，也不能让 mention 一个不剩 —— 那会让 L5 直接返回空。
    典型场景："城市人口最多的国家" 整句都是类词。
    """
    src = ["城市", "科学家", "地方"]
    assert kr.rerank_mentions_by_specificity(
        src, kg, drop_generic=True) == src


def test_rerank_empty_input(kg):
    assert kr.rerank_mentions_by_specificity([], kg) == []


# =============================================================================
# extract_mentions 集成
# =============================================================================

def test_extract_mentions_applies_rerank(monkeypatch, kg):
    """extract_mentions 默认应启用重排。"""
    monkeypatch.setattr(
        kr, "_extract_mentions_hybrid",
        lambda q, k: ["科学家", "地方", "欧洲"])
    got = kr.extract_mentions("任意", kg, method="hybrid")
    assert got[0] == "欧洲", f"未启用重排: {got}"


def test_extract_mentions_rerank_can_be_disabled(monkeypatch, kg):
    """开关必须真的能关掉 —— 便于 A/B 对比与出问题时回退。"""
    raw = ["科学家", "地方", "欧洲"]
    monkeypatch.setattr(kr, "_extract_mentions_hybrid", lambda q, k: list(raw))
    got = kr.extract_mentions(
        "任意", kg, method="hybrid", rerank_specificity=False)
    assert got == raw


def test_extract_mentions_drop_generic(monkeypatch, kg):
    monkeypatch.setattr(
        kr, "_extract_mentions_hybrid",
        lambda q, k: ["科学家", "地方", "欧洲"])
    got = kr.extract_mentions(
        "任意", kg, method="hybrid", drop_generic=True)
    assert got == ["欧洲"]


def test_extract_mentions_empty_result(monkeypatch, kg):
    """抽取结果为空时不应因重排而报错。"""
    monkeypatch.setattr(kr, "_extract_mentions_hybrid", lambda q, k: [])
    assert kr.extract_mentions("任意", kg, method="hybrid") == []


def test_budget_crowding_scenario(monkeypatch, kg):
    """端到端复现那道 BCZ 题的预算挤占，并验证修复。

    模拟 `query_kg_end_to_end` 的预算逻辑：按序消耗 max_entities=5，
    每个 mention 最多 top_k=2。
    """
    raw = ["科学家", "地方", "城市", "欧洲", "爱因斯坦"]
    monkeypatch.setattr(kr, "_extract_mentions_hybrid", lambda q, k: list(raw))

    def linked(mentions, top_k=2, max_entities=5):
        used, out = 0, []
        for m in mentions:
            if used >= max_entities:
                break
            used += min(top_k, max_entities - used)
            out.append(m)
        return out

    before = linked(kr.extract_mentions(
        "q", kg, method="hybrid", rerank_specificity=False))
    after = linked(kr.extract_mentions("q", kg, method="hybrid"))

    # 修复前：真实体被泛实体挤出预算
    assert "欧洲" not in before and "爱因斯坦" not in before
    # 修复后：真实体进入预算
    assert "欧洲" in after and "爱因斯坦" in after
