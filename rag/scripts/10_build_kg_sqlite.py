"""
10_build_kg_sqlite.py
=====================

把 09 步过滤好的 ``entities.jsonl`` + ``triples.tsv`` 灌入 SQLite。
建成的库文件（``kg_db_file``）就是 L5 KG 的最终形态，包含 5 张表 + 1 个 FTS5。

表结构一览
----------

* ``entities``    每个实体一行的主表；查询主键                      # qid, label_zh, label_en, description, article_rank, popularity
* ``mentions``    "mention 字面串 → qid" 一对多映射；含权重和来源    # mention, qid, weight, source
* ``properties``  P<id> 到人类可读名字的字典表                     # pid, label_zh, laben_en: default Null
* ``triples``     核心三元组表（subject → predicate → object）    # id, subject_qid,
* ``mentions_fts`` FTS5 全文索引，加速拼音/前缀/模糊查询

查询典型链路：
    query = "屈原"
    → SELECT qid FROM mentions WHERE mention = ?  → [Q7259]
    → SELECT * FROM triples WHERE subject_qid = ? → [(Q7259, P106, Q49757, ...), ...]
    → JOIN entities 拿到 object 的 label
    → JOIN properties 拿到 predicate 的中文名
    → 拼成 "屈原 - 职业 - 诗人; 屈原 - 国籍 - 楚国; ..." 喂给 LLM

设计要点
--------

1. **建库全速写**：``synchronous=OFF`` + ``journal_mode=WAL`` + 大 cache_size；
   索引留到数据全部灌完最后再建，比"边插边建"快 5-10 倍。
2. **executemany 分批**：一次 5 万行；再多会占太多回滚日志内存。
3. **mentions 从 label / alias 两处产生**：一个 mention 可能指向多个 qid
   （比如 "苹果" → Q89 水果 / Q312 公司），weight 按来源打分（label > alias）。
4. **properties 表可选内置**：keep_predicates 里的 PID 都在这里翻译成中文，
   翻译不到的先留空，需要时可手工回填。

## todo:
1. 工业级应用时，真实线上C++部署落地，一般不用是sqlite3,使用什么工具？
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wiki_rag.config import load_config


# ---------------------------------------------------------------- 内置属性中文表
# Wikidata P<id> → 中文人类可读名。同 default.yaml 里 keep_predicates 对齐。
# 用户可以在库建好后 UPDATE properties SET label_zh=... WHERE pid=... 覆盖。
_PROP_LABELS_ZH = {
    "P31":   "是一个",       "P279":  "子类",         "P17":   "所属国家",
    "P27":   "国籍",         "P19":   "出生地",       "P20":   "逝世地",
    "P569":  "出生日期",     "P570":  "逝世日期",     "P106":  "职业",
    "P39":   "担任职位",     "P26":   "配偶",         "P22":   "父亲",
    "P25":   "母亲",         "P40":   "子女",         "P463":  "所属组织",
    "P108":  "雇主",         "P69":   "就读于",       "P800":  "代表作",
    "P50":   "作者",         "P57":   "导演",         "P58":   "编剧",
    "P86":   "作曲家",       "P175":  "表演者",       "P170":  "创作者",
    "P136":  "类型",         "P495":  "起源国",       "P577":  "发行日期",
    "P36":   "首都",         "P30":   "所属大洲",     "P131":  "行政隶属",
    "P150":  "下辖",         "P276":  "位置",         "P625":  "坐标",
    "P361":  "属于",         "P527":  "包含",         "P138":  "得名自",
    "P571":  "创立日期",     "P576":  "解散日期",     "P159":  "总部所在地",
    "P452":  "行业",         "P112":  "创始人",       "P169":  "CEO",
    "P1830": "拥有",         "P127":  "被拥有者",     "P155":  "前作",
    "P156":  "后续作品",     "P37":   "官方语言",     "P41":   "国旗",
    "P18":   "图像",         "P373":  "Commons分类", "P910":  "主分类",
}


# ---------------------------------------------------------------- schema
_SCHEMA_SQL = """
-- 关闭外键（建库阶段无用且拖慢）
PRAGMA foreign_keys = OFF;

-- 实体主表
CREATE TABLE IF NOT EXISTS entities (
    qid          TEXT PRIMARY KEY,
    label_zh     TEXT,
    label_en     TEXT,
    description  TEXT,
    article_rank INTEGER,          -- 与 L2 pageviews rank join；无对应则 NULL
    popularity   INTEGER DEFAULT 0 -- 后续可回填 pageviews 数值
);

-- mention → qid（多对多，同 mention 可指多个实体）; 从"自然语言字符串"进入知识图谱的入口
CREATE TABLE IF NOT EXISTS mentions (
    mention   TEXT NOT NULL,
    qid       TEXT NOT NULL,
    weight    REAL DEFAULT 1.0,   -- label=1.0 alias=0.6 等
    source    TEXT,               -- "label" | "alias"
    PRIMARY KEY (mention, qid)
);

-- 属性字典
CREATE TABLE IF NOT EXISTS properties (
    pid       TEXT PRIMARY KEY,
    label_zh  TEXT,
    label_en  TEXT
);

-- 三元组
CREATE TABLE IF NOT EXISTS triples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_qid   TEXT NOT NULL,
    predicate_pid TEXT NOT NULL,
    object_qid    TEXT,           -- object 是实体时用；否则 NULL
    object_value  TEXT,           -- object 是字面量时用；否则 NULL
    object_type   TEXT NOT NULL   -- "entity" | "string" | "time" | "quantity"
);

-- FTS5 用来做 mention 的模糊 / 前缀查询
-- unicode61 对中文的默认策略是**"按字符切分"**（每个汉字算一个 token），并不是真正的中文分词（不会切成词组）。后续优化可以考虑使用jieba 或者simple+trigram
CREATE VIRTUAL TABLE IF NOT EXISTS mentions_fts USING fts5(
    mention,                --  被索引的文本列。FTS5 会对它做分词并建倒排索引。
    qid UNINDEXED,          --  UNINDEXED只存储不索引——不需要按 qid 全文搜（qid 是精确值），只需要在命中 mention 后把对应的 qid 拿出来即可。这样能省下大量索引空间。
    tokenize = "unicode61 remove_diacritics 2"  -- 分词器配置， Unicode 6.1 规则识别字符类别，能处理中文、英文、日文、变音符号等多语言；remove_diacritics：去除变音符号
);
"""

# 索引留到最后创建
# 数据库为某一列（或多列）额外维护的一份"排好序的查找结构"，用空间换时间，让 WHERE 该列 = ? 从"扫全表"变成"直接跳到答案"。
# 从全表扫描 -> B-Tree 查找; 耗时从 O(N)->  O(log N); 从1-2s到0.1-0.5 毫秒
_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_entities_label_zh ON entities(label_zh);
CREATE INDEX IF NOT EXISTS idx_entities_rank     ON entities(article_rank);
CREATE INDEX IF NOT EXISTS idx_mentions_mention  ON mentions(mention);
CREATE INDEX IF NOT EXISTS idx_triples_subj      ON triples(subject_qid);
CREATE INDEX IF NOT EXISTS idx_triples_obj       ON triples(object_qid);
CREATE INDEX IF NOT EXISTS idx_triples_pred      ON triples(predicate_pid);
"""


def _apply_pragmas(conn: sqlite3.Connection, pragmas: dict) -> None:
    """把 config 里的 pragma dict 逐条 apply 到连接上。"""
    for k, v in pragmas.items():
        # 数值型 pragma 不能加引号
        conn.execute(f"PRAGMA {k}={v}")


def _load_top_titles(top_titles_file: Path) -> dict[str, int]:
    """读 04 步产出的 top_titles.txt，返回 {title_simp: rank}。
    rank 从 0 开始（越小越热）。"""
    m: dict[str, int] = {}
    if not top_titles_file.exists():
        print(f"[10] top_titles not found: {top_titles_file}  (skip article_rank join)")
        return m
    with open(top_titles_file, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            t = ln.strip()
            if t:
                m[t] = i
    print(f"[10] loaded {len(m):,} top titles for article_rank join")
    return m


def _iter_entities(entities_jsonl: Path):
    """流式产出 09 步的 entities.jsonl，每条 yield 一个 dict。"""
    with open(entities_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)    # str -> json


def _iter_triples(triples_tsv: Path):
    """流式产出 09 步的 triples.tsv。每行 5 列。"""
    with open(triples_tsv, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            s, p, o_qid, o_val, o_type = parts
            yield s, p, (o_qid if o_qid else None), (o_val if o_val else None), o_type


def build_db(cfg: dict) -> None:
    paths = cfg["paths"]
    kg_cfg = cfg["kg"]

    db_path: Path = paths["kg_db_file"]                 # "data/wikidata_zh_kg.db"
    entities_jsonl: Path = paths["kg_entities_jsonl"]   # "data/wikidata_zh_entities.jsonl"
    triples_tsv: Path = paths["kg_triples_tsv"]         # "data/wikidata_zh_triples.tsv"
    top_titles: Path = paths["top_titles_file"]         # "data/wiki_zh_top_titles.txt"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    # 干净重建：删掉旧的，避免脏数据混进来
    if db_path.exists():
        print(f"[10] removing old db {db_path}")
        db_path.unlink()

    print(f"[10] opening sqlite {db_path} ...")
    conn = sqlite3.connect(str(db_path))            # 创建一个sqlite3链接到一个本地文件
    _apply_pragmas(conn, kg_cfg["sqlite_pragmas"])      # 把 config 里的 pragma dict 逐条 apply 到连接上。
    conn.executescript(_SCHEMA_SQL)
    conn.commit()

    batch = kg_cfg["insert_batch"]      # 50000，一次 executemany 的 batch 大小

    # ---- 1) properties 表：keep_predicates 里的 PID 都写进来 ----
    print("[10] filling properties ...")
    prop_rows = [(pid, _PROP_LABELS_ZH.get(pid), None) for pid in kg_cfg["keep_predicates"]]
    conn.executemany(
        "INSERT OR REPLACE INTO properties(pid, label_zh, label_en) VALUES (?,?,?)",
        prop_rows,
    )
    conn.commit()

    # ---- 2) entities 表 + mentions（label + alias）----
    print("[10] loading entities.jsonl and inserting entities + mentions ...")
    title_to_rank = _load_top_titles(top_titles_file=top_titles)    # {title, rank_id}

    ent_buf: list[tuple] = []
    men_buf: list[tuple] = []
    fts_buf: list[tuple] = []
    n_ent = 0
    n_men = 0

    def _flush_ent():
        nonlocal ent_buf
        if not ent_buf:
            return
        conn.executemany(
            "INSERT OR REPLACE INTO entities"
            "(qid, label_zh, label_en, description, article_rank, popularity)"
            " VALUES (?,?,?,?,?,?)",
            ent_buf,
        )
        ent_buf = []

    def _flush_men():
        nonlocal men_buf, fts_buf
        if not men_buf:
            return
        conn.executemany(
            "INSERT OR IGNORE INTO mentions(mention, qid, weight, source) VALUES (?,?,?,?)",
            men_buf,
        )
        # FTS5 是 append-only 的，每条也进 FTS 表
        conn.executemany(
            "INSERT INTO mentions_fts(mention, qid) VALUES (?,?)",
            fts_buf,
        )
        men_buf = []
        fts_buf = []

    for e in tqdm(_iter_entities(entities_jsonl), desc="entities"): # {qid, label_zh, label_en, description, aliases}
        qid = e["qid"]
        label_zh = e.get("label_zh")
        label_en = e.get("label_en")
        desc = e.get("description")
        aliases = e.get("aliases") or []
        # article_rank：只有主 label 能 join 上 top_titles 时才有值
        rank = title_to_rank.get(label_zh) if label_zh else None    # {title, rank_id} -> rank_id

        ent_buf.append((qid, label_zh, label_en, desc, rank, 0))
        n_ent += 1

        # label 本身作为一个 mention，权重 1.0
        if label_zh:
            men_buf.append((label_zh, qid, 1.0, "label"))
            fts_buf.append((label_zh, qid))
            n_men += 1
        # label_en 也进一份，方便英文 mention 命中
        if label_en:
            men_buf.append((label_en, qid, 0.8, "label"))
            fts_buf.append((label_en, qid))
            n_men += 1
        # alias 权重降到 0.6，避免过多 alias 冲淡主 label 的排序
        for a in aliases:
            if not a:
                continue
            men_buf.append((a, qid, 0.6, "alias"))
            fts_buf.append((a, qid))
            n_men += 1

        if len(ent_buf) >= batch:
            _flush_ent()
        if len(men_buf) >= batch:
            _flush_men()

    _flush_ent()
    _flush_men()
    conn.commit()
    print(f"[10]   entities inserted: {n_ent:,}")
    print(f"[10]   mentions inserted: {n_men:,}")

    # ---- 3) triples 表 ----
    print("[10] loading triples.tsv and inserting ...")
    tri_buf: list[tuple] = []
    n_tri = 0

    def _flush_tri():
        nonlocal tri_buf
        if not tri_buf:
            return
        conn.executemany(
            "INSERT INTO triples(subject_qid, predicate_pid, object_qid, object_value, object_type)"
            " VALUES (?,?,?,?,?)",
            tri_buf,
        )
        tri_buf = []

    # subj_qid, pid, object_qid, object_value, object_type
    for s, p, o_qid, o_val, o_type in tqdm(_iter_triples(triples_tsv), desc="triples"):
        tri_buf.append((s, p, o_qid, o_val, o_type))
        n_tri += 1
        if len(tri_buf) >= batch:
            _flush_tri()
    _flush_tri()
    conn.commit()
    print(f"[10]   triples inserted: {n_tri:,}")

    # ---- 4) 最后建索引 ----
    print("[10] creating indices ...")
    conn.executescript(_INDEX_SQL)
    conn.commit()

    # ---- 5) 落盘 & 收尾 ----
    print("[10] running ANALYZE (for planner) ...")
    # 在数据全部灌完 + 索引全部建好之后，让 SQLite 去"了解一下"数据分布，把统计信息记到内部表里
    # 这样以后每次执行查询时，SQL 优化器就能选出最快的执行计划。
    conn.execute("ANALYZE")
    # 一个综合优化命令，会自动决定该做什么维护工作。主要做两件事：
    # 1. 对"上次 ANALYZE 之后有大量变化"的表/索引，触发一次增量 ANALYZE
    # 2. 有些 SQLite 版本还会做点小规模的重建索引 / 优化 B-Tree 布局
    conn.execute("PRAGMA optimize")
    conn.close()
    print(f"[10] done. db saved -> {db_path}")          # "data/wikidata_zh_kg.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    build_db(cfg)


if __name__ == "__main__":
    main()
