"""
13_fix_kg_unicode_escape.py
===========================

修复 `wikidata_zh_kg.db` 里**中文被存成 `\\uXXXX` 字面转义串**的数据污染。

════════════════════════════════════════════════════════════════════════
问题现象
════════════════════════════════════════════════════════════════════════
KG 库里的中文不是真中文，而是字面量的反斜杠转义序列：

    sqlite> select mention from mentions where mention like '%爱因%';
    (空)
    sqlite> select mention,qid from mentions limit 1 offset 12345;
    ('1979\\u5E74\\u7EAA\\u5FF5\\u7231\\u56E0\\u65AF\\u5766...', 'Q133121725')
     ↑ 这不是中文，是 '\\','u','5','E','7','4',... 这些 ASCII 字符

后果是 L5 知识图谱层**几乎完全失效**：

    extract_mentions('爱因斯坦的出生地')  →  []
    extract_mentions('特朗普的配偶')      →  []

mention 抽取靠的是拿 query 的 n-gram / jieba 切词去 `mentions` 表精确
查表，表里存的既然是转义串，用户输入的真中文自然一条都对不上。偶尔命中的
（比如 "CEO" 这种纯 ASCII）拿回来的 text 又是一串 `\\u7EF4\\u57FA...`
喂给 LLM —— 等于噪声。

实测污染面（本机 10G 库，修复前）：

    entities.label_zh      10,042,460 / 10,385,628  =  96.7%
    entities.description    4,581,495 / 10,385,628  =  44.1%
    entities.label_en         659,451 / 10,385,628  =   6.3%
    mentions.mention       11,807,915 / 16,846,443  =  70.1%
    triples.object_value      113,822 / 31,286,253  =   0.4%

L5 每次检索都被激活、付 ~434ms，命中率却接近 0 —— 纯亏损。

════════════════════════════════════════════════════════════════════════
根因
════════════════════════════════════════════════════════════════════════
Wikidata 的 N-Triples dump 里，非 ASCII 字符本身就是以 `\\uXXXX` 转义的
（这是 N-Triples 规范 RDF 1.1 §7 允许的写法）：

    <.../Q148> <...#label> "\\u4E2D\\u534E\\u4EBA\\u6C11\\u5171\\u548C\\u56FD"@zh .

而 `09_filter_wikidata_zh.py` 的 `_parse_object()` / label 分支只用正则
把引号里的内容原样切出来（`m.group(1)`），**没有做 N-Triples 反转义**：

    _LITERAL_LANG_RE = re.compile(r'^"(.*)"@([a-zA-Z\\-]+)$')
    val = m.group(1)        # ← 拿到的是 '\\u4E2D\\u534E...' 这个字面串

于是转义串一路原样流进 JSONL → TSV → SQLite。
注意 09 脚本写 JSONL 时用的是 `ensure_ascii=False`（写法本身没问题），
所以**不是 json 序列化的锅**，而是上游少了一步 unescape。

源头已在 `09_filter_wikidata_zh.py` 里修好（新增 `_nt_unescape()`，
接在 label / alias / description / `_parse_object` 四个字面量解析点上），
所以下次从 dump 重建不会复发。本脚本负责修**已经建好的库**。

════════════════════════════════════════════════════════════════════════
修复方案
════════════════════════════════════════════════════════════════════════
就地反转义（in-place UPDATE），不重建库 —— 重建要重新解压 250GB dump，
2~4 小时；就地修 2700 万个字段值约 40 分钟。

    '\\u4E2D\\u534E'  --unescape-->  '中华'

三个必须小心的点：

1) **主键冲突**
   `mentions` 表有 `PRIMARY KEY (mention, qid)`。解码后的 mention 完全
   可能与表中已存在的干净行撞主键（比如 alias 里既有转义版又有原文版）。
   直接 UPDATE 会抛 IntegrityError 中断整个事务。
   → 用 `UPDATE OR REPLACE`：冲突时删掉旧行、保留新行，语义正确
     （两行本来就该是同一条），且不会中断。

2) **FTS5 必须整表重建**
   `mentions_fts` 是独立的 FTS5 表，内容是构建时一条条 INSERT 进去的
   转义文本。UPDATE 主表**不会**同步它（本库没建触发器）。
   → 修完主表后 DELETE 全表 + 从 mentions 重灌。

3) **热门实体向量会失配**
   `wikidata_zh_kg_hot_emb.npy` 是用**转义后的 label** 过 BGE-M3 编出来的，
   语义完全错误。修完 label 必须重跑 `11_encode_hot_entities.py`。
   本脚本只提示，不自动执行（那一步要加载模型）。

════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════
    # 1. 干跑：只统计 + 抽样展示解码效果，不改任何数据（默认）
    python rag/scripts/13_fix_kg_unicode_escape.py

    # 2. 备份后实跑
    cp data/rag_data/wikidata_zh_kg.db data/rag_data/wikidata_zh_kg.db.bak
    python rag/scripts/13_fix_kg_unicode_escape.py --apply

    # 3. 重建热门实体向量（修完 label 后必做）
    #    ⚠️ BGE-M3 在 MPS 上加载可能 segfault，用 CPU 更稳（2871 条只要 5s）
    RAG_EMBED_DEVICE=cpu python rag/scripts/11_encode_hot_entities.py \\
        --config rag/configs/default.yaml

可选参数：
    --db PATH        指定库路径（默认 data/rag_data/wikidata_zh_kg.db）
    --batch N        每批提交行数（默认 20000）
    --skip-fts       跳过 FTS 重建（调试用，正常别加）
    --tables a,b     只修指定表（entities/mentions/triples）
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- 反转义

# 匹配 N-Triples 的转义序列：\uXXXX（BMP）与 \UXXXXXXXX（增补平面）
# 之所以自己写而不用 codecs.decode(s, 'unicode_escape')：
#   * unicode_escape 走的是 latin-1 语义，字符串里若混有**真**中文
#     （dump 里确实有一部分行是不转义的）会被逐字节拆坏，
#     产生 'ä¸­æ' 这类 mojibake；
#   * 它还会把 '\n' '\t' '\\' 也一并解释掉，而我们只想处理 \u/\U，
#     其余反斜杠应原样保留（实体名里出现 '\' 是合法的）。
# 用正则只替换 \uXXXX，对混合内容安全、幂等。
_ESCAPE_RE = re.compile(r"\\U([0-9A-Fa-f]{8})|\\u([0-9A-Fa-f]{4})")

# 快速预筛：用 GLOB 让 SQLite 走字符类匹配，比 LIKE '%\u%' 精确得多
# （后者会把正常含 "\u" 文本的行也捞出来，虽然无害但会多扫很多行）
_GLOB_ESCAPED = r"*\u[0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f]*"


def unescape(s: str | None) -> str | None:
    """把 '\\u4E2D\\u534E' 还原成 '中华'。

    幂等：已经是真中文的字符串原样返回（正则匹配不到）。
    非法码点（如落在代理区的孤立 surrogate）保持原样，不抛异常 ——
    宁可留一条脏数据，也不能让整批 2000 万行的修复中断。
    """
    if not s or "\\" not in s:
        return s

    def _sub(m: re.Match) -> str:
        hexs = m.group(1) or m.group(2)
        try:
            cp = int(hexs, 16)
            # 代理区 D800-DFFF 单独出现是非法的，chr() 不报错但后续
            # 写入 SQLite 时会炸（sqlite3 要求合法 UTF-8）→ 保持原样
            if 0xD800 <= cp <= 0xDFFF:
                return m.group(0)
            return chr(cp)
        except (ValueError, OverflowError):
            return m.group(0)

    return _ESCAPE_RE.sub(_sub, s)


# ---------------------------------------------------------------- 表配置

# (表名, 主键列, [需要修复的文本列])
# mentions 是复合主键 (mention, qid)，没法用单列定位 → 走 rowid。
TABLES: dict[str, tuple[str, list[str]]] = {
    "entities": ("qid", ["label_zh", "label_en", "description"]),
    "mentions": ("__rowid__", ["mention"]),
    "triples":  ("id", ["object_value"]),
}


def count_dirty(con: sqlite3.Connection, table: str, cols: list[str]) -> dict[str, int]:
    """统计每列有多少行含转义序列。"""
    out: dict[str, int] = {}
    for col in cols:
        out[col] = con.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {col} GLOB ?", (_GLOB_ESCAPED,)
        ).fetchone()[0]
    return out


def sample_preview(con: sqlite3.Connection, table: str, col: str, limit: int = 4) -> None:
    """抽样打印「修复前 → 修复后」，供人眼确认解码正确。"""
    rows = con.execute(
        f"SELECT {col} FROM {table} WHERE {col} GLOB ? LIMIT ?",
        (_GLOB_ESCAPED, limit),
    ).fetchall()
    for (raw,) in rows:
        print(f"      {raw[:52]!r}")
        print(f"   →  {unescape(raw)[:52]!r}")


def fix_table(
    con: sqlite3.Connection,
    table: str,
    pk: str,
    cols: list[str],
    batch: int,
    apply: bool,
) -> int:
    """就地修复一张表，返回实际更新的字段数。

    分批读 → 批量 UPDATE OR REPLACE → 提交。不用一条大 UPDATE 的原因：
      * 2000 万行的单事务会让 WAL 膨胀到数 GB，中途断电全部回滚；
      * 分批能打进度，长任务里这一点很重要。
    """
    pk_expr = "rowid" if pk == "__rowid__" else pk
    total = 0
    t0 = time.perf_counter()

    for col in cols:
        n_dirty = con.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {col} GLOB ?", (_GLOB_ESCAPED,)
        ).fetchone()[0]
        if n_dirty == 0:
            print(f"  [{table}.{col}] 无污染，跳过")
            continue
        print(f"  [{table}.{col}] 待修 {n_dirty:,} 行 ...")

        done = 0
        while True:
            # 每轮重新查「仍含转义」的一批。因为 UPDATE 后这些行不再匹配
            # GLOB，天然形成游标推进，不需要维护 offset（offset 在大表上
            # 是 O(n) 扫描，会越走越慢）。
            rows = con.execute(
                f"SELECT {pk_expr}, {col} FROM {table} WHERE {col} GLOB ? LIMIT ?",
                (_GLOB_ESCAPED, batch),
            ).fetchall()
            if not rows:
                break

            payload = [(unescape(raw), key) for key, raw in rows
                       if unescape(raw) != raw]

            if not payload:
                # 理论上不该发生（GLOB 命中但解码无变化）。真发生了说明
                # 存在无法解码的畸形序列 —— 必须 break，否则死循环。
                print(f"    ⚠️ {len(rows)} 行匹配但无法解码（畸形转义），跳过")
                break

            if apply:
                # OR REPLACE：解码后若与已有行撞主键，删旧留新。
                # 对 mentions(mention,qid) 这种复合主键尤其必要。
                con.executemany(
                    f"UPDATE OR REPLACE {table} SET {col}=? WHERE {pk_expr}=?",
                    payload,
                )
                con.commit()
            done += len(payload)
            total += len(payload)
            print(f"    ... {done:,}/{n_dirty:,}  ({time.perf_counter()-t0:.0f}s)",
                  end="\r", flush=True)

            if not apply:
                print(f"    [干跑] 首批 {len(payload):,} 行可解码，示例：")
                sample_preview(con, table, col, limit=3)
                break
        print()
    return total


def rebuild_fts(con: sqlite3.Connection, apply: bool) -> None:
    """重建 mentions_fts。

    FTS5 表不会随主表 UPDATE 自动同步（本库没建触发器），修完主表后
    索引里仍是转义文本，`_extract_mentions_hybrid` 的模糊分支照样打不中。
    """
    if not con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name='mentions_fts'"
    ).fetchone()[0]:
        print("  未找到 mentions_fts，跳过")
        return

    n = con.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]
    print(f"  重建 mentions_fts（{n:,} 行）...")
    if not apply:
        print("  [干跑] 跳过实际重建")
        return

    t0 = time.perf_counter()
    con.execute("DELETE FROM mentions_fts")
    con.commit()
    con.execute(
        "INSERT INTO mentions_fts(mention, qid) SELECT mention, qid FROM mentions"
    )
    con.commit()
    print(f"  ✓ FTS 重建完成 ({time.perf_counter()-t0:.0f}s)")


def verify(con: sqlite3.Connection) -> None:
    """修完抽查几个教科书级 KG 实体能否精确命中。"""
    print("\n── 验证：常见实体精确查表 ──")
    # 注意「中华人民共和国」预期为 0：Wikidata 的 zh label 混着简繁，
    # Q148 存的是繁体「中華人民共和國」。这是另一个问题（繁简不统一），
    # 不属于本脚本范围。
    for p in ["爱因斯坦", "阿尔伯特·爱因斯坦", "特朗普", "苹果公司",
              "北京", "中華人民共和國"]:
        n = con.execute(
            "SELECT COUNT(*) FROM mentions WHERE mention=?", (p,)
        ).fetchone()[0]
        print(f"  {'✓' if n else '✗'} {p:<20s} → {n} 条")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="KG sqlite 路径")
    ap.add_argument("--apply", action="store_true",
                    help="真正写库（不加则只干跑统计）")
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--skip-fts", action="store_true")
    ap.add_argument("--tables", default="entities,mentions,triples")
    args = ap.parse_args()

    if args.db:
        db_path = Path(args.db)
    else:
        # 本仓库布局是 src/rag/scripts/13_*.py，data/ 在**仓库根**下，
        # 即需要向上四级（scripts → rag → src → 仓库根）。
        # 原脚本写的是 parent.parent.parent（三级），那是 rag/scripts/ 的
        # 布局，在这里会算到 src/data/… 而找不到库。
        root = Path(__file__).resolve().parents[3]
        db_path = root / "data" / "rag_data" / "wikidata_zh_kg.db"
    if not db_path.exists():
        sys.exit(f"找不到 KG 库：{db_path}")

    size_gb = db_path.stat().st_size / 1024**3
    print(f"[13] {'🔧 APPLY（会写库）' if args.apply else '👀 DRY-RUN（不写库）'}")
    print(f"[13] db = {db_path}  ({size_gb:.1f} GB)\n")

    if args.apply:
        bak = db_path.with_suffix(db_path.suffix + ".bak")
        if not bak.exists():
            print(f"⚠️  未检测到备份 {bak.name}")
            print(f"   强烈建议先执行：cp {db_path} {bak}")
            if input("   仍要继续？(yes/N) ").strip().lower() != "yes":
                sys.exit("已取消")

    con = sqlite3.connect(str(db_path))
    # 大批量写入的常规调优：WAL + 放宽 fsync + 大 page cache。
    # synchronous=NORMAL 在 WAL 下仍然崩溃安全（只可能丢最后几个事务），
    # 而我们有备份，这个取舍完全划算 —— 实测能快 3~5 倍。
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-200000")     # ~200MB
    con.execute("PRAGMA temp_store=MEMORY")

    want = [t.strip() for t in args.tables.split(",") if t.strip()]

    print("── 污染统计（修复前）──")
    for table in want:
        _, cols = TABLES[table]
        tot = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for col, n in count_dirty(con, table, cols).items():
            pct = f"{n/tot:.1%}" if tot else "-"
            print(f"  {table+'.'+col:<26s} {n:>12,} / {tot:>12,}  = {pct}")

    if not args.apply:
        print("\n── 解码抽样预览 ──")
        sample_preview(con, "entities", "label_zh", limit=4)

    print("\n── 修复 ──")
    grand = 0
    for table in want:
        pk, cols = TABLES[table]
        grand += fix_table(con, table, pk, cols, args.batch, args.apply)

    if not args.skip_fts and "mentions" in want:
        print("── FTS5 索引 ──")
        rebuild_fts(con, args.apply)

    if args.apply:
        print("\n── 污染统计（修复后）──")
        for table in want:
            _, cols = TABLES[table]
            for col, n in count_dirty(con, table, cols).items():
                print(f"  {'✓' if n == 0 else '⚠️'} {table+'.'+col:<26s} 残留 {n:,}")
        verify(con)
        print("\n── ANALYZE（刷新查询计划统计）──")
        con.execute("ANALYZE")
        con.commit()

    con.close()

    print(f"\n[13] 完成，共更新 {grand:,} 个字段值")
    if args.apply:
        print("\n⚠️  下一步（必做）：热门实体向量是用**转义前**的 label 编的，")
        print("    语义完全错误，必须重新编码：")
        print("        RAG_EMBED_DEVICE=cpu python rag/scripts/11_encode_hot_entities.py \\")
        print("            --config rag/configs/default.yaml")
        print("    然后重启 main_web.py（向量是进程内单例，不重启读不到新文件）。")
    else:
        print("\n干跑结束。确认无误后执行：")
        print(f"    cp {db_path} {db_path}.bak")
        print(f"    python rag/scripts/{Path(__file__).name} --apply")


if __name__ == "__main__":
    main()