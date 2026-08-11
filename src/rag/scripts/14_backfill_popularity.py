"""
14_backfill_popularity.py
=========================

给**已建好**的 KG 库回填 `entities.popularity`（实体入度），修复实体消歧
无法排序的问题。

════════════════════════════════════════════════════════════════════════
问题现象
════════════════════════════════════════════════════════════════════════
同名实体的候选排序完全失效，主实体排不进前列：

    link("北京")  → 第一候选 Q578328「美國伊利諾伊州塔茲韋爾縣的縣城」
    link("中国")  → 第一候选 Q2736887「1972年安東尼奧尼的電影」
    Q956(北京市) / Q148(中国) 连前 5 都进不去

后果比"召回为空"更危险：KG 会把**张冠李戴的事实**喂给 LLM，
而且全程不报错，看起来一切正常。

════════════════════════════════════════════════════════════════════════
根因
════════════════════════════════════════════════════════════════════════
`kg_store.link()` 的排序是 `ORDER BY weight DESC, article_rank ...`，
但这两个字段在本库里几乎都没有区分度：

    popularity   非空(>0):     0 / 10,385,628  =  0.00%   ← 建库时恒置 0
    article_rank 非空:      2,869 / 10,385,628 =  0.03%

于是同名候选（全是 weight=1.0、rank=NULL）之间**没有任何排序依据**，
先后顺序只取决于物理存储顺序（即 QID 数值大小）。QID 越小代表在
Wikidata 里登记越早，与"是否是用户想要的那个实体"毫无关系。

源头已在 `10_build_kg_sqlite.py` 里修好（灌完 triples 后回填 popularity
并加索引），所以下次重建不会复发。本脚本负责修**已经建好的库**。

════════════════════════════════════════════════════════════════════════
为什么用入度（in-degree）
════════════════════════════════════════════════════════════════════════
`triples.object_qid` 的被引用次数 = 有多少实体指向它，是 KG **自带的、
不依赖任何外部数据**的重要度信号，区分度达 5 个数量级：

    Q148    中国            入度 1,069,530
    Q1074318 中国国家博物馆   入度   267,761
    Q956    北京市          入度     5,314
    Q578328 美国伊州小镇北京  入度        11
    Q1305924 北京(消歧义页)   入度         0

相比之下 article_rank 依赖 04 步的 pageviews 数据，覆盖率只有 0.03%，
无法作为主排序键。

════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════
    # 干跑：只统计，不写库（默认）
    python src/rag/scripts/14_backfill_popularity.py

    # 实跑
    python src/rag/scripts/14_backfill_popularity.py --apply

幂等：重复执行结果一致（每次都是按当前 triples 重算，不是累加）。
所以可以放进定期更新流程里无脑重跑。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="KG sqlite 路径")
    ap.add_argument("--apply", action="store_true", help="真正写库")
    args = ap.parse_args()

    if args.db:
        db_path = Path(args.db)
    else:
        db_path = (Path(__file__).resolve().parents[3]
                   / "data" / "rag_data" / "wikidata_zh_kg.db")
    if not db_path.exists():
        sys.exit(f"找不到 KG 库：{db_path}")

    print(f"[14] {'🔧 APPLY' if args.apply else '👀 DRY-RUN'}  db = {db_path}")
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-200000")
    con.execute("PRAGMA temp_store=MEMORY")

    tot = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    cur = con.execute(
        "SELECT COUNT(*) FROM entities WHERE popularity > 0").fetchone()[0]
    print(f"[14] 当前 popularity>0 的实体: {cur:,} / {tot:,}")

    if not args.apply:
        # 干跑：展示几个典型实体的入度，供人眼确认信号有效
        print("\n── 入度抽样（干跑）──")
        for qid, name in [("Q148", "中国"), ("Q956", "北京市"),
                          ("Q578328", "美国伊州小镇北京"),
                          ("Q1305924", "北京(消歧义页)")]:
            n = con.execute(
                "SELECT COUNT(*) FROM triples WHERE object_qid=?", (qid,)
            ).fetchone()[0]
            print(f"  {qid:<12}{name:<18} 入度 = {n:,}")
        print("\n干跑结束。确认无误后加 --apply 执行。")
        con.close()
        return

    # ── 回填 ──
    # 用**关联子查询**而不是先建临时表再 JOIN：SQLite 对
    # `WHERE object_qid = entities.qid` 能直接走 idx_triples_obj 索引，
    # 一次 UPDATE 扫完即可，不需要额外的临时表和排序开销。
    print("[14] 回填中（一次全表 UPDATE）...")
    t0 = time.perf_counter()
    con.execute("""
        UPDATE entities SET popularity = COALESCE((
            SELECT COUNT(*) FROM triples t WHERE t.object_qid = entities.qid
        ), 0)
    """)
    con.commit()
    print(f"[14] ✓ 回填完成 ({time.perf_counter() - t0:.0f}s)")

    # popularity 是消歧排序的主键，没索引每次 link 都要全表扫
    print("[14] 建索引 idx_entities_pop ...")
    t0 = time.perf_counter()
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_entities_pop ON entities(popularity DESC)")
    con.commit()
    print(f"[14] ✓ 索引完成 ({time.perf_counter() - t0:.0f}s)")

    n = con.execute(
        "SELECT COUNT(*) FROM entities WHERE popularity > 0").fetchone()[0]
    print(f"[14] popularity>0 的实体: {n:,} / {tot:,}  ({n/tot:.1%})")

    print("\n── 验证：常见 mention 的首位候选 ──")
    for mention in ["北京", "中国", "苹果公司", "爱因斯坦", "长江"]:
        row = con.execute("""
            SELECT e.qid, e.label_zh, e.popularity, e.description
            FROM mentions m JOIN entities e ON m.qid = e.qid
            WHERE m.mention = ?
            ORDER BY m.weight DESC, e.popularity DESC
            LIMIT 1
        """, (mention,)).fetchone()
        if row:
            print(f"  {mention:<8} → {row[0]:<11} pop={row[2]:<8} "
                  f"{row[1]} | {str(row[3] or '')[:26]}")
        else:
            print(f"  {mention:<8} → (无精确候选)")

    print("\n[14] ANALYZE ...")
    con.execute("ANALYZE")
    con.commit()
    con.close()
    print("[14] done. ⚠️ 需重启服务：KGStore 是进程内单例。")


if __name__ == "__main__":
    main()
