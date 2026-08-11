"""
15_gen_suffix_aliases.py
========================

为**高知名度实体**补充"后缀剥离"别名，修复 Wikidata alias 覆盖缺失导致
的实体消歧召回失败。

════════════════════════════════════════════════════════════════════════
问题现象
════════════════════════════════════════════════════════════════════════
    link("北京")  → Q578328「美國伊利諾伊州塔茲韋爾縣的縣城」(入度 11)
    真正的 Q956「北京市」(入度 5,314) **根本不在候选里**

════════════════════════════════════════════════════════════════════════
根因：不是排序问题，是召回问题
════════════════════════════════════════════════════════════════════════
Q956 在 `mentions` 表里登记的字面串是：

    ['北京市', '京兆地方', '京城', '北平', '北平市', '帝都', '平', '燕', ...]

**没有"北京"这两个字**。Wikidata 的 alias 是人工维护的，
"label 去掉行政区后缀"这种在中文里天经地义的说法反而常常没人登记。
排序再怎么改也救不回一个压根没被召回的实体。

【为什么不能靠 FTS5 前缀兜底】已实测否决：
  - `'"北京"*'` 前缀命中 **156,306 行**，且 FTS5 的 MATCH **本身不排序**，
    要拿到高入度实体就得把这 15 万行全 JOIN + 排序 —— 实测 2.3~9.1 秒，
    而 L5 整层预算只有几百毫秒；
  - 若截断候选池提速，截出来的是**任意子集**（FTS 无序），
    pool=1000 时 top1 是「2003 北京」(入度 1)，pool=20000 才捞到 Q956
    但要 2.9 秒。快和准不可兼得。
本质上前缀匹配语义就是错的："北京大学""北京烤鸭"都是合法前缀命中，
它们和"北京"指的不是一回事。

════════════════════════════════════════════════════════════════════════
方案：建库时把缺失的 alias 补进去
════════════════════════════════════════════════════════════════════════
对 label 形如「X + 通用后缀」的实体，补一条 X 的 alias。
但**必须严格设限**，否则会造出大量垃圾（实测无限制时会生成
"美国城"←美国城市、"未建制社"←未建制社区、"上市"←上市公司）：

  ① 只处理 popularity（入度）≥ MIN_POP 的实体 —— 冷门实体本来就不该
     在消歧里胜出，给它们加 alias 只会增加噪声；
  ② 后缀必须是**行政区划**类的真后缀（市/省/县/区/自治区），
     且剥离后剩余长度 ≥ 2（避免"平""燕"这种单字歧义串）。
     ⚠️ 刻意**不收「大学」「公司」**：干跑实测「东京大学→东京」
     「北京大学→北京」会与真正的城市实体撞车，见 _SUFFIXES 处说明；
  ③ 用 P31(instance of) 排除**类概念**：Q1093829「美国城市」的 P31 是
     「行政區劃類型」，Q891723「上市公司」的 P31 是「组织类型」——
     它们是"类"不是"实例"，剥后缀毫无意义。而 Q956 的 P31 是
     「城市/直辖市/首都」，是真实例；
  ④ 要求实体带 P17(国家) 或 P131(所属行政区)，确保是**真实地理实体**。
     干跑实测：光看后缀会误伤抽象概念 ——「大城市→大城」「近邻社区→近邻社」
     「南极条约地区→南极条约地」，它们只是碰巧以"市/区"结尾。
     真行政区必定挂在某国家/上级区划下（北京市、京都市、華盛頓特區
     均 P17=True、P131=True），而这些抽象概念两个属性都没有。
     这个结构性判据比维护黑名单可靠得多；
  ⑤ 若 base 已指向另一个**入度更高**的实体则不插入，避免把已经正确的
     映射带偏（"东京"已正确指向东京都，就不该再指向东京大学）；
  ⑥ 该 (mention, qid) 已存在则跳过，绝不覆盖 Wikidata 原始数据。

新增 alias 的 weight 取 0.5（低于原生 alias 的 0.6）：它是**推导**出来
的而非 Wikidata 明示的，证据强度更弱，让原生数据在同分时优先。
配合 `kg_store.link()` 的组合分 `weight × log1p(popularity)`，
高入度实体仍能凭入度优势胜出。

════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════
    python src/rag/scripts/15_gen_suffix_aliases.py            # 干跑
    python src/rag/scripts/15_gen_suffix_aliases.py --apply    # 实跑

⚠️ 必须在 `14_backfill_popularity.py` **之后**跑（依赖 popularity 筛选）。
幂等：已存在的 (mention,qid) 会跳过，重复执行不会产生重复行，
可直接放进定期更新流程。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

# 只给入度 >= 这个值的实体补 alias。
# 取 500：省级行政区(浙江 1162)、知名高校(清华 1740)、直辖市(上海 3778)
# 都在这之上；而同名的虚构地名、美国小镇普遍在 100 以下。
# 调低会显著增加噪声 alias，调高会漏掉一些地级市。
MIN_POP = 500

# 可剥离的后缀。**只收行政区划**：这类「简称=去后缀」在中文里是稳定
# 用法（北京市→北京、浙江省→浙江、朝阳区→朝阳）。
#
# ⚠️ **不收「大学」「公司」** —— 干跑实测发现这两类剥离会造出
# **灾难性的错误映射**：
#     东京大学(pop=3404) → “东京”   ← 东京应该指城市！
#     北京大学(pop=2642) → “北京”   ← 同上
#     花旗银行公司       → “花旗银行” ← 尚可，但不稳定
# 中文高校的简称是「北大」「清华」这种**缩略词**，不是机械地去掉
# “大学”二字；而去掉后剩下的恰恰是**城市名**，会与真正的城市实体
# 直接撞车。行政区划没有这个问题：“北京市”去掉“市”就是北京本尊。
_SUFFIXES = (
    "自治区", "自治區", "特别行政区", "特別行政區",
    "市", "省", "县", "縣", "区", "區",
)

# P31 指向这些"类概念"的实体要排除。
# 判据：它们描述的是一个**类型**而不是一个具体实例。
# 例：Q1093829「美国城市」P31=行政區劃類型 → 剥成"美国城"纯属垃圾。
_TYPE_MARKERS = ("類型", "类型", "列表", "列錶")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-pop", type=int, default=MIN_POP)
    args = ap.parse_args()

    if args.db:
        db_path = Path(args.db)
    else:
        db_path = (Path(__file__).resolve().parents[3]
                   / "data" / "rag_data" / "wikidata_zh_kg.db")
    if not db_path.exists():
        sys.exit(f"找不到 KG 库：{db_path}")

    print(f"[15] {'🔧 APPLY' if args.apply else '👀 DRY-RUN'}  db = {db_path}")
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-200000")

    if con.execute("SELECT COUNT(*) FROM entities WHERE popularity>0"
                   ).fetchone()[0] == 0:
        sys.exit("popularity 全为 0：请先跑 14_backfill_popularity.py --apply")

    # 预取"类概念" QID 集合，避免在循环里反复查库
    type_qids = {
        r["qid"] for r in con.execute(
            "SELECT qid, label_zh FROM entities WHERE label_zh IS NOT NULL "
            "AND (" + " OR ".join(
                f"label_zh LIKE '%{m}%'" for m in _TYPE_MARKERS) + ")")
    }
    print(f"[15] 类概念实体 {len(type_qids):,} 个（用于排除）")

    rows = list(con.execute(
        "SELECT qid, label_zh, popularity FROM entities "
        "WHERE popularity >= ? AND label_zh IS NOT NULL",
        (args.min_pop,)))
    print(f"[15] 候选实体（入度 >= {args.min_pop}）: {len(rows):,}")

    t0 = time.perf_counter()
    new_rows, skipped_type, skipped_exist, demo = [], 0, 0, []
    skipped_rival = 0
    for r in rows:
        label, qid = r["label_zh"], r["qid"]
        for suf in _SUFFIXES:
            if not label.endswith(suf):
                continue
            base = label[: -len(suf)]
            # 剥完至少要剩 2 个字：单字串（"平""燕"）歧义太大
            if len(base) < 2:
                break
            # ① 必须是**真实的地理实体**：带 P17(国家) 或 P131(所属行政区)。
            # 干跑实测发现光看后缀会把**抽象概念**也剥了：
            #     大城市(pop=2950)   → “大城”
            #     近邻社区(pop=3001) → “近邻社”
            #     南极条约地区         → “南极条约地”
            # 它们只是碰巧以“市/区”结尾，并不是行政区划。
            # 而真行政区必定挂在某个国家/上级区划下（北京市、京都市、
            # 華盛頓特區均 P17=True P131=True），而上述抽象概念**两个都没有**。
            # 这个结构性判据比维护黑名单可靠得多。
            preds = {x[0] for x in con.execute(
                "SELECT DISTINCT predicate_pid FROM triples WHERE subject_qid=?",
                (qid,))}
            if not ({"P17", "P131"} & preds):
                skipped_type += 1
                break
            # ② P31 指向类概念 → 这是个“类”不是“实例”，跳过
            p31 = {x[0] for x in con.execute(
                "SELECT object_qid FROM triples "
                "WHERE subject_qid=? AND predicate_pid='P31'", (qid,))}
            if p31 & type_qids or not p31:
                skipped_type += 1
                break
            if con.execute("SELECT 1 FROM mentions WHERE mention=? AND qid=?",
                           (base, qid)).fetchone():
                skipped_exist += 1
                break
            # ⚠️ 冲突检查：剪出来的 base 若已经指向另一个**更重要**的
            # 实体，就不要再添一条与之竞争。否则会把已经正确的映射带偏：
            #     “东京” 已正确指向 Q1490(东京都)，不该再指向东京大学。
            # 只有当新实体的入度**严格高于**现有最优候选时才插入。
            rival = con.execute(
                "SELECT MAX(e.popularity) FROM mentions m "
                "JOIN entities e ON e.qid=m.qid WHERE m.mention=?",
                (base,)).fetchone()[0]
            if rival is not None and rival >= r["popularity"]:
                skipped_rival += 1
                break
            new_rows.append((base, qid, 0.5, "alias_suffix"))
            if len(demo) < 12:
                demo.append(f"{base:<12} → {qid:<11} "
                            f"(from {label}, pop={r['popularity']:,})")
            break

    print(f"[15] 扫描完成 ({time.perf_counter() - t0:.0f}s)")
    print(f"[15]   可新增 alias : {len(new_rows):,}")
    print(f"[15]   跳过(类概念) : {skipped_type:,}")
    print(f"[15]   跳过(已存在) : {skipped_exist:,}")
    print(f"[15]   跳过(有更强同名): {skipped_rival:,}")
    print("\n── 样例 ──")
    for d in demo:
        print("   ", d)

    if not args.apply:
        print("\n干跑结束。确认无误后加 --apply 执行。")
        con.close()
        return

    print(f"\n[15] 写入 {len(new_rows):,} 条 ...")
    # OR IGNORE：mentions 的主键是 (mention,qid)，万一有并发/重复
    # 不要让整批失败。同时 FTS 是 external-content 表、无触发器，
    # 必须手工同步，否则新 alias 走不到 FTS 兜底那一路。
    con.executemany(
        "INSERT OR IGNORE INTO mentions(mention,qid,weight,source) "
        "VALUES (?,?,?,?)", new_rows)
    con.executemany(
        "INSERT INTO mentions_fts(mention,qid) VALUES (?,?)",
        [(m, q) for m, q, _, _ in new_rows])
    con.commit()

    print("\n── 验证 ──")
    for mention in ["北京", "上海", "浙江", "东京", "广东"]:
        row = con.execute("""
            SELECT e.qid, e.label_zh, e.popularity
            FROM mentions m JOIN entities e ON m.qid = e.qid
            WHERE m.mention = ?
            ORDER BY m.weight * (
                CASE WHEN e.popularity > 0 THEN e.popularity ELSE 1 END
            ) DESC LIMIT 1
        """, (mention,)).fetchone()
        print(f"  {mention:<6} → " +
              (f"{row[0]:<11} pop={row[2]:<8} {row[1]}" if row else "(无)"))

    con.execute("ANALYZE")
    con.commit()
    con.close()
    print("\n[15] done. ⚠️ 需重启服务：KGStore 是进程内单例。")


if __name__ == "__main__":
    main()
