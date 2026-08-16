#!/usr/bin/env python3
# scripts/repair_cache_db.py
"""修复被 git 操作损坏的 diskcache 库（无损，不丢数据）。

════════════════════════════════════════════════════════════════════════
什么时候需要它
════════════════════════════════════════════════════════════════════════
启动时报：

    sqlite3.DatabaseError: database disk image is malformed

触发路径通常是 `QACache.__init__` 里的 `_diskcache.Cache(cache_dir)`。

════════════════════════════════════════════════════════════════════════
为什么会坏：git 与 SQLite 的固有冲突
════════════════════════════════════════════════════════════════════════
`data/qa_cache/cache.db` 与 `data/search_cache/cache.db` 是**被 git 追踪**的
（仓库自带一批预热问答，让新克隆的人开箱就有 L1 命中，见 .gitignore 的说明）。

代价是：**任何切换工作区内容的 git 操作都会按字节覆写这些库**——
`git stash` / `git stash pop` / `git checkout <branch>` / `git reset --hard`
都会。git 只保证文件内容一致，它不理解 SQLite 的页结构和 rowid 顺序，
覆写后主库与残留的 `-wal`/`-shm` 也不再匹配。

实测损坏形态（4 个库全中）：

    Tree 3 page 3 cell 4: Rowid 3017 out of order
    row 6 missing from index sqlite_autoindex_Settings_1

关键观察：**数据页本身通常是好的**——记录都还读得出来（实测 81/103/19/286
条一条没少），坏的只是索引和 rowid 顺序。所以这是可以无损修复的。

════════════════════════════════════════════════════════════════════════
为什么不用更简单的办法
════════════════════════════════════════════════════════════════════════
✗ `git checkout -- data/qa_cache/cache.db`
    会丢数据。git 里的版本是上次提交时的快照，而工作区的库还包含之后
    新写入的缓存（实测差 1 条）。用它"修"等于拿旧版本覆盖新数据。

✗ `sqlite3 db .dump`
    走正常的 B-tree 遍历，遇到 "Rowid out of order" 直接报错中断，
    在本场景下根本跑不完。

✗ 直接删库重建
    丢掉全部预热问答和已积累的缓存，L1 命中率归零。

✓ `sqlite3 db .recover`
    专为损坏库设计：直接扫描数据页抽取记录，绕过损坏的索引结构，
    再重放成一份结构完好的新库。

════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════
    python scripts/repair_cache_db.py            # 检查 + 修复
    python scripts/repair_cache_db.py --check    # 只体检，不改动

安全保证：新库必须同时满足「integrity_check == ok」且「条数不少于原库」
才会替换；任一不满足就跳过并保留原文件，绝不会让情况变得更糟。
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# diskcache 会在主目录下再建 _embeddings / _meta 两个独立子库
TARGETS = [
    REPO / "data" / "qa_cache" / "cache.db",
    REPO / "data" / "qa_cache" / "_embeddings" / "cache.db",
    REPO / "data" / "qa_cache" / "_meta" / "cache.db",
    REPO / "data" / "search_cache" / "cache.db",
]


def integrity(db: Path) -> str:
    """返回 'ok' 或首条错误描述；打不开时返回异常文本。

    ⚠️ 必须自己按换行截断：`PRAGMA integrity_check` 的**单个返回值里就内嵌
    换行**——`fetchone()[0]` 拿到的是一整段多行文本，不是一行。
    损坏库上实测能刷出 30+ 行 "Rowid out of order"，直接打印会把真正
    有用的进度信息淹掉（看起来像是子进程在乱输出，实则来自这里）。
    """
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
            row = c.execute("PRAGMA integrity_check;").fetchone()
        if not row:
            return "unknown"
        lines = [ln for ln in str(row[0]).splitlines() if ln.strip()]
        if not lines:
            return "unknown"
        head = lines[0]
        # 首行常是 "*** in database main ***"，真正的错因在第 2 行
        if head.startswith("***") and len(lines) > 1:
            head = lines[1]
        more = f"（共 {len(lines)} 处）" if len(lines) > 1 else ""
        return f"{head}{more}"
    except Exception as e:                      # noqa: BLE001
        return f"<open failed: {e}>"


def count_rows(db: Path) -> int | None:
    """Cache 表条数；读不出来返回 None（而非 0，两者含义不同）。

    返回 None 和返回 0 必须区分：前者是"读不出来"，后者是"确实空库"。
    若混为一谈，会把读失败误判成空库，从而通过"条数没变少"的门禁
    而用一个空库覆盖掉有数据的原库。
    """
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
            return int(c.execute("SELECT COUNT(*) FROM Cache;").fetchone()[0])
    except Exception:                           # noqa: BLE001
        return None


def repair(db: Path) -> bool:
    """用 .recover 重建。仅在新库校验通过时才替换，返回是否实际修复。"""
    before = count_rows(db)
    with tempfile.TemporaryDirectory() as tmp:
        sql = Path(tmp) / "rec.sql"
        new = Path(tmp) / "rec.db"

        # sqlite3 CLI 的 .recover 没有 Python 侧等价 API，只能走子进程
        try:
            with sql.open("w") as fh:
                _ = subprocess.run(["sqlite3", str(db), ".recover"],
                                   stdout=fh, stderr=subprocess.DEVNULL,
                                   check=True, timeout=300)
            with sql.open() as fh:
                _ = subprocess.run(["sqlite3", str(new)], stdin=fh,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   check=True, timeout=300)
        except FileNotFoundError:
            print("   ✗ 未找到 sqlite3 命令行工具，无法修复")
            return False
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"   ✗ .recover 失败：{e}")
            return False

        chk, after = integrity(new), count_rows(db=new)

        # 双重门禁：结构必须完好，且不能比原库少数据
        if chk != "ok":
            print(f"   ⚠ 跳过：重建后仍不完整（{chk}）")
            return False
        if after is None:
            print("   ⚠ 跳过：重建后读不到 Cache 表")
            return False
        if before is not None and after < before:
            print(f"   ⚠ 跳过：重建后条数变少（{before} → {after}）")
            return False

        shutil.move(str(new), str(db))

    # WAL/SHM 与旧库版本耦合，必须一并清掉，否则 SQLite 可能读到不一致状态
    for suffix in ("-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    print(f"   ✓ 修复成功  {before} → {after} 条")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="修复被 git 操作损坏的 diskcache 库（无损，不丢数据）。")
    _ = ap.add_argument("--check", action="store_true", help="只体检，不修改")
    args = ap.parse_args()

    broken, fixed = 0, 0
    for db in TARGETS:
        if not db.exists():
            continue
        rel = db.relative_to(REPO)
        chk = integrity(db)
        if chk == "ok":
            print(f"── {rel}\n   ok（{count_rows(db)} 条）")
            continue

        broken += 1
        print(f"── {rel}\n   ✗ 损坏：{chk}")
        if not args.check:
            fixed += repair(db)

    if broken == 0:
        print("\n全部正常，无需修复。")
    elif args.check:
        print(f"\n发现 {broken} 个损坏的库。去掉 --check 即可修复。")
    else:
        print(f"\n损坏 {broken} 个，成功修复 {fixed} 个。")
        if fixed < broken:
            bk = "cp -R data/qa_cache data/search_cache /tmp/db_backup"
            print(f"仍有未修复的库。建议先备份再考虑重建：{bk}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
