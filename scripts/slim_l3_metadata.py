#!/usr/bin/env python3
"""一次性迁移：给历史 L3 归档数据的 `sources` 字段瘦身。

背景
────
早期版本归档 L3 时直接写入 `[p.to_dict() for p in passages]`，其中
`metadata` 又携带着上一轮归档的 `sources`。于是每归档一轮就多套一层，
嵌套深度随轮数线性增长、单条体积指数增长。实测 670 条记录把
`metadata.jsonl` 撑到 157 MB（最大单条 22.9 MB、嵌套 53 层），
仅进程启动时的 json 解析就要接近 1 秒。

代码侧已经在 `rag/layers.py` 的**读写两侧**都做了拦截，新数据不会再膨胀。
但**存量数据**仍然是胖的，需要这个脚本清理一次。

安全性
──────
* 只重写 `sources` 字段，其余字段逐字保留。
* **严格保持行数与行序不变** —— `metadata.jsonl` 的第 N 行必须对应
  `vectors.faiss` 的第 N 个向量。任何增删行都会让索引与元数据错位，
  导致检索结果张冠李戴。因此哪怕某行解析失败，也**原样写回**而不是跳过。
* 先写临时文件，全部成功后才原子替换；中途失败不会损坏原文件。
* 默认会留一份 `.bak` 备份。

用法
────
    python scripts/slim_l3_metadata.py                 # 处理默认路径
    python scripts/slim_l3_metadata.py --dry-run       # 只统计，不写入
    python scripts/slim_l3_metadata.py --path <file>   # 指定文件
    python scripts/slim_l3_metadata.py --no-backup     # 不留 .bak
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.layers import L3HistoryLayer  # noqa: E402

DEFAULT_PATH = os.path.join("data", "rag_data", "l3_history", "metadata.jsonl")


def _fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> int:
    ap = argparse.ArgumentParser(description="L3 归档 sources 瘦身迁移")
    ap.add_argument("--path", default=DEFAULT_PATH, help="metadata.jsonl 路径")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写入")
    ap.add_argument("--no-backup", action="store_true", help="不生成 .bak")
    args = ap.parse_args()

    path = args.path
    if not os.path.exists(path):
        print(f"❌ 文件不存在：{path}")
        return 1

    before = os.path.getsize(path)
    print(f"📄 {path}\n   处理前：{_fmt(before)}")

    tmp = path + ".slim.tmp"
    total = bad = slimmed = 0
    max_line_before = 0

    # 逐行流式处理：文件可能有上百 MB，一次性 readlines() 会把内存打爆。
    with open(path, "r", encoding="utf-8") as fin, \
            open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            max_line_before = max(max_line_before, len(line))
            stripped = line.strip()
            if not stripped:
                # 空行也照写，保持行号对齐
                fout.write(line)
                continue
            try:
                obj = json.loads(stripped)
            except Exception:
                # 解析失败 → 原样写回。宁可留一条胖记录，
                # 也不能让行号与 faiss 向量错位。
                bad += 1
                fout.write(line)
                continue
            if isinstance(obj, dict) and obj.get("sources"):
                new_src = L3HistoryLayer._slim_sources(obj.get("sources"))
                if new_src != obj.get("sources"):
                    obj["sources"] = new_src
                    slimmed += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    after = os.path.getsize(tmp)
    ratio = (before / after) if after else 0.0
    print(f"   处理后：{_fmt(after)}   （缩减 {ratio:.1f}x）")
    print(f"   共 {total} 行｜瘦身 {slimmed} 行｜解析失败保留原样 {bad} 行")
    print(f"   处理前最大单行：{_fmt(max_line_before)}")

    if args.dry_run:
        os.remove(tmp)
        print("🔍 dry-run：未改动原文件")
        return 0

    if not args.no_backup:
        bak = path + ".bak"
        shutil.copy2(path, bak)
        print(f"💾 备份：{bak}")

    os.replace(tmp, path)   # 原子替换
    print("✅ 完成。确认检索正常后可删除 .bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
