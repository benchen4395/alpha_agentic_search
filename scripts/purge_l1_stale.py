#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按**当前**准入判据体检并清理 L1（QACache）里的存量脏条目。

════════════════════════════════════════════════════════════════════════
为什么需要这个脚本 —— 「判据升级不回溯」
════════════════════════════════════════════════════════════════════════
`cache_policy.decide_cacheability()` 是**写入侧**门禁，语义是
「今后不准再写这种条目」，而不是「这种条目不准被返回」。所以每次收紧
判据（补拒答线索、调长度下限…）都对**存量条目零作用**。

`agent._l1_admissible()` 已经在**读取侧**补了一道复核，能让判据升级
立刻生效（脏条目被当作 miss）。但抑制不等于清理：
  * 条目仍占磁盘、仍参与 fuzzy 向量检索的候选集；
  * `retriever` 内部的 L1 层仍可能把它当证据召回；
  * 每次命中都要多跑一次判据 + 打一行日志。
所以仍然需要一个能**物理删除**的运维入口。

════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════
    # 只体检，不删（默认，安全）
    python scripts/purge_l1_stale.py

    # 确认无误后真删
    python scripts/purge_l1_stale.py --apply

    # 只看某一类
    python scripts/purge_l1_stale.py --tier reject_partial_refusal

⚠️ 默认 dry-run 是刻意的：判据本身在迭代，一条误判的规则配上自动删除
   会静默清掉真正有价值的积累，而缓存积累恰恰是"越用越强"的全部资产。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cache.cache_policy import decide_cacheability  # noqa: E402
from src.cache.qa_cache import QACache  # noqa: E402
from src.pipeline.web_intent import wants_web_search  # noqa: E402


def _split_key(key: str) -> tuple[str | None, str]:
    """把存储 key 拆成 (namespace, 原始 query)。

    key 形如 `u:<uid>::<query>` / `s:<sid>::<query>` / `<query>`。
    """
    if "::" in key:
        ns, _, q = key.partition("::")
        return ns, q
    return None, key


def main() -> int:
    ap = argparse.ArgumentParser(description="按当前判据清理 L1 存量脏条目")
    ap.add_argument("--apply", action="store_true",
                    help="真正删除（默认只体检不删）")
    ap.add_argument("--tier", default="",
                    help="只处理指定 tier，如 reject_partial_refusal")
    args = ap.parse_args()

    cache = QACache()
    # `_mem` 是 QACache 的内存索引，冷启动时已从后端全量载入，可直接枚举。
    # 拷一份再遍历：删除会改动原 dict。
    items = dict(cache._mem)
    print(f"L1 共 {len(items)} 条，开始按当前判据复核…\n")

    stale: list[tuple[str, str, str, str]] = []   # (key, query, tier, reason)
    for key, answer in items.items():
        ns, query = _split_key(key)

        # ---- ① key 本身就是联网指令 → 永久死条目 ----
        # `chat()` 里对这类 query 恒有 `force_web=True`，于是 L1 的**读**
        # 被永久旁路 —— 这条记录再也不可能被命中，纯粹占着磁盘和 fuzzy
        # 候选集。它们是 `skip_l1` 修复**之前**写进去的存量垃圾。
        #
        # 语义上也本就不该缓存：「请帮我重新联网搜一次 X」这句话的正确
        # 语义是"每次都要重新搜"，给它一个冻结 30 天的答案与用户下指令的
        # 本意完全相反。
        if wants_web_search(query):
            if not args.tier or args.tier == "web_directive_key":
                stale.append((key, query, "web_directive_key",
                              "key 本身是联网指令 → force_web 恒为真、"
                              "L1 读被永久旁路，此条永远不可能命中"))
            continue

        # ---- ② 按当前准入判据复核 ----
        # ⚠️ 不能传 layer_hits：存量条目没有保存当初的层命中信息。
        # 影响仅限于 TTL 档位（web_backed vs stable），而我们只关心
        # cacheable 这个布尔值，所以不传是安全的。
        d = decide_cacheability(query, answer)
        if d.cacheable:
            continue
        if args.tier and d.tier != args.tier:
            continue
        stale.append((key, query, d.tier, d.reason))

    if not stale:
        print("✅ 没有需要清理的条目。")
        return 0

    print(f"❗ 按当前判据**本应被拒**却仍留在 L1 的条目：{len(stale)} 条\n")
    for key, query, tier, reason in stale:
        print(f"  [{tier}] {query[:48]}")
        print(f"      {reason}")

    if not args.apply:
        print(f"\n（dry-run，未删除任何条目。确认无误后加 --apply 执行删除）")
        return 0

    deleted = 0
    for key, query, _tier, _reason in stale:
        ns, q = _split_key(key)
        try:
            if cache.remove(q, namespace=ns):
                deleted += 1
        except Exception as e:
            print(f"  ⚠️ 删除失败 {key[:40]}: {e}")
    cache.flush(timeout=10)
    print(f"\n✅ 已删除 {deleted}/{len(stale)} 条，L1 剩余 {len(cache._mem)} 条。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
