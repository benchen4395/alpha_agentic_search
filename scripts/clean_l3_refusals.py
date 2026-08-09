#!/usr/bin/env python
# scripts/clean_l3_refusals.py
"""清理 L3 历史层里的「拒答 / 部分拒答」条目（修复②的配套工具）。

════════════════════════════════════════════════════════════════════════
为什么需要这个脚本
════════════════════════════════════════════════════════════════════════
`cache_policy.is_partial_refusal()` + `agent._archive_if_enabled()` 的修复
只能阻止**今后**的拒答进入 L3，**已经写进去的历史污染需要单独清理**。

这不是可选的收尾工作，而是修复生效的**前置条件**。实测证明两个 bug
会互相掩盖：

    query = "茅盾文学奖 历届 获奖名单"

    含被污染的 L3 条目 → 实词覆盖率 0.75   ← 不触发 L4（判为"证据够用"）
    仅干净的 L2 条目   → 实词覆盖率 0.25   ← 正确触发 L4

原因很微妙：被污染的 L3 条目里存着 **query 原文**
（格式是「历史问答：Q: <原问题> A: <拒答>」），于是 query 的每个实词都能
在"证据"里找到 —— 覆盖率判据被自己的历史拒答骗过去了。

也就是说：**不清理 L3，修复①就是失效的**。这也是把这三个问题
放在同一个 Stage 一起修的原因。

════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════
    # 先干跑，只看会删什么（强烈建议先跑这个）
    python scripts/clean_l3_refusals.py

    # 确认无误后真正执行
    python scripts/clean_l3_refusals.py --apply

实现说明：
  * L3 用 `metadata.jsonl` + `vectors.faiss` 两个文件平行存储，
    **第 i 行 metadata 对应第 i 个向量**。所以删除时必须同步重建
    两者，不能只改 jsonl —— 否则索引错位，检索会返回张冠李戴的内容。
  * 会先备份原文件（`.bak`），出问题可以直接还原。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache_policy import is_low_quality_answer, is_partial_refusal  # noqa: E402
from rag import config as rag_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="清理 L3 里的拒答条目")
    ap.add_argument("--apply", action="store_true",
                    help="真正执行删除（默认只干跑预览）")
    args = ap.parse_args()

    d = rag_config.L3_HISTORY_INDEX_DIR
    meta_path = os.path.join(d, "metadata.jsonl")
    vec_path = os.path.join(d, "vectors.faiss")

    if not os.path.exists(meta_path):
        print(f"[clean-l3] 未找到 {meta_path}，L3 尚未建立，无需清理。")
        return 0

    rows = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # 标记要删除的行号。两类都删：
    #   - 纯拒答（is_low_quality_answer，看开头）
    #   - 部分拒答（is_partial_refusal，看结尾）—— 最容易漏掉的一类
    drop: list[int] = []
    for i, r in enumerate(rows):
        ans = r.get("answer") or ""
        if is_low_quality_answer(ans) or is_partial_refusal(ans):
            drop.append(i)

    print(f"[clean-l3] L3 共 {len(rows)} 条，检出污染 {len(drop)} 条：")
    for i in drop:
        q = (rows[i].get("query") or "")[:46]
        kind = "部分拒答" if is_partial_refusal(rows[i].get("answer") or "") else "纯拒答"
        print(f"    #{i:<4} [{kind}] {q}")

    if not drop:
        print("[clean-l3] 无需清理。")
        return 0

    if not args.apply:
        print("\n[clean-l3] 这是**干跑**，未做任何修改。"
              "确认无误后加 --apply 真正执行。")
        return 0

    # ---- 备份 ----
    for p in (meta_path, vec_path):
        if os.path.exists(p):
            shutil.copy2(p, p + ".bak")
    print(f"[clean-l3] 已备份为 *.bak")

    keep = [i for i in range(len(rows)) if i not in set(drop)]

    # ---- 同步重建 metadata + faiss 向量 ----
    # ⚠️ 关键：两者必须**按同一顺序**重建。只删 jsonl 会让第 i 行
    # metadata 对应到第 i 个（错位的）向量，检索结果张冠李戴 ——
    # 这种错误不会报异常，只会静默返回错内容，极难排查。
    try:
        import faiss  # type: ignore
        import numpy as np

        idx = faiss.read_index(vec_path)
        n = idx.ntotal
        if n != len(rows):
            print(f"[clean-l3] ⚠️ 向量数({n}) 与 metadata 行数({len(rows)}) 不一致，"
                  f"已中止以免造成更严重的错位。请手工检查。")
            return 1
        # 取出要保留的向量，重建一个同类型的新索引
        vecs = np.vstack([idx.reconstruct(i) for i in keep]).astype("float32")
        new_idx = faiss.IndexFlatIP(vecs.shape[1])
        new_idx.add(vecs)
        faiss.write_index(new_idx, vec_path)
        print(f"[clean-l3] 向量索引已重建：{n} → {new_idx.ntotal}")
    except ImportError:
        print("[clean-l3] ⚠️ 未安装 faiss，跳过向量重建。"
              "请注意 metadata 与向量将不再对齐，建议改为整体重建 L3。")
        return 1

    with open(meta_path, "w", encoding="utf-8") as f:
        for i in keep:
            f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    print(f"[clean-l3] metadata 已重写：{len(rows)} → {len(keep)}")
    print("[clean-l3] ✓ 清理完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
