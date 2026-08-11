# -*- coding: utf-8 -*-
"""建议1（并列实体并发子检索）的专项评测。

⚠️ 为什么要单独一个脚本
────────────────────────────────────────────────────────────────────
GAIA / BrowseComp-ZH 都**不是并列实体题型**，没法直接测建议1 的收益：
    GAIA          0/116  含并列实体（0.0%）
    BrowseComp-ZH 63/289 含并列实体（21.8%）—— 但全是**误报**
BCZ 那 63 条不是真并列题，是 `extract_parallel_entities` 被混淆式
题干骗了，例如：
    「…毕业于北京著名音乐院校…曾前往美国学习先进的音乐制作方法…」
    → ents=['北京','美国']   实际上这是一条单实体反向查找题
之前测出的"单实体误报 0/9"是在**正常口语 query** 上测的，
BCZ 这种长从句堆叠的题干超出了那批样本的分布。

所以这个脚本测两件事：
  ① 收益：在**真**并列题上，并发子检索能不能提高准确率（自建集，
     因为公开集里没有这个题型）
  ② 风险：在 BCZ 的**误报**题上，多拆几路子检索会不会把答案挤掉
     —— 这是上线前必须知道的负面代价
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.datasets import load_bcz  # noqa: E402
from evals.run_multihop_eval import arm_baseline, arm_suggest1  # noqa: E402
from evals.datasets import judge  # noqa: E402
from src.configs.models_config import override_stage  # noqa: E402
from src.rag import entities as rag_entities  # noqa: E402
from src.rag.retriever import LayeredRetriever  # noqa: E402

# ── 真并列题（公开集里没有这个题型，只能自建）──
#   answer_keys: 外层 AND、内层 OR。全部命中才算对 —— 并列题的价值
#   就在于每个实体都要答对，只答对一半等于没答对。
PARALLEL_CASES: list[tuple[str, str, list[tuple[str, ...]]]] = [
    ("PAR-01", "美国、法国和日本的法定最低饮酒年龄分别是多少？",
     [("21",), ("18",), ("20",)]),
    ("PAR-02", "北京和上海的常住人口分别是多少？",
     [("北京",), ("上海",)]),
    ("PAR-03", "珠穆朗玛峰和乔戈里峰的海拔分别是多少米？",
     [("8848", "8844"), ("8611",)]),
    ("PAR-04", "Python、Java 和 Go 分别由谁创造？",
     [("guido", "吉多", "罗苏姆"), ("gosling", "高斯林"),
      ("google", "谷歌", "thompson", "汤普森", "pike", "派克")]),
    ("PAR-05", "德国、意大利和西班牙的现任领导人分别是谁？",
     [("默茨", "merz", "朔尔茨", "scholz"), ("梅洛尼", "meloni"),
      ("桑切斯", "sanchez", "sánchez")]),
    ("PAR-06", "特斯拉和比亚迪 2024 年的全球销量分别是多少？",
     [("特斯拉",), ("比亚迪",)]),
    ("PAR-07", "法国、德国和英国的首都分别是哪座城市？",
     [("巴黎", "paris"), ("柏林", "berlin"), ("伦敦", "london")]),
    ("PAR-08", "长江和黄河的全长分别是多少公里？",
     [("6300", "6397", "6380"), ("5464", "5400", "5464")]),
]


def _judge_multi(text: str, keys) -> tuple[bool, int]:
    """多关键点判分：外层 AND、内层 OR。

    ⚠️ 必须复用 `datasets._norm` 做归一化，不能直接 `lower() + in`。
    实测踩过的坑：PAR-08 模型答的是「长江全长 6,380 多公里」，
    带**千分位逗号**，裸子串匹配 "6380" 直接判错 —— 答案其实是对的。
    这类判分假阴性会污染 arm 之间的比较（两个 arm 都被扣分，
    但扣得未必对称），比少测几题危害更大。
    """
    from evals.datasets import _norm
    t = _norm(text or "")
    hit = sum(1 for g in keys if any(_norm(a) in t for a in g))
    return hit == len(keys), hit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both",
                    choices=["gain", "risk", "both"],
                    help="gain=真并列题收益  risk=BCZ 误报题风险")
    ap.add_argument("--limit", type=int, default=8, help="risk 模式取几题")
    ap.add_argument("--max-tokens", type=int, default=1200,
                    help="summary 输出上限（仅评测用，控制单题耗时）")
    args = ap.parse_args()

    # 与 run_multihop_eval 保持一致：限制 summary 输出长度。
    # 实测 LLM 延迟由**输出**长度主导，不限制的话单题可以卡到
    # 400s+。两个 arm 用同一套限制，不影响相对比较。
    override_stage("summary", extra={"max_tokens": args.max_tokens})
    print(f"[eval] summary max_tokens={args.max_tokens}（仅评测用）")

    r = LayeredRetriever(qa_cache=None, enable_l3=False)
    print("[eval] 已禁用 L1/L3（避免 arm 之间互相污染）")
    r.warmup(verbose=False)

    # ═══════════ ① 收益：真并列题 ═══════════
    if args.mode in ("gain", "both"):
        print(f"\n{'='*88}\n① 收益：真并列实体题（自建，公开集无此题型）\n{'='*88}")
        stat: dict[str, list] = {"baseline": [], "suggest1": []}
        for cid, q, keys in PARALLEL_CASES:
            ents = rag_entities.extract_parallel_entities(q)
            print(f"\n[{cid}] {q}\n  抽到实体: {ents}")
            for name, fn in (("baseline", arm_baseline),
                             ("suggest1", arm_suggest1)):
                t0 = time.perf_counter()
                try:
                    text, meta = fn(r, q)
                except Exception as e:
                    print(f"  {name:<9} 💥 {type(e).__name__}: {e}")
                    continue
                dt = (time.perf_counter() - t0) * 1000
                ok, hit = _judge_multi(text, keys)
                stat[name].append((ok, hit, len(keys), dt))
                mark = "✅" if ok else ("🟡" if hit else "❌")
                print(f"  {name:<9} {mark} {hit}/{len(keys)}  {dt:7.0f}ms"
                      f"{'  子query=' + str(len(meta['subs'])) + '路' if meta['subs'] else ''}")

        print(f"\n{'-'*88}")
        print(f"{'arm':<10}{'全对':>12}{'关键点召回':>14}{'中位耗时':>12}{'均值耗时':>12}")
        for name in ("baseline", "suggest1"):
            rows = stat[name]
            if not rows:
                continue
            n = len(rows)
            full = sum(1 for x in rows if x[0])
            hits = sum(x[1] for x in rows)
            tot = sum(x[2] for x in rows)
            ts = [x[3] for x in rows]
            print(f"{name:<10}{f'{full}/{n}':>12}{f'{hits}/{tot}':>14}"
                  f"{statistics.median(ts):>10.0f}ms{statistics.mean(ts):>10.0f}ms")

    # ═══════════ ② 风险：BCZ 误报题 ═══════════
    if args.mode in ("risk", "both"):
        print(f"\n\n{'='*88}\n② 风险：BCZ 上 extract_parallel_entities 的**误报**题\n"
              f"   （这些不是并列题，多拆几路会不会把答案挤掉）\n{'='*88}")
        fp = [c for c in load_bcz()
              if len(rag_entities.extract_parallel_entities(c.question)) >= 2]
        print(f"  BCZ 误报题共 {len(fp)} 条，取前 {args.limit} 条")
        rows2: dict[str, list] = {"baseline": [], "suggest1": []}
        for c in fp[:args.limit]:
            ents = rag_entities.extract_parallel_entities(c.question)
            print(f"\n[{c.cid}] gold={c.answer}\n  {c.question[:80]}…\n"
                  f"  误报实体: {ents}")
            for name, fn in (("baseline", arm_baseline),
                             ("suggest1", arm_suggest1)):
                t0 = time.perf_counter()
                try:
                    text, _meta = fn(r, c.question)
                except Exception as e:
                    print(f"  {name:<9} 💥 {type(e).__name__}: {e}")
                    continue
                dt = (time.perf_counter() - t0) * 1000
                ok = judge(text, c.answer)
                rows2[name].append((ok, dt))
                print(f"  {name:<9} {'✅' if ok else '❌'}  {dt:7.0f}ms")

        print(f"\n{'-'*88}")
        print(f"{'arm':<10}{'正确':>12}{'中位耗时':>12}{'均值耗时':>12}")
        for name in ("baseline", "suggest1"):
            rr = rows2[name]
            if not rr:
                continue
            ts = [x[1] for x in rr]
            print(f"{name:<10}{f'{sum(1 for x in rr if x[0])}/{len(rr)}':>12}"
                  f"{statistics.median(ts):>10.0f}ms{statistics.mean(ts):>10.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
