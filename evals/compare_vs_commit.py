# -*- coding: utf-8 -*-
"""用 **tests/ 里的用例** 对比「当前工作区」与「git commit 版本」。

⚠️ 先说清楚这里的"准确率"是什么
────────────────────────────────────────────────────────────────────
`tests/` 是**单元测试套件**，不是问答评测集 —— 它里面没有
「query → 标准答案」这样的成对数据，所以没法直接算出"回答对了几道题"。
它衡量的是**行为正确性**：每条 test 是一条断言过的行为契约
（缓存该不该命中、实体该不该抽出来、超时参数该不该传下去…）。

因此本脚本报的两个数：
    准确率 = 通过的测试条数 / 总条数     （行为契约的满足率）
    耗时   = 整套测试的墙钟时间           （含真实 BGE-M3 编码等重活）

这与 evals/ 上的"问答准确率"是**两个不同口径**，不能互相替代：
    tests/  测的是"代码行为对不对"     —— 确定性、可复现、无网络依赖
    evals/  测的是"最终答得对不对"     —— 有 LLM 随机性、依赖外部 API

对比方式
────────────────────────────────────────────────────────────────────
git worktree 检出旧 commit 到独立目录，两边跑**同一套**测试文件
（用当前的 tests/ 覆盖过去），这样：
  · 断言口径完全一致，差异只来自 src/ 的实现
  · 新增的测试在旧版上会**失败**，这正是我们要量化的"修好了什么"

⚠️ 必须覆盖 tests/：如果各自跑各自的 tests，旧版跑的是旧断言，
测出来的"两边都 100%"毫无意义 —— 那只说明各自满足各自的标准。
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "evals" / "out"


def _parse(text: str) -> dict[str, int]:
    """从 pytest 输出里抽出各类计数。

    ⚠️ 不解析 "N passed, M failed in X.XXs" 那行统计行 —— 本项目里它
    **拿不到**：`test_prefetch_failure_does_not_break_warmup` 会故意触发
    `PytestUnhandledThreadExceptionWarning`，warnings summary 把尾部刷掉，
    实测 grep "passed" 在输出里一条都匹配不到。

    改为解析**进度字符**（`......F..s [ 18%]` 那几行），它在
    warnings 之前输出，一定拿得到，且逐条对应一个用例：
        . = passed   F = failed   E = error   s = skipped   x = xfail
    """
    prog = []
    for line in text.splitlines():
        m = re.match(r"^([.FEsxX]+)\s+\[\s*\d+%\]", line)
        if m:
            prog.append(m.group(1))
    chars = "".join(prog)
    return {
        "passed": chars.count("."),
        "failed": chars.count("F"),
        "error": chars.count("E"),
        "skipped": chars.count("s"),
        "xfail": chars.count("x") + chars.count("X"),
    }


def _failed_ids(text: str) -> list[str]:
    """抽出失败/报错的用例 ID，用于逐条对比"两边分别坏在哪"。"""
    ids = []
    for line in text.splitlines():
        m = re.match(r"(FAILED|ERROR)\s+(\S+)", line.strip())
        if m:
            ids.append(m.group(2).split(" ")[0])
    return sorted(set(ids))


def run_pytest(cwd: Path, tag: str, repeats: int) -> dict:
    """在指定目录跑整套 tests/，重复 `repeats` 次取耗时中位数。

    为什么要重复：单次墙钟时间受磁盘缓存、MPS kernel 编译、系统负载
    影响很大，首次跑会因为要 mmap GB 级索引而明显偏慢。取多次的
    **最小值**更能反映稳态耗时（最小值受噪声干扰最小，是基准测试的
    常规做法；均值会被偶发的系统抖动整体抬高）。
    """
    OUT.mkdir(parents=True, exist_ok=True)
    times: list[float] = []
    text = ""
    for i in range(repeats):
        t0 = time.perf_counter()
        p = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "-p",
             "no:cacheprovider", "--tb=no", "-rf"],
            cwd=cwd, capture_output=True, text=True, errors="replace",
        )
        dt = time.perf_counter() - t0
        times.append(dt)
        text = p.stdout + p.stderr
        print(f"  [{tag}] run {i+1}/{repeats}: {dt:.1f}s  exit={p.returncode}")
    (OUT / f"cmp_{tag}.txt").write_text(text, errors="replace")
    stat = _parse(text)
    stat["wall_min"] = min(times)
    stat["wall_all"] = times
    stat["failed_ids"] = _failed_ids(text)
    return stat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="8dd2802", help="对照的 commit")
    ap.add_argument("--worktree", default="/tmp/base_ver")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    base = Path(args.worktree)
    if not base.exists():
        print(f"❌ 找不到 worktree {base}，先执行：\n"
              f"   git worktree add {base} {args.base}")
        return 1

    # ── 关键一步：把当前的 tests/ 复制到旧版目录 ──
    # 让两边用**同一套断言**。否则旧版跑旧断言，测出来的"都通过"
    # 只说明各自满足各自的标准，无法回答"改进了多少"。
    shutil.rmtree(base / "tests", ignore_errors=True)
    shutil.copytree(REPO / "tests", base / "tests",
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    # evals/ 也要带过去：有测试会 import evals.datasets 做误报率上界检查
    shutil.rmtree(base / "evals", ignore_errors=True)
    shutil.copytree(REPO / "evals", base / "evals",
                    ignore=shutil.ignore_patterns("__pycache__", "out"))
    print(f"[cmp] 已把当前 tests/ + evals/ 同步到 {base}（保证断言口径一致）")

    print(f"\n{'='*78}\n跑 commit 版本 ({args.base})\n{'='*78}")
    old = run_pytest(base, "old", args.repeats)
    print(f"\n{'='*78}\n跑当前工作区\n{'='*78}")
    new = run_pytest(REPO, "new", args.repeats)

    def total(s):
        return s["passed"] + s["failed"] + s["error"]

    def rate(s):
        t = total(s)
        return s["passed"] / t if t else 0.0

    print(f"\n\n{'='*78}\n结果（数据来源：tests/ 全量用例）\n{'='*78}")
    print(f"{'版本':<12}{'通过':>10}{'失败':>8}{'报错':>8}{'跳过':>8}"
          f"{'通过率':>10}{'耗时':>12}")
    for name, s in (("commit版", old), ("当前版", new)):
        print(f"{name:<12}{s['passed']:>10}{s['failed']:>8}{s['error']:>8}"
              f"{s['skipped']:>8}{rate(s)*100:>9.1f}%{s['wall_min']:>10.1f}s")

    d_rate = (rate(new) - rate(old)) * 100
    d_pass = new["passed"] - old["passed"]
    d_time = new["wall_min"] - old["wall_min"]
    pct = d_time / old["wall_min"] * 100 if old["wall_min"] else 0
    print(f"\n{'-'*78}")
    print(f"通过率  {rate(old)*100:.1f}% → {rate(new)*100:.1f}%  "
          f"（{d_rate:+.1f} 个百分点，多通过 {d_pass:+d} 条）")
    print(f"耗时    {old['wall_min']:.1f}s → {new['wall_min']:.1f}s  "
          f"（{d_time:+.1f}s, {pct:+.1f}%）")
    print("⚠️ 耗时差包含了**新增测试本身**的开销（它们在旧版上直接报错退出、几乎不花时间，\n"
          "   在新版上则要真的加载 KG、跑 8 线程并发、扫全量 BCZ）。\n"
          "   要看**生产代码**的真实耗时变化，得单独基准热路径，不能看这个总差。")

    only_old = [x for x in old["failed_ids"] if x not in new["failed_ids"]]
    only_new = [x for x in new["failed_ids"] if x not in old["failed_ids"]]
    print(f"\n本次修好（旧版失败、当前通过）: {len(only_old)} 条")
    for x in only_old:
        print(f"   ✅ {x}")
    print(f"本次弄坏（旧版通过、当前失败）: {len(only_new)} 条")
    for x in only_new:
        print(f"   ❌ {x}")
    if not only_new:
        print("   （无 —— 现有功能未受影响）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
