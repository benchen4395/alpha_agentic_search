# -*- coding: utf-8 -*-
"""真实评测集加载器：GAIA + BrowseComp-ZH。

数据来源
────────────────────────────────────────────────────────────────────
GAIA          gaia-benchmark/GAIA 是 gated（401），改用公开的纯文本子集
              sayan1101/gaia_filtered_text_only（validation split，127 题，
              带 Final answer 与 Level 1/2/3）。
              选 validation 而非 test：test 的答案不公开，必须提交
              leaderboard 才有分，没法本地判分。
              选"纯文本子集"也符合我们的实际能力 —— 本系统没有
              读 xlsx/mp3/png 的工具，含附件的题无论如何都会挂，
              把它们算进分母只会得到一个无法用于决策的低分。

BrowseComp-ZH PALIN2018/BrowseComp-ZH（test.parquet，289 题）。
              原始 Topic/Question/Answer 三列是 XOR 加密的，密码是
              每行自带的 canary 字段，用官方 decrypt 脚本还原。

拉取方式见 evals/README.md。数据文件不入库（.gitignore）。

⚠️ 关于难度的重要提示
────────────────────────────────────────────────────────────────────
这两个集都是**故意设计成搜索引擎难解**的：
  · GAIA Level 2/3 平均需要 5~10+ 次工具调用，官方 baseline
    GPT-4 + 插件在 Level 1 仅 ~30%、Level 3 接近 0%。
  · BrowseComp-ZH 刻意把实体名**混淆掉**（"某个知名的葡萄酒产区中的
    某个地区…20-40公里范围内存在一家足球俱乐部"），OpenAI 自己的
    BrowseComp 上 GPT-4o 只有 0.6%、带浏览的 o1 约 9.9%。
所以低分是预期内的，绝对分数没有意义；**有意义的是三个 arm 之间的
相对差异** —— 这才是"要不要上 loop agent"的判据。
"""
from __future__ import annotations

import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# ── GAIA 里需要读附件才能做的题：本系统无此能力，直接排除 ──
_FILE_HINT = re.compile(
    r"attached|the file|this file|spreadsheet|xlsx|\.csv|\.pdf|\.mp3|\.png|"
    r"\.jpg|\.docx|\.pptx|\.zip|audio|image|video|recording",
    re.I,
)


class Case:
    """一道评测题。

    Attributes:
        cid:      题号（含数据集前缀）
        source:   gaia / bcz
        level:    GAIA 的 Level（1/2/3）；BrowseComp-ZH 用 Topic
        question: 题干
        answer:   标准答案（短答案，用于判分）
    """

    __slots__ = ("cid", "source", "level", "question", "answer")

    def __init__(self, cid, source, level, question, answer):
        self.cid = cid
        self.source = source
        self.level = str(level)
        self.question = question
        self.answer = str(answer)

    def __repr__(self) -> str:
        return f"<{self.cid} L{self.level} {self.question[:40]}…>"


def load_gaia(limit: int = 0, levels: tuple[str, ...] = ("1", "2", "3"),
              exclude_file_tasks: bool = True) -> list[Case]:
    import pandas as pd
    p = DATA_DIR / "gaia_val.parquet"
    if not p.exists():
        raise FileNotFoundError(f"缺 {p}，先看 evals/README.md 拉数据")
    df = pd.read_parquet(p)
    out: list[Case] = []
    for i, row in df.iterrows():
        lvl = str(row["Level"])
        if lvl not in levels:
            continue
        q = str(row["Question"]).strip()
        fn = row.get("file_name")
        # 双重过滤：有 file_name 字段的，以及题干里明确提到附件的
        if exclude_file_tasks:
            if fn is not None and str(fn).strip() not in ("", "None", "nan"):
                continue
            if _FILE_HINT.search(q):
                continue
        out.append(Case(f"GAIA-{i:03d}", "gaia", lvl, q, row["Final answer"]))
    return out[:limit] if limit else out


def load_bcz(limit: int = 0, topics: tuple[str, ...] = ()) -> list[Case]:
    import pandas as pd
    p = DATA_DIR / "bcz_plain.parquet"
    if not p.exists():
        raise FileNotFoundError(f"缺 {p}（需先解密），见 evals/README.md")
    df = pd.read_parquet(p)
    out: list[Case] = []
    for i, row in df.iterrows():
        topic = str(row["Topic"])
        if topics and topic not in topics:
            continue
        out.append(Case(f"BCZ-{i:03d}", "bcz", topic,
                        str(row["Question"]).strip(), row["Answer"]))
    return out[:limit] if limit else out


# ════════════════════════════════════════════════════════════════════
#                              判分
# ════════════════════════════════════════════════════════════════════
_PUNCT = re.compile(r"[\s,，。、；;：:！!？?（）()\[\]【】\"'`*_#\-—~]+")


def _norm(s: str) -> str:
    return _PUNCT.sub("", str(s)).lower()


def judge(answer_text: str, gold: str) -> bool:
    """宽松 exact-match：标准答案（归一化后）出现在模型答案里即算对。

    为什么用"包含"而不是严格相等：我们的 summary 模型输出的是带引用
    标注的自然语言段落（"…因此他获奖那年是 55 岁[2,5]"），不是短答案。
    强制严格相等会把全部题判错，测不出 arm 差异。

    对纯数字答案额外做边界检查 —— 否则 gold="41" 会被 "2041年" 误判为对。
    """
    if not answer_text:
        return False
    g = _norm(gold)
    if not g:
        return False
    t = _norm(answer_text)

    if re.fullmatch(r"[\d.]+", g):
        # 数字答案：要求前后不是数字，避免 41 命中 2041
        return re.search(rf"(?<![\d.]){re.escape(g)}(?![\d.])", t) is not None
    return g in t


if __name__ == "__main__":
    g = load_gaia()
    b = load_bcz()
    print(f"GAIA (纯文本、无附件): {len(g)} 题")
    from collections import Counter
    print("  Level:", dict(Counter(c.level for c in g)))
    print(f"BrowseComp-ZH: {len(b)} 题")
    print("  Topic:", dict(Counter(c.level for c in b)))
    print("\n样例：")
    for c in g[:2] + b[:2]:
        print(f"  [{c.cid}] {c.question[:90]}…\n      → {c.answer}")
