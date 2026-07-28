"""
04_filter_top_articles.py
=========================

把 pageviews 的热度信息和 Wikipedia dump 的正文做连接：只保留 Top-N 最热门
的条目，并以 JSONL 落盘。两份产物都严格按热度升序，保证下游任意一步都能
继承同一个 rank 语义。

流水线
------

1. 读 ``pageviews.tsv``，按浏览量降序挑出 Top-N（同时做 OpenCC 繁→简
   归一化 + 去重）。
2. 用进程池并发扫 ``wikiextractor`` 的分片，每个 worker 对每篇命中 Top-N
   的文章输出 ``(rank, jsonl_line)``。
3. 主进程汇总 tuple，若同 rank 命中多次则保留正文最长的那条，然后按 rank
   升序落盘。

输出文件的不变量
----------------

* ``top_titles.txt``       —— 第 ``i`` 行 = rank-``i`` 的简体标题。
* ``filtered_articles.jsonl`` —— 按 rank 升序。可能比 ``top_titles.txt``
  短，因为部分热门 title 在 dump 里没有正文（重定向、消歧义占位、被
  wikiextractor 丢弃等）。

为什么用进程池
--------------

OpenCC 繁转简是 CPU-bound。先做便宜的 "title 命中" 过滤，只对命中的文章
才调用 ``to_simplified(text)`` 处理正文，整体开销正比于"最终保留的文章
数"而不是"dump 大小"。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wiki_rag.config import load_config


# ---------------------------------------------------------------- OpenCC
_CC = None


def _get_cc():
    global _CC
    if _CC is None:
        try:
            from opencc import OpenCC
            _CC = OpenCC("t2s")
        except ImportError:
            _CC = False
    return _CC


def to_simplified(s: str) -> str:
    cc = _get_cc()
    if not cc:
        return s
    return cc.convert(s)


# ---------------------------------------------------------------- title loading
def load_top_titles_ranked(pv_file: Path, top_n: int) -> tuple[list[str], dict[str, int]]:
    """对 pageviews 排序，并提供 O(1) 的 "title → rank" 查表。

    Returns:
        ordered_titles_simp: 按热度降序排列的**简体**标题列表，长度 ≤ ``top_n``。
        title_to_rank: 每个出现过的 title 变体（原始 or 简体）都映射到从 0
            开始的 rank。分片 worker 用它做 O(1) 命中判断。
    """
    print(f"[04] reading pageviews {pv_file} ...")
    pairs: list[tuple[str, int]] = []
    with open(pv_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                title, views = line.rstrip("\n").split("\t")
                pairs.append((title, int(views)))
            except ValueError:
                continue
    print(f"[04] total titles from pageviews: {len(pairs)}")
    pairs.sort(key=lambda x: x[1], reverse=True)

    ordered_titles_simp: list[str] = []
    title_to_rank: dict[str, int] = {}
    for raw_title, _views in pairs:
        simp = to_simplified(raw_title)
        if simp in title_to_rank:
            # 已经有一个更热的变体占了这个 rank；这里把当前繁体拼写也别名
            # 映射到同一个 rank，方便后面 O(1) 命中。
            title_to_rank.setdefault(raw_title, title_to_rank[simp])
            continue
        rank = len(ordered_titles_simp)
        ordered_titles_simp.append(simp)
        title_to_rank[simp] = rank
        title_to_rank[raw_title] = rank
        if len(ordered_titles_simp) >= top_n:
            break

    return ordered_titles_simp, title_to_rank


# ---------------------------------------------------------------- worker
_TITLE_TO_RANK: dict[str, int] = {}
_MIN_TEXT_LEN: int = 0


def _init_worker(title_to_rank: dict[str, int], min_text_len: int):
    global _TITLE_TO_RANK, _MIN_TEXT_LEN
    _TITLE_TO_RANK = title_to_rank
    _MIN_TEXT_LEN = min_text_len


def process_one_file(fp: str) -> list[tuple[int, str]]:
    """扫描一个 wikiextractor 分片，输出命中项 ``(rank, jsonl_line)``。"""
    out: list[tuple[int, str]] = []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    art = json.loads(line)
                except json.JSONDecodeError:
                    continue

                title = art.get("title", "")
                text = art.get("text", "")
                if not title or not text:
                    continue
                if len(text) < _MIN_TEXT_LEN:
                    continue

                title = title.strip()
                # 便宜的命中判断：先按原文匹配，未命中再转简体试一次
                rank = _TITLE_TO_RANK.get(title)
                if rank is None:
                    title_simp = to_simplified(title)
                    rank = _TITLE_TO_RANK.get(title_simp)
                    if rank is None:
                        continue
                else:
                    title_simp = to_simplified(title)

                # 确认命中后，才做昂贵的正文繁转简
                art["title"] = title_simp
                art["text"] = to_simplified(text)
                out.append((rank, json.dumps(art, ensure_ascii=False)))
    except Exception as e:
        print(f"[04] worker fail {fp}: {e}", file=sys.stderr)
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--top", type=int, default=None, help="覆盖 config 中的 top_n")
    ap.add_argument("--workers", type=int, default=None,
                    help="并行进程数，默认 = os.cpu_count()")
    args = ap.parse_args()

    cfg = load_config(args.config)
    top_n = args.top or cfg["filter"]["top_n"]
    min_text_len = cfg["filter"]["min_text_len"]

    pv_file: Path = cfg["paths"]["pageviews_file"]
    extracted: Path = cfg["paths"]["extracted_dir"]
    top_titles_out: Path = cfg["paths"]["top_titles_file"]
    filtered_out: Path = cfg["paths"]["filtered_file"]

    ordered_titles, title_to_rank = load_top_titles_ranked(pv_file, top_n)
    print(f"[04] target top titles (deduped, simplified): {len(ordered_titles)}")
    print(f"[04] title_to_rank entries (with variants): {len(title_to_rank)}")

    # 写 top_titles.txt：行号 == rank
    top_titles_out.parent.mkdir(parents=True, exist_ok=True)
    with open(top_titles_out, "w", encoding="utf-8") as f:
        for t in ordered_titles:
            f.write(t + "\n")
    print(f"[04] saved -> {top_titles_out}  (line i = rank i)")

    # 收集 extracted 目录下所有 wikiextractor 分片
    files = sorted(glob.glob(os.path.join(str(extracted), "*", "wiki_*")))
    if not files:
        print(f"[04] no wiki_* files under {extracted}, abort.")
        return
    workers = args.workers or (os.cpu_count() or 4)
    print(f"[04] scanning {len(files)} shard files with {workers} processes ...")

    # 从各分片汇总所有 (rank, line) 命中项
    hits: list[tuple[int, str]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(title_to_rank, min_text_len),
    ) as ex:
        futures = [ex.submit(process_one_file, fp) for fp in files]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="shards"):
            hits.extend(fut.result())

    # 同一 rank 偶尔会被多条命中（极少发生：dump 里有多个页面同标题），
    # 此时保留正文更长的那条。
    by_rank: dict[int, str] = {}
    dup = 0
    for rank, ln in hits:
        if rank in by_rank:
            dup += 1
            try:
                a = json.loads(by_rank[rank])["text"]
                b = json.loads(ln)["text"]
                if len(b) > len(a):
                    by_rank[rank] = ln
            except Exception:
                pass
        else:
            by_rank[rank] = ln
    if dup:
        print(f"[04] warn: {dup} duplicate-title hits, kept longest text.")

    # 按 rank 升序写出 → 与 top_titles.txt 逐行对齐
    filtered_out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    missing = 0
    with open(filtered_out, "w", encoding="utf-8") as fout:
        for rank in range(len(ordered_titles)):
            ln = by_rank.get(rank)
            if ln is None:
                # 这个热门 title 在 dump 里没有对应正文（重定向 / 消歧义占位 / 被 wikiextractor 丢弃 / 正文过短）
                missing += 1
                continue
            fout.write(ln + "\n")
            kept += 1

    print(f"[04] kept {kept} articles -> {filtered_out}")
    print(f"[04] missing (top title without body): {missing}")
    print(f"[04] both files are now sorted by pageviews rank (desc). "
          f"Note: filtered.jsonl may be shorter than top_titles.txt "
          f"because some top titles have no matching body in the dump.")


if __name__ == "__main__":
    main()
