"""
05_build_chunks_and_embed.py
============================

把 filtered 后的每篇文章切成 chunk，并用 BGE-M3 在"多 producer / 多 consumer"
流水线里做编码，最终产出两份**行号严格对齐**的文件：
输出：
* ``data/wiki_zh_chunks.jsonl`` —— 一行一个 chunk，按 ``(article_rank,
  chunk_idx)`` 升序。
* ``data/wiki_zh_emb.npy``       —— ``float32``、已 L2 归一化，第 *i* 行
  就是 JSONL 第 *i* 行 chunk 的向量。

架构总览
--------
    Producer × N（CPU）                            Consumer × M（GPU / MPS / CPU）
    +--------------------------+                   +--------------------------------+
    | 读 filtered.jsonl 切片   |                   | 从 task_queue 抢一个           |
    | 段落切分 → 滑窗切 chunk  |   task_queue      |  (global_offset, texts)        |
    | 原子领 global_offset     |  ============>    | 调 BGE-M3 encode + L2 归一化   |
    | 追加写自己的 part.jsonl  |  (~256 条/包)     | 写入 emb_tmp[g_off:g_off+n]    |
    +--------------------------+                   | 不同段并发写不冲突              |
                                                   | 结束时 put ("__done__", ...)   |
                                                   +--------------------------------+
流水线并行度：
  - N 个 producer 并行读+切分（CPU-bound，小任务）
  - M 个 consumer 并行编码（每个绑一张 GPU，GPU-bound，是真正瓶颈）
  - 通过全局原子计数器 global_counter 保证不同 producer 领到的 offset 段不重叠
"""

from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wiki_rag.chunker import chunk_paragraphs, split_paragraphs
from wiki_rag.config import load_config


SENTINEL = None   # consumer 收工信号


# ============================================================ device util
def _detect_devices(device_cfg: str, override_num_gpus: int | None) -> list[str]:
    """
    根据 config 的 embedder.device + CLI --num-gpus 决定 consumer 列表。
    行为：
      device="auto":
        - CUDA 可用 → 所有 GPU（或 --num-gpus 限制）
        - MPS  可用 → ["mps"]
        - 否则       → ["cpu"]
      device="cuda":     同 auto 的 CUDA 分支
      device="cuda:N":   只用指定卡
      device="mps"/"cpu": 单 consumer
    """
    dev = device_cfg
    if dev == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                if override_num_gpus is not None:
                    n = min(n, max(1, override_num_gpus))
                return [f"cuda:{i}" for i in range(n)]
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return ["mps"]
        except ImportError:
            pass
        return ["cpu"]

    if dev.startswith("cuda"):
        if dev == "cuda":
            try:
                import torch
                n = torch.cuda.device_count()
                if override_num_gpus is not None:
                    n = min(n, max(1, override_num_gpus))
                return [f"cuda:{i}" for i in range(max(1, n))]
            except ImportError:
                return ["cuda:0"]
        return [dev]

    return [dev]


# ============================================================ 1. Producer
def producer_worker(worker_id: int,     # 3
                    src_file: str,      # "data/filtered_articles.jsonl"
                    line_range: tuple[int, int],    # (32601, 43468)
                    part_file: str,     # "data/wiki_zh_chunks.part2.jsonl"
                    chunker_cfg: dict,
                    super_batch: int,   # 256
                    global_counter,           # mp.Value('q')
                    task_queue: "mp.Queue"):
    """
    单个 producer 负责 filtered.jsonl 的 [start, end) 行。
    对每篇文章切成若干 chunk，攒够 super_batch 就：
      1) 原子领一段 global_offset（保证跨 producer 的 offset 不重叠）
      2) 把这批 chunk 的元信息写到自己的 part.jsonl，每行包含：
           article_rank \t chunk_idx \t global_offset \t json_line
         其中 article_rank = 文章在 filtered.jsonl 的行号（04 保证 = pageviews rank）
      3) 把 (global_offset, texts) 塞进 task_queue 供 consumer 编码
    """
    start, end = line_range

    texts_buf: list[str] = []            # 待编码文本
    metas_buf: list[tuple[int, int, dict]] = []  # (article_rank, chunk_idx, meta_dict)

    def flush():
        """把 buf 里的 batch flush 出去：领 offset + 写 part.jsonl + 入队。"""
        if not texts_buf:
            return
        n = len(texts_buf)
        # 原子领号：保证不同 producer 拿到的 [g_off, g_off+n) 不重叠
        with global_counter.get_lock():
            g_off = global_counter.value
            global_counter.value = g_off + n

        # 追加写 part.jsonl；每行 4 字段用 \t 分隔，主进程 merge 时用来排序 + 重排 emb
        with open(part_file, "a", encoding="utf-8") as fout:
            for i, (a_rank, c_idx, meta) in enumerate(metas_buf):
                fout.write(
                    f"{a_rank}\t{c_idx}\t{g_off + i}\t"
                    f"{json.dumps(meta, ensure_ascii=False)}\n"
                )

        # 推给 consumer；consumer 拿到 g_off 就能直接写 emb[g_off:g_off+n]
        # 注意：mp.Queue.put 是异步的（feeder 线程延迟序列化），
        # 如果这里直接把 texts_buf 引用送出去然后 clear()，
        # 存在 feeder 还没 pickle 就被清空 -> consumer 收到空 list 的风险。
        # 所以先浅拷贝一份再 put，本地 buf 用切片重建，避免共享引用。
        task_queue.put((g_off, list(texts_buf)))
        texts_buf[:] = []
        metas_buf[:] = []

    # 每次任务开始前把 part 文件清空（覆盖历史残留）
    open(part_file, "w").close()

    with open(src_file, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if i < start:
                continue
            if i >= end:
                break
            try:
                art = json.loads(line)
            except json.JSONDecodeError:
                continue
            # 分段 + 滑窗切 chunk
            paras = split_paragraphs(art.get("text", ""),       # article实际文本内容
                                     min_para_len=chunker_cfg["min_para_len"])  # 10
            chunks = chunk_paragraphs(paras,
                                      chunk_size=chunker_cfg["chunk_size"],     # 512
                                      overlap=chunker_cfg["overlap"])           # 64
            # article_rank 就是 i（filtered.jsonl 行号，04 已保证按 rank 升序）
            for k, ck in enumerate(chunks):
                # 跳过空/纯空白 chunk，避免把空文本送进 encoder
                if not ck or not ck.strip():
                    continue
                metas_buf.append((
                    i,                                   # article_rank, 按revid倒排后的顺序编码
                    k,                                   # chunk_idx within article，当前文档的第k个chunk
                    {
                        "chunk_id": f'{art["id"]}_{k}',  # article_id 的第k个chunk
                        "doc_id":   art["id"],           # article_id
                        "title":    art["title"],        # article_title
                        "text":     ck,                  # 第k个chunk的内容
                    },
                ))
                texts_buf.append(ck)
                if len(texts_buf) >= super_batch:
                    flush()
        flush()


# ============================================================ 2. Consumer
def consumer_worker(consumer_id: int,
                    device: str,
                    emb_file: str,
                    dim: int,
                    total_chunks_hint: int,
                    embedder_cfg: dict,
                    task_queue: "mp.Queue",
                    progress_queue: "mp.Queue"):
    """
    每个 consumer 绑一张 GPU（或 MPS/CPU），循环从 task_queue 取任务：
      - 收到 SENTINEL(None) 就退出
      - 否则解出 (g_off, texts)，调 bge-m3 编码 + L2 归一化
      - 直接写 emb_tmp[g_off:g_off+len(texts)]（不同段并发写不冲突）
    """
    # 关键：多进程多卡最稳的做法是 —— 在 import torch 之前，
    # 用 CUDA_VISIBLE_DEVICES 让子进程"只看得见"分配给它那张卡。
    device_for_model = device
    if device.startswith("cuda:"):
        gpu_id = device.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        # 子进程视角下 device_count == 1，用 "cuda" 就是那唯一一张卡
        device_for_model = "cuda"

    from wiki_rag.embedder import get_model   # 延迟到子进程再 import torch

    model = get_model(embedder_cfg["model_name"],   # "BAAI/bge-m3"
                      use_fp16=embedder_cfg["use_fp16"],            # true
                      device=device_for_model)

    # 打开共享 memmap（主进程 open_memmap 预分配好，我们只做写入）
    emb_mmap = np.load(emb_file, mmap_mode="r+")
    assert emb_mmap.shape == (total_chunks_hint, dim), \
        f"emb mmap shape mismatch: got {emb_mmap.shape}"

    encoded = 0
    while True:
        item = task_queue.get()
        if item is SENTINEL:
            break
        g_off, texts = item
        # 防御：万一收到空 batch，直接跳过。
        # FlagEmbedding 1.2.x 的 encode([]) 会在最后 np.concatenate([]) 上崩掉，
        # 报 "need at least one array to concatenate"。
        if not texts:
            continue
        vecs = model.encode(
            texts,
            batch_size=embedder_cfg["batch_size"],
            max_length=embedder_cfg["max_length"],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"]
        vecs = np.asarray(vecs, dtype="float32")
        # L2 归一化：为后续 IP 检索等价于 cosine
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        emb_mmap[g_off: g_off + len(texts)] = vecs
        encoded += len(texts)
        progress_queue.put(len(texts))

    emb_mmap.flush()
    # 用 tuple 标记本 consumer 结束，主进程借此知道所有 consumer 都收工
    progress_queue.put(("__done__", consumer_id, encoded))


# ============================================================ 3. utils
def count_lines(fp: Path) -> int:
    """快速数一遍行数，用来切 producer 分片。"""
    n = 0
    with open(fp, "rb") as f:
        for _ in f:
            n += 1
    return n


def split_line_ranges(total: int, n: int) -> list[tuple[int, int]]:
    """
    把 [0, total) 均匀切成 n 段。返回 [(start, end), ...] 左闭右开。
    注意：这样每个 producer 分到的是"连续一段 filtered.jsonl"，即
    "一段热度区间的文章"，方便调试观察。
    """
    step = (total + n - 1) // n
    out = []
    for i in range(n):
        s = i * step
        e = min((i + 1) * step, total)
        if s >= e:
            break
        out.append((s, e))
    return out


# ============================================================ 4. merge & reorder
def merge_and_reorder(part_files: list[Path],
                      chunks_final_file: Path,
                      emb_tmp_file: Path,
                      emb_final_file: Path,
                      dim: int,
                      total_chunks: int,
                      copy_batch: int = 8192):
    """
    做两件事，都是主进程完成，consumer 已全部退出：

    1) 读所有 part.jsonl，拿到 list[(article_rank, chunk_idx, global_offset, json_line)]
       按 (article_rank, chunk_idx) 升序排序 → 这就是最终 chunks.jsonl 的写出顺序
       同时构造 permutation[new_row] = old_global_offset

    2) 按 permutation 把 emb_tmp 里的向量搬到 emb_final：
         emb_final[new_row] = emb_tmp[old_global_offset]
       为了内存友好，分批（copy_batch 行一批）读写；即便 650 万 × 1024 float32
       也只占瞬时 ~30 MB 内存

    结果：
       chunks.jsonl 第 i 行  ⇄  emb.npy 第 i 行
       两者都按 (article_rank, chunk_idx) 升序
    """
    print("[05] merging chunk parts (sort by article_rank, chunk_idx) ...")
    entries: list[tuple[int, int, int, str]] = []
    for pf in part_files:
        if not pf.exists():
            continue
        with open(pf, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    a_rank_str, c_idx_str, g_off_str, jsonl = line.split("\t", 3)
                    entries.append((int(a_rank_str), int(c_idx_str),
                                    int(g_off_str), jsonl))
                except ValueError:
                    continue

    # 按 (article_rank, chunk_idx) 升序 —— 这就是最终顺序
    entries.sort(key=lambda x: (x[0], x[1]))

    # 简单校验：条数、offset 覆盖 [0..total-1]
    if len(entries) != total_chunks:
        print(f"[05] warn: parts entries={len(entries)} vs "
              f"produced chunks={total_chunks}")
    if entries:
        offs = sorted(e[2] for e in entries)
        if offs[0] != 0 or offs[-1] != len(entries) - 1:
            print(f"[05] warn: offset range [{offs[0]}..{offs[-1]}] "
                  f"vs [0..{len(entries)-1}]")

    # ---- 写 chunks.jsonl（顺序 = entries 顺序）----
    chunks_final_file.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_final_file, "w", encoding="utf-8") as fout:
        for _, _, _, jsonl in entries:
            fout.write(jsonl + "\n")
    print(f"[05] saved chunks -> {chunks_final_file}")

    # ---- 重排 emb ----
    print("[05] reordering emb.npy by (article_rank, chunk_idx) ...")
    perm = np.fromiter((e[2] for e in entries), dtype=np.int64, count=len(entries))
    # 释放 entries 内存（后面不再用 json_line 了）
    del entries

    src = np.load(emb_tmp_file, mmap_mode="r")   # (hint_upper_bound, dim) float32
    real_n = perm.shape[0]

    # 直接 np.save 会一次性写整块；用 open_memmap 分块写，内存友好
    emb_final_file.parent.mkdir(parents=True, exist_ok=True)
    # 用一个临时 memmap 作为最终存储，最后 rename 成 emb.npy
    reordered_tmp = emb_final_file.with_suffix(".reorder.tmp.npy")
    dst = np.lib.format.open_memmap(
        reordered_tmp, mode="w+", dtype="float32", shape=(real_n, dim))

    # 分批 gather：dst[i:i+B] = src[perm[i:i+B]]
    # numpy 的花式索引 src[perm_slice] 会一次性把这一批读进内存，
    # 所以 copy_batch 控制瞬时内存（B × dim × 4B，例如 8192×1024×4 ≈ 32 MB）
    for i in tqdm(range(0, real_n, copy_batch), desc="reorder emb"):
        j = min(i + copy_batch, real_n)
        dst[i:j] = src[perm[i:j]]
    dst.flush()
    del dst, src

    # 把 reordered_tmp（.npy 文件）替换成正式的 emb.npy
    if emb_final_file.exists():
        emb_final_file.unlink()
    reordered_tmp.rename(emb_final_file)
    print(f"[05] saved emb ({real_n}, {dim}) -> {emb_final_file}")

    # 清理 tmp 和 part 文件
    if emb_tmp_file.exists():
        emb_tmp_file.unlink()
    for pf in part_files:
        pf.unlink(missing_ok=True)


# ============================================================ 5. main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--num-producers", type=int, default=None,
                    help="CPU 侧 producer 进程数（切 chunk 用）")
    ap.add_argument("--num-gpus", type=int, default=None,
                    help="限制使用的 GPU 数（默认 = 检测到的全部）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    e_cfg = cfg["embedder"]
    chunker_cfg = cfg["chunker"]

    src_file: Path = cfg["paths"]["filtered_file"]  # "data/filtered_articles.jsonl" 
    chunks_file: Path = cfg["paths"]["chunks_file"] # "data/wiki_zh_chunks.jsonl"
    emb_file: Path = cfg["paths"]["emb_file"]       # "data/wiki_zh_emb.npy"

    n_producers = args.num_producers or e_cfg["num_producers"]  # 8, 从 filtered_articles.jsonl 读+切+组 batch 的进程数（CPU 密集）
    super_batch = e_cfg["super_batch"]         # 256, 一次性组的 super-batch 大小（越大越好，但受显存限制）
    queue_size = e_cfg["queue_size"]           # producer/consumer 队列缓冲的 batch 数
    dim = e_cfg["dim"]

    # -------- 1. 决定 consumer 设备（auto 逻辑保持不变）--------
    devices = _detect_devices(e_cfg["device"], args.num_gpus)
    print(f"[05] compute device is: {devices}")
    n_consumers = len(devices)
    print(f"[05] consumers ({n_consumers}) devices: {devices}")

    # -------- 2. 统计 filtered.jsonl 行数并切 producer 分片 --------
    print(f"[05] counting lines in {src_file} ...")
    total_articles = count_lines(src_file)  # 86929, 10w文档，去除一些重复和内容为空的情况
    print(f"[05] total articles: {total_articles:,}")
    # ranges = [(0, 10867),(10867, 21734),(21734, 32601),(32601, 43468),(43468, 54335),(54335, 65202),(65202, 76069),(76069, 86929)]
    ranges = split_line_ranges(total_articles, n_producers)
    n_producers = len(ranges)
    print(f"[05] producers ({n_producers}) ranges: {ranges}")

    # -------- 3. 预分配 emb_tmp memmap --------
    # numpy memmap 必须一次性给定 shape，不能中途扩容。
    # 我们不知道最终会产生多少 chunk（取决于每篇文章的正文长度 + chunk_size），
    # 所以先按"乐观上限"分配一个足够大的 memmap，跑完后再裁到真实长度。
    #
    # est_chunks_per_article：每篇文章的 chunk 数估计上限（不是精确平均！）
    #   - 中文维基条目大多短小，实测平均 3~8 chunk/篇
    #   - 这里用 20 是"3~4 倍余量"的保守估计，防止溢出
    #   - 只影响临时文件大小，不影响最终 emb.npy 大小
    #
    # 磁盘临时开销 = total_articles × est_chunks_per_article × dim × 4B
    #   例：10w 文章 × 20 × 1024 × 4B ≈ 8 GB     (emb.tmp.npy)
    #      130w 文章 × 20 × 1024 × 4B ≈ 100 GB  ← 全量维基时留意磁盘
    #
    # 如果你把 chunk_size 调很小（比如 128）导致每篇切出更多 chunk，
    # 请把 est_chunks_per_article 调大（40~50）避免下标越界。
    est_chunks_per_article = 20
    total_chunks_hint = max(1, total_articles * est_chunks_per_article)
    print(f"[05] emb hint (upper bound): {total_chunks_hint:,} × {dim} "
          f"(will be trimmed to real length after encoding)")

    # 创建文件夹路径，parents=True：自动创建多级父目录；exist_ok=True：目标文件夹已经存在时不报错，静默跳过；
    emb_file.parent.mkdir(parents=True, exist_ok=True)
    # 创建一个新的文件路径，"data/wiki_zh_emb.npy" -> "data/wiki_zh_emb.tmp.npy"
    tmp_emb_file = emb_file.with_suffix(".tmp.npy") 
    # ①：在磁盘上"创建并预分配"一个 "data/wiki_zh_emb.tmp.npy" 文件，并预留 10w*dim*4B字节的空间
    emb_mmap = np.lib.format.open_memmap(
        tmp_emb_file, mode="w+", dtype="float32", shape=(total_chunks_hint, dim))
    # ②：释放主进程对这个 memmap 的 Python 引用
    del emb_mmap   # 关掉主进程句柄；consumer 各自 mmap_mode="r+" 打开

    # -------- 4. 每个 producer 独占一个 part 文件（避免锁竞争）--------
    # "data/wiki_zh_chunks.part1.jsonl"
    part_files = [chunks_file.with_suffix(f".part{w}.jsonl") for w in range(n_producers)]

    # -------- 5. 起 consumer/producer --------
    #         ┌────────────┐   ┌────────────┐   ┌────────────┐
    #         │ Producer 1 │   │ Producer 2 │   │ Producer N │   (CPU 进程)
    #         └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
    #               │                │                │
    #               │ 领 offset      │ 领 offset       │ 领 offset
    #               ▼                ▼                ▼
    #         ┌──────────────── global_counter (④) ──────────────────┐
    #         │        大家轮流 +=batch_size，拿到自己的 offset          │
    #         └───────────────────────────────────────────────────────┘
    #               │                │                │
    #               │ put(super_batch, offset)                        (②)
    #               ├────────────────┼────────────────┤
    #               ▼                ▼                ▼
    #          ┌──────────────────── task_queue ────────────────────┐
    #          │ [(offset=0, texts=[...256 条])                     │
    #          │  (offset=256, texts=[...256 条])                   │  ← 最多堆 32 包
    #          │  ...                                               │
    #          └────────────────────────────────────────────────────┘
    #                                │
    #                                ▼
    #                        ┌──────────────┐
    #                        │  Consumer    │  (GPU 进程 / 主进程)
    #                        │  bge-m3 编码  │
    #                        │  写 emb[offset:offset+256] │
    #                        └──────┬───────┘
    #                               │
    #                               ▼
    #                        ┌──────────────┐
    #                        │ progress_queue (③)  ← Producer 完事时 put(SENTINEL)
    #                        │                   Consumer 收到 N 个 SENTINEL 就退出
    #                        └──────────────┘
    ctx = mp.get_context("spawn")   # ① 配合 CUDA 必须用 spawn（macOS，  默认就是 spawn；Linux 默认还是 fork，显式设置保持一致）
    # 为何要设置两个队列: 
    #   1. task_queue: 传送待编码的大对象（几百条文本）
    #   2. progress_queue：传送小消息（进度、结束标记 SENTINEL）
    task_queue: mp.Queue = ctx.Queue(maxsize=queue_size)    # ②
    progress_queue: mp.Queue = ctx.Queue()        # ③
    global_counter = ctx.Value("q", 0)   # ④ int64 原子计数器（跨 producer 分配 offset）

    t0 = time.time()

    # 先起 consumer，让它先加载 bge-m3（几秒 ~ 几十秒），此时 producer 可以慢慢切分
    consumers = []
    for cid, dev in enumerate(devices):
        p = ctx.Process(
            target=consumer_worker,
            args=(cid, dev, str(tmp_emb_file), dim,
                  total_chunks_hint, e_cfg, task_queue, progress_queue),
        )
        p.start()
        consumers.append(p)

    producers = []
    for wid, r in enumerate(ranges):
        p = ctx.Process(
            target=producer_worker,
            args=(wid, str(src_file), r, str(part_files[wid]),
                  chunker_cfg, super_batch, global_counter, task_queue),
        )
        p.start()
        producers.append(p)

    # -------- 6. 主进程守护：进度条 + 收工 --------
    pbar = tqdm(desc="encoding", unit="chunk")

    # 等所有 producer 结束（不再有新任务入队）
    for p in producers:
        p.join()
    total_chunks = global_counter.value
    print(f"[05] producers done. total chunks = {total_chunks:,}")

    # 给每个 consumer 塞一个 SENTINEL 让它们收工
    for _ in consumers:
        task_queue.put(SENTINEL)

    # 消费 progress_queue 直到所有 consumer 报完 __done__
    consumer_encoded: dict[int, int] = {}
    while len(consumer_encoded) < n_consumers:
        msg = progress_queue.get()
        if isinstance(msg, int):
            pbar.update(msg)
        elif isinstance(msg, tuple) and len(msg) == 3 and msg[0] == "__done__":
            _, cid, enc = msg
            consumer_encoded[cid] = enc
    pbar.close()

    for p in consumers:
        p.join()

    dt = time.time() - t0
    total_encoded = sum(consumer_encoded.values())
    print(f"[05] encoded {total_encoded:,} chunks in {dt:.1f}s "
          f"(~{total_encoded / max(dt, 1e-6):.1f} chunks/s) "
          f"across {n_consumers} device(s)")
    for cid, enc in sorted(consumer_encoded.items()):
        print(f"       consumer #{cid} on {devices[cid]}: {enc:,} chunks")

    if total_encoded != total_chunks:
        print(f"[05] warn: encoded({total_encoded}) != produced({total_chunks})")

    # -------- 7. Merge + Reorder --------
    # 把并发写入的 emb_tmp（按 global_offset 乱序）重排成
    #   (article_rank, chunk_idx) 升序
    # 同时输出 chunks.jsonl；两文件行号严格对齐
    merge_and_reorder(
        part_files=part_files,          # ["data/wiki_zh_chunks.part0.jsonl", xx1.jsonl, xx2.jsonl, ...],
        chunks_final_file=chunks_file,  # "data/wiki_zh_chunks.jsonl"
        emb_tmp_file=tmp_emb_file,      # "data/wiki_zh_emb.tmp.npy"
        emb_final_file=emb_file,        # "data/wiki_zh_emb.npy"
        dim=dim,                        # 1024
        total_chunks=total_chunks,
    )

    print("[05] done.")
    print("      chunks.jsonl line i  ⇄  emb.npy row i")
    print("      order = pageviews rank 升序 + 每篇内 chunk 顺序")


if __name__ == "__main__":
    main()
