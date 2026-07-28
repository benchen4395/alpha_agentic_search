"""
03_fetch_pageviews.py
=====================

聚合 Wikimedia 中文维基（``zh``）的 hourly pageviews，把可配置时间窗（默认
近 30 天）内的所有小时级 dump 汇总成一份 ``title \\t views`` 的 TSV。

设计要点
--------

* **多日 × 小时级粒度。** 一次运行会下载 N 天 × 24 个 gzip 文件，通过
  流式解压 + 内存字典累加，即便 30 天 720 个文件也不会撑爆内存。
* **并发下载 + 礼貌退让。** 用一个小规模线程池（``max_workers``）拉取
  Wikimedia 镜像；遇到 ``429/503`` 会尊重 ``Retry-After``，并触发一个
  **全局节流窗口** 让所有线程同时退让，而不是各自互相干扰。
* **可恢复、幂等。** 每个 hourly ``.gz`` 都会落盘缓存；重跑时命中缓存
  的小时直接跳过网络，失败的小时会被记到 ``data/pageviews_failed.txt``
  等待下次补跑。
* **retry 模式不会丢历史。** ``--retry-failed`` 只重跑失败小时，然后对
  **整个 cache 目录**做一次完整重聚合，保证 ``pageviews.tsv`` 永远
  等价于"当前 cache 里所有小时的聚合结果"。

典型用法
--------

::

    # 常规：下载 + 聚合 + 写 TSV（命中缓存的小时会自动跳过下载）
    python scripts/03_fetch_pageviews.py

    # 只补跑上次失败的小时，然后对整个 cache 做全量重聚合
    python scripts/03_fetch_pageviews.py --retry-failed

每个 hourly 文件里一行的字段（空格分隔）：
``domain_code page_title count_views total_response_size``。
只保留 ``domain_code == "zh"`` 的行。
"""
from __future__ import annotations

import argparse
import gzip
import io
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from tqdm import tqdm

# Wikimedia 要求所有请求带一个可辨识的 User-Agent（详见其 User-Agent 政策：
# https://meta.wikimedia.org/wiki/User-Agent_policy），否则一律 403。
HEADERS = {
    "User-Agent": "wiki_rag/0.1 (https://github.com/benchen4395/wiki_rag; benchen4395@gmail.com) python-requests"
}

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wiki_rag.config import load_config


# ---------------------------------------------------------------- date utils
def resolve_end_date(s: str) -> datetime:
    """把 config 里的 ``end_date`` 字符串解析成 naive datetime。

    Wikimedia dump 以 UTC 为准，所以 ``"yesterday"`` 意味着"UTC 的昨天"。
    """
    if s == "yesterday":
        return (datetime.now(timezone.utc) - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    return datetime.strptime(s, "%Y-%m-%d")


def build_hour_range(end_date: datetime, days: int) -> list[datetime]:
    """把 ``[end_date - (days-1), end_date]`` 展开成扁平的小时列表。"""
    hours: list[datetime] = []
    start_day = end_date - timedelta(days=days - 1)
    for d_offset in range(days):
        day = start_day + timedelta(days=d_offset)
        for h in range(24):
            hours.append(day.replace(hour=h))
    return hours


def hourly_url(dt: datetime) -> str:
    """给定 UTC 小时，构造 Wikimedia dumps 的规范 URL。"""
    return (f"https://dumps.wikimedia.org/other/pageviews/"
            f"{dt.year}/{dt.year}-{dt.month:02d}/"
            f"pageviews-{dt.strftime('%Y%m%d')}-{dt.hour:02d}0000.gz")


def hourly_cache_path(cache_dir: Path, dt: datetime) -> Path:
    """给定 UTC 小时的本地缓存路径。"""
    return cache_dir / f"pageviews-{dt.strftime('%Y%m%d')}-{dt.hour:02d}0000.gz"


# ---------------------------------------------------------------- fetch
# 全局节流：某个线程遇到 429/503 后，把 "所有线程最早可以恢复的时间" 推到
# 这个 unix 时间戳；其他线程发新请求前会先看一眼。这样整个线程池是一起
# 退让，避免各线程互相打脸。
_throttle_until = 0.0
_throttle_lock = threading.Lock()


def _set_global_throttle(seconds: float) -> None:
    """让所有线程至少再等 ``seconds`` 秒。"""
    global _throttle_until
    with _throttle_lock:
        _throttle_until = max(_throttle_until, time.time() + seconds)


def _wait_global_throttle() -> None:
    """阻塞直到全局节流窗口结束。"""
    while True:
        with _throttle_lock:
            remaining = _throttle_until - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 5.0))


def fetch_one_hour_bytes(dt: datetime, cache_dir: Path, timeout: int = 120,
                         max_retries: int = 6) -> bytes | None:
    """下载单个 hourly gzip；命中缓存 / 遵循重试策略。

    * 缓存文件已存在且非空 → 直接返回其字节内容，不走网络。
    * 4xx（除 429 外）视为永久性错误：镜像上就没有这个文件，立刻放弃。
    * 429 / 503 优先读 ``Retry-After``，否则退避到 5 秒起步的指数退避 +
      抖动，并同时触发全局节流让所有线程一起等。
    * 网络/超时错误走普通指数退避。

    最终失败返回 ``None``，调用方把该小时记入失败列表以便下次补跑，
    而不会抛异常打断整批任务。
    """
    cache_fp = hourly_cache_path(cache_dir, dt)
    if cache_fp.exists() and cache_fp.stat().st_size > 0:
        return cache_fp.read_bytes()

    url = hourly_url(dt)
    resp = None
    last_err: Exception | None = None
    for attempt in range(max_retries):
        _wait_global_throttle()
        # 加一点随机抖动，避免 N 个线程在同一 tick 同时打服务器。
        time.sleep(random.uniform(0.0, 0.3))

        try:
            resp = requests.get(url, timeout=timeout, headers=HEADERS)
            resp.raise_for_status()
            last_err = None
            break
        except Exception as e:
            last_err = e
            r = getattr(e, "response", None)
            status = getattr(r, "status_code", None)

            # 非 429 的 4xx 属于永久错误，不再重试
            if status is not None and 400 <= status < 500 and status != 429:
                break
            if attempt >= max_retries - 1:
                break

            if status in (429, 503):
                # 优先按服务器建议等；否则走指数退避（≥5s 起步）
                retry_after = 0.0
                if r is not None:
                    ra = r.headers.get("Retry-After")
                    if ra:
                        try:
                            retry_after = float(ra)
                        except ValueError:
                            retry_after = 0.0
                base = max(retry_after, 5.0 * (2 ** attempt))
                sleep_s = base + random.uniform(0.0, 3.0)
                _set_global_throttle(sleep_s)
            else:
                sleep_s = (2 ** attempt) + random.uniform(0.0, 1.0)

            tqdm.write(f"[03] retry {attempt+1}/{max_retries-1} in {sleep_s:.1f}s "
                       f"{dt.isoformat()} : {e}")
            time.sleep(sleep_s)

    if last_err is not None or resp is None:
        tqdm.write(f"[03] fetch fail {dt.isoformat()} : {last_err}")
        return None

    # 原子落盘：先写 .tmp 再 rename，避免"下到一半崩了"留下半截文件。
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_fp.with_suffix(".gz.tmp")
    tmp.write_bytes(resp.content)
    tmp.rename(cache_fp)
    return resp.content


def aggregate_gz_bytes(blob: bytes) -> dict[str, int]:
    """流式解压单个 hourly gzip，累加 ``domain==zh`` 的浏览量。"""
    local: dict[str, int] = defaultdict(int)
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(blob)) as gf:
            for raw in gf:
                try:
                    line = raw.decode("utf-8", errors="ignore").rstrip("\n")
                except Exception:
                    continue
                parts = line.split(" ")
                if len(parts) < 3:
                    continue
                if parts[0] != "zh":
                    continue
                # dump 里的 title 用下划线替代空格；这里还原成空格，方便
                # 后续和 wikiextractor 输出的 article title 直接对齐。
                title = parts[1].replace("_", " ")
                try:
                    local[title] += int(parts[2])
                except ValueError:
                    continue
    except OSError as e:
        tqdm.write(f"[03] gz decode fail: {e}")
    return local


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--days", type=int, default=None,
                    help="覆盖 cfg.pageviews.days")
    ap.add_argument("--end-date", type=str, default=None,
                    help="覆盖 cfg.pageviews.end_date")
    ap.add_argument("--retry-failed", action="store_true",
                    help="只补跑 pageviews_failed.txt 中列出的失败小时，"
                         "随后对整个 cache_dir 做一次完整重聚合")
    ap.add_argument("--max-workers", type=int, default=None,
                    help="覆盖 cfg.pageviews.max_workers；被镜像限流时调小它")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pv_cfg = cfg["pageviews"]
    end_date_str = args.end_date or pv_cfg["end_date"]
    days = args.days or pv_cfg["days"]
    max_workers = args.max_workers or pv_cfg["max_workers"]
    cache_dir = Path(pv_cfg["cache_dir"])
    out_file: Path = cfg["paths"]["pageviews_file"]
    failed_file = out_file.parent / "pageviews_failed.txt"

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # ---- 决定本次要处理哪些小时 ----
    if args.retry_failed:
        if not failed_file.exists():
            print(f"[03] no failed list at {failed_file}, nothing to retry.")
            return
        hours = []
        for line in failed_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            hours.append(datetime.fromisoformat(line))
        print(f"[03] retry mode: {len(hours)} failed hours to re-fetch")
    else:
        end_date = resolve_end_date(end_date_str)
        hours = build_hour_range(end_date, days)
        print(f"[03] aggregating pageviews from "
              f"{hours[0].strftime('%Y-%m-%d %H')} to {hours[-1].strftime('%Y-%m-%d %H')} UTC "
              f"({len(hours)} hourly files, workers={max_workers})")

    agg: dict[str, int] = defaultdict(int)
    agg_lock = threading.Lock()
    failed_hours: list[datetime] = []
    failed_lock = threading.Lock()

    # 提前给用户看一眼"命中缓存 / 总数"，好估算预期用时。
    already_cached = sum(1 for dt in hours if hourly_cache_path(cache_dir, dt).exists()
                         and hourly_cache_path(cache_dir, dt).stat().st_size > 0)
    print(f"[03] cache hit: {already_cached}/{len(hours)} (will skip download for these)")

    def worker(dt: datetime) -> int:
        blob = fetch_one_hour_bytes(dt, cache_dir)
        if blob is None:
            with failed_lock:
                failed_hours.append(dt)
            return 0
        local = aggregate_gz_bytes(blob)
        with agg_lock:
            for k, v in local.items():
                agg[k] += v
        return len(local)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, dt) for dt in hours]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="hours"):
            pass

    print(f"[03] unique titles after aggregation: {len(agg):,}")
    print(f"[03] failed hours: {len(failed_hours)} / {len(hours)}")

    # ---- 持久化 / 清理失败列表 ----
    if failed_hours:
        failed_hours.sort()
        with open(failed_file, "w", encoding="utf-8") as f:
            f.write("# Failed hours; re-run with `--retry-failed` to fetch only these.\n")
            for dt in failed_hours:
                f.write(dt.isoformat() + "\n")
        print(f"[03] failed list -> {failed_file}")
        print(f"[03] hint: re-run with `--retry-failed` to only fetch these hours")
    else:
        if failed_file.exists():
            failed_file.unlink()
        print(f"[03] all hours OK, no failed list.")

    # ---- retry 模式：拿全量 cache 重新聚合一次 ----
    #
    # retry 模式下 `agg` 只累加了"本次补跑的那些小时"，如果直接覆盖
    # pageviews.tsv 就相当于"历史消失了"。所以此时把 agg 清空，然后对
    # cache_dir 里所有 .gz 全量重聚合一次，保证输出永远等价于"整个 cache
    # 的聚合结果"。
    if args.retry_failed:
        print(f"[03] retry-failed done, re-aggregating ALL cached .gz files ...")
        agg.clear()
        gz_files = sorted(cache_dir.glob("pageviews-*.gz"))
        print(f"[03] full re-aggregate: {len(gz_files)} .gz files")

        def agg_worker(fp: Path) -> int:
            try:
                blob = fp.read_bytes()
            except Exception as e:
                tqdm.write(f"[03] read fail {fp.name}: {e}")
                return 0
            local = aggregate_gz_bytes(blob)
            with agg_lock:
                for k, v in local.items():
                    agg[k] += v
            return len(local)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(agg_worker, fp) for fp in gz_files]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="agg"):
                pass
        print(f"[03] unique titles after full re-aggregation: {len(agg):,}")

    # 排序 / 截断的工作放到 04 步做；本文件产出的是"原始聚合结果"，方便
    # 离线做别的分析或换一种排名策略。
    with open(out_file, "w", encoding="utf-8") as f:
        for title, views in agg.items():
            f.write(f"{title}\t{views}\n")
    print(f"[03] saved -> {out_file}")


if __name__ == "__main__":
    main()
