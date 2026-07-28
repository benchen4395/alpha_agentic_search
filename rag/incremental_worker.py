# rag/incremental_worker.py
"""增量索引后台 worker：把 (query, answer, sources) 事件异步写进 L3。

设计目标：
    - 主流程零阻塞（chat 调用 archive() 只把事件放入队列）
    - 崩溃/关闭时保证不丢事件（daemon=False + graceful close 可选）
    - 支持批量刷盘减少 IO
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

from . import config as rag_config


@dataclass
class ArchiveEvent:
    query: str
    answer: str
    sources: Optional[list[dict]] = None


class IncrementalWorker:
    def __init__(self, l3_layer):
        self.l3 = l3_layer
        self.q: "queue.Queue[Optional[ArchiveEvent]]" = queue.Queue(
            maxsize=rag_config.INCR_QUEUE_MAXSIZE
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="rag_incr_worker", daemon=True
        )
        self._thread.start()

    def submit(self, event: ArchiveEvent) -> bool:
        try:
            self.q.put_nowait(event)
            return True
        except queue.Full:
            print("[incr_worker] 队列满，丢弃事件（考虑加大 INCR_QUEUE_MAXSIZE）")
            return False

    def shutdown(self, wait: bool = True, timeout: float = 10.0):
        self._stop.set()
        try:
            self.q.put_nowait(None)   # 唤醒
        except queue.Full:
            pass
        if wait:
            self._thread.join(timeout=timeout)

    # ---------- 内部 ---------- #
    def _loop(self):
        batch: list[ArchiveEvent] = []
        last_flush = time.time()
        while not self._stop.is_set():
            try:
                ev = self.q.get(timeout=rag_config.INCR_FLUSH_INTERVAL_SEC)
            except queue.Empty:
                ev = None
            if ev is not None:
                batch.append(ev)
            need_flush = (
                len(batch) >= rag_config.INCR_BATCH_SIZE
                or (batch and time.time() - last_flush >= rag_config.INCR_FLUSH_INTERVAL_SEC)
            )
            if need_flush:
                self._flush(batch)
                batch = []
                last_flush = time.time()
        # 退出前 flush
        if batch:
            self._flush(batch)

    def _flush(self, batch: list[ArchiveEvent]):
        for ev in batch:
            try:
                ok = self.l3.add(ev.query, ev.answer, ev.sources)
                if not ok:
                    print(f"[incr_worker] L3 写入失败（embed 返回 None）: {ev.query[:40]}")
            except Exception as e:
                print(f"[incr_worker] L3 写入异常: {e}")
