# context_provider.py
"""为 Agent 提供"此时此地"的事实性上下文。

这个模块解决的是 LLM 的一个根本缺陷：**它不知道今天是几号**。
训练数据有截止日期，推理时又没有时钟，所以只要问题涉及"今天/最新/当前"，
模型就只能猜 —— 而它猜的往往是训练数据里的高频日期。

实测故障（本次修复的起因）：
    提问：今日黄金价格（真实日期 2026-08-05）
    回答：「今日黄金价格（2026年4月16日）… 伦敦金现 4822.88 美元/盎司」
          └── 模型把检索到的 4 月网页当成"今日"，凭空造了个日期

根因是**只有 rewriter 阶段拿到了环境信息，summary 阶段没有**：
rewriter 的 prompt 里有 `{context}`，所以它能正确改写出"2026年8月"；
但真正写答案的 summary 阶段完全不知道今天几号，于是资料里的日期
被它当成了当前日期。见 configs/prompts.py 的 `_SUMMARY_SYSTEM_BASE`。
"""
from datetime import datetime
import os, json, threading, time, urllib.request


def get_time_context() -> str:
    """当前日期时间。**必须每次实时计算**，不能缓存。

    实测 0.1ms，纯本地调用，没有任何缓存的理由。
    ⚠️ 反面教训：如果把它和 location 一起缓存，进程跑过午夜后
    日期就会停在启动那天 —— 这正是我们要修的那类 bug 的另一种形式。
    """
    now = datetime.now().astimezone()
    return (
        f"当前日期时间：{now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        f"（星期{'一二三四五六日'[now.weekday()]}）"
    )


# ---- 地理位置缓存 ----
# 为什么要缓存：`get_location_context()` 走 ipinfo.io 的 HTTP 请求，
# **实测单次 500~1580ms**（波动大）。而它被两个阶段调用（rewriter +
# summary），每轮对话就是 1~3s 的纯等待，且这段时间完全没有产出。
#
# 为什么可以缓存：用户所在城市在一次会话里几乎不变 —— 与 time 不同，
# 它不是"每时每刻都在变"的量。TTL 兜住"用户换网络/VPN"的情况。
_LOC_TTL_SEC: float = float(os.getenv("CONTEXT_LOC_TTL", "1800"))  # 30 分钟
_loc_cache: dict = {"value": None, "ts": 0.0}
_loc_lock = threading.Lock()

# ---- in-flight 去重（单飞 / single-flight）----
# ⚠️ 这是实测发现的坑，不是理论上的洁癖：
#   Agent.__init__ 会起一个后台线程预取地理位置，但预取还没返回时
#   （IP 定位 ~500ms）首条查询就来了。由于**缓存只在成功返回后才写入**，
#   此刻缓存仍是空的，于是首条查询**又发起了一次自己的请求**，
#   实测仍然阻塞了 814ms —— 预取完全白做。
#
#   `_loc_lock` 只保护"读/写缓存"这两个瞬间，网络请求在锁外，
#   所以它防不住并发重复请求。
#
# 修法：用一个 Event 标记"已有人在路上"。后到的线程不再自己发请求，
# 而是**等那个人的结果**。这样 N 个并发调用只产生 1 次网络请求。
_loc_inflight: threading.Event | None = None
_inflight_lock = threading.Lock()


def _fetch_location_via_ip() -> str | None:
    """真正发起 IP 定位请求。成功返回字符串，失败返回 None。"""
    try:
        with urllib.request.urlopen("https://ipinfo.io/json", timeout=2) as r:
            data = json.load(r)
            return (f"用户所在城市（IP 推测）：{data.get('city','')}, "
                    f"{data.get('region','')}, {data.get('country','')}")
    except Exception:
        return None


def _read_cache() -> str | None:
    with _loc_lock:
        if (_loc_cache["value"] is not None
                and time.time() - _loc_cache["ts"] < _LOC_TTL_SEC):
            return _loc_cache["value"]
    return None


def get_location_context(use_cache: bool = True, wait: float = 3.0) -> str:
    """用户所在城市。带 TTL 缓存 + in-flight 去重。

    优先级：`USER_CITY` 环境变量 > 缓存 > IP 定位 > "未知"。
    显式配置的城市既最准也**零延迟**，强烈建议生产环境直接配上：
        export USER_CITY="北京"

    Args:
        use_cache: False 则跳过缓存强制重新获取（仅测试/诊断用）。
        wait: 当已有其它线程在获取时，最多等它多少秒。
            超时则返回"未知"而不是继续阻塞 —— 宁可少一条环境信息，
            也不能让用户的查询卡住。
    """
    # 显式配置：最稳且免费，连缓存都不需要
    city = os.environ.get("USER_CITY")
    if city:
        return f"用户所在城市：{city}"

    if use_cache:
        hit = _read_cache()
        if hit is not None:
            return hit

    # ---- in-flight 去重：决定"我去取"还是"我等别人取" ----
    global _loc_inflight
    should_fetch = False
    with _inflight_lock:
        if _loc_inflight is None:
            _loc_inflight = threading.Event()
            should_fetch = True          # 我是第一个，由我发请求
        ev = _loc_inflight

    if not should_fetch:
        # 已有人在路上 —— 等它，别重复发请求（这正是预取白做的原因）
        ev.wait(timeout=wait)
        hit = _read_cache()
        return hit if hit is not None else "用户所在城市：未知"

    try:
        out = _fetch_location_via_ip()
        if out is not None:
            with _loc_lock:
                _loc_cache["value"] = out
                _loc_cache["ts"] = time.time()
            return out
        # 失败**不缓存**：网络抽风是暂时的，缓存下来会让"未知"粘住 30 分钟
        return "用户所在城市：未知"
    finally:
        # 无论成败都要放行等待者，否则它们会一直等到 timeout
        with _inflight_lock:
            _loc_inflight = None
        ev.set()


def build_context_block(include_location: bool = True) -> str:
    """组装 `[环境信息]` 块，供 rewriter / summary 等阶段注入 prompt。

    Args:
        include_location: 是否包含地理位置。默认 True。
            置 False 可拿到**纯本地、零网络**的上下文（只有时间），
            适合对延迟敏感、又不关心地点的场景。
    """
    parts = [get_time_context()]
    if include_location:
        parts.append(get_location_context())
    return "[环境信息]\n" + "\n".join(parts)
