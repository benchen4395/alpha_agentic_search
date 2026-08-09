# tools/current_time.py
"""当前时间 / 日期 / 星期 / 时区工具。

用于回答：
- 现在几点？
- 今天是几号？
- 今天星期几 / 周几？
- 拉斯维加斯时间是多少？
- America/Los_Angeles 当前时间？

设计原则：
1. 时间是运行时状态，交给工具计算，而不是让模型猜。
2. 支持 location（城市/地区名）和 timezone（IANA 时区）两类输入。
3. 不联网；先用内置常用城市 → IANA 时区映射，后续可替换为 geocoding/timezonefinder。
"""
from __future__ import annotations

import re
import json
import urllib.request
import urllib.error
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError    # 专门用来处理时区

# 兜底 IP 定位的超时（秒）。工具通路对延迟敏感，这里必须给死上限：
# 失败就退本机时区，绝不阻塞主链路。
_GEO_TIMEOUT = 2


_WEEKDAY_CN = "一二三四五六日"

# 常用城市 / 地区 / 别名 → IANA timezone
# 说明：这里不是要穷举全世界，而是提供高频场景的稳定离线能力；
# 后续如需全量城市，可接入 geocoding + timezonefinder。
_CITY_TZ_MAP: dict[str, str] = {
    # 中国 / 东八区
    "中国": "Asia/Shanghai",
    "北京时间": "Asia/Shanghai",
    "北京": "Asia/Shanghai",
    "上海": "Asia/Shanghai",
    "广州": "Asia/Shanghai",
    "深圳": "Asia/Shanghai",
    "杭州": "Asia/Shanghai",
    "香港": "Asia/Hong_Kong",
    "澳门": "Asia/Macau",
    "台北": "Asia/Taipei",
    "台湾": "Asia/Taipei",
    "singapore": "Asia/Singapore",
    "新加坡": "Asia/Singapore",

    # 美国
    "拉斯维加斯": "America/Los_Angeles",
    "las vegas": "America/Los_Angeles",
    "vegas": "America/Los_Angeles",
    "洛杉矶": "America/Los_Angeles",
    "los angeles": "America/Los_Angeles",
    "旧金山": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "西雅图": "America/Los_Angeles",
    "seattle": "America/Los_Angeles",
    "纽约": "America/New_York",
    "new york": "America/New_York",
    "华盛顿": "America/New_York",
    "washington": "America/New_York",
    "芝加哥": "America/Chicago",
    "chicago": "America/Chicago",
    "达拉斯": "America/Chicago",
    "dallas": "America/Chicago",
    "丹佛": "America/Denver",
    "denver": "America/Denver",
    "凤凰城": "America/Phoenix",
    "phoenix": "America/Phoenix",
    "阿拉斯加": "America/Anchorage",
    "夏威夷": "Pacific/Honolulu",
    "hawaii": "Pacific/Honolulu",

    # 欧洲
    "伦敦": "Europe/London",
    "london": "Europe/London",
    "巴黎": "Europe/Paris",
    "paris": "Europe/Paris",
    "柏林": "Europe/Berlin",
    "berlin": "Europe/Berlin",
    "罗马": "Europe/Rome",
    "rome": "Europe/Rome",
    "马德里": "Europe/Madrid",
    "madrid": "Europe/Madrid",
    "莫斯科": "Europe/Moscow",
    "moscow": "Europe/Moscow",
    "布达佩斯": "Europe/Budapest",
    "budapest": "Europe/Budapest",
    "布加勒斯特": "Europe/Bucharest",
    "bucharest": "Europe/Bucharest",
    "阿姆斯特丹": "Europe/Amsterdam",
    "amsterdam": "Europe/Amsterdam",
    "苏黎世": "Europe/Zurich",
    "zurich": "Europe/Zurich",
    "斯德哥尔摩": "Europe/Stockholm",
    "stockholm": "Europe/Stockholm",
    "岳父": "Europe/Kyiv",
    "kyiv": "Europe/Kyiv",
    "雅典": "Europe/Athens",
    "athens": "Europe/Athens",
    "里斯本": "Europe/Lisbon",
    "lisbon": "Europe/Lisbon",
    "都柏林": "Europe/Dublin",
    "dublin": "Europe/Dublin",

    # 亚洲其他
    "东京": "Asia/Tokyo",
    "tokyo": "Asia/Tokyo",
    "日本": "Asia/Tokyo",
    "首尔": "Asia/Seoul",
    "seoul": "Asia/Seoul",
    "韩国": "Asia/Seoul",
    "曼谷": "Asia/Bangkok",
    "bangkok": "Asia/Bangkok",
    "迪拜": "Asia/Dubai",
    "dubai": "Asia/Dubai",
    "印度": "Asia/Kolkata",
    "新德里": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "孟买": "Asia/Kolkata",
    "mumbai": "Asia/Kolkata",
    "雅加达": "Asia/Jakarta",
    "jakarta": "Asia/Jakarta",
    "吉隆坡": "Asia/Kuala_Lumpur",
    "kuala lumpur": "Asia/Kuala_Lumpur",
    "马尼拉": "Asia/Manila",
    "manila": "Asia/Manila",
    "河内": "Asia/Ho_Chi_Minh",
    "胡志明市": "Asia/Ho_Chi_Minh",
    "伊斯坦布尔": "Europe/Istanbul",
    "istanbul": "Europe/Istanbul",
    "特拉维夫": "Asia/Jerusalem",
    "tel aviv": "Asia/Jerusalem",
    "耶路撒冷": "Asia/Jerusalem",
    "利雅得": "Asia/Riyadh",
    "riyadh": "Asia/Riyadh",
    "开罗": "Africa/Cairo",
    "cairo": "Africa/Cairo",
    "约翰内斯堡": "Africa/Johannesburg",
    "johannesburg": "Africa/Johannesburg",
    "内罗毕": "Africa/Nairobi",
    "nairobi": "Africa/Nairobi",
    "拉各斯": "Africa/Lagos",
    "lagos": "Africa/Lagos",

    # 美洲其他
    "多伦多": "America/Toronto",
    "toronto": "America/Toronto",
    "温哥华": "America/Vancouver",
    "vancouver": "America/Vancouver",
    "墨西哥城": "America/Mexico_City",
    "mexico city": "America/Mexico_City",
    "圣保罗": "America/Sao_Paulo",
    "sao paulo": "America/Sao_Paulo",
    "巴西": "America/Sao_Paulo",
    "布宜诺斯艾利斯": "America/Argentina/Buenos_Aires",
    "buenos aires": "America/Argentina/Buenos_Aires",

    # 大洋洲
    "悉尼": "Australia/Sydney",
    "sydney": "Australia/Sydney",
    "墨尔本": "Australia/Melbourne",
    "melbourne": "Australia/Melbourne",
    "奥克兰": "Pacific/Auckland",
    "auckland": "Pacific/Auckland",

    # 常见时区缩写（注意：缩写有歧义，仅作常用语境兜底）
    "utc": "UTC",
    "gmt": "Etc/GMT",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "est": "America/New_York",
    "edt": "America/New_York",
    "cst": "America/Chicago",
    "cdt": "America/Chicago",
}


_LOCATION_NOISE_RE = re.compile(
    r"(现在|当前|此刻|今天|今日|当地|本地|时间|日期|几号|几点|是多少|是几点|"
    r"星期几|周几|礼拜几|请问|帮我查|查一下|告诉我|的|\?|？|!|！|。|，|,)"
)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _clean_location(location: str | None) -> str:
    """把模型抽取出的 location 做轻量清洗。

    例如：
        "拉斯维加斯时间是多少" → "拉斯维加斯"
        "纽约现在几点"       → "纽约"
    """
    if not location:
        return ""
    loc = _LOCATION_NOISE_RE.sub("", location).strip()
    return _normalize_text(loc)


def _is_valid_timezone(tz_name: str | None) -> bool:
    if not tz_name:
        return False
    try:
        ZoneInfo(tz_name)
        return True
    except ZoneInfoNotFoundError:
        return False


def _geo_timezone() -> tuple[str | None, str | None]:
    """通过 IP 定位猜测时区，失败返回 (None, None)。

    只捕获**预期内**的异常（网络/解析/字段缺失），不使用裸 `except`，
    以免连 KeyboardInterrupt、SystemExit 一起吞掉。

    返回的时区名必须通过 `_is_valid_timezone` 校验：第三方接口的字段
    不可信，若直接返回，后续 `ZoneInfo(tz_name)` 会抛异常。
    """
    try:
        with urllib.request.urlopen(
            "https://ipinfo.io/json", timeout=_GEO_TIMEOUT
        ) as r:
            data = json.load(r)
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None, None

    if not isinstance(data, dict):
        return None, None
    tz = (data.get("timezone") or "").strip()
    if not _is_valid_timezone(tz):
        return None, None
    return tz, (data.get("city") or "").strip() or None


def resolve_timezone(
    location: str | None = None,
    timezone: str | None = None,
) -> tuple[str, str, str]:
    """解析 location/timezone 到 IANA 时区。

    返回：
        (tz_name, display_location, source)

    source:
        - "timezone": 用户直接给了合法 IANA timezone
        - "location_map": 通过内置城市映射命中
        - "ip_geo": 用户未给地点时，通过 IP 定位推断
        - "local": 用户未给地点且 IP 定位失败，用本机时区
        - "unresolved_location": 用户给了地点但无法解析（结果不可信，
          时间值为本机时区，调用方应视为“没查到”而非“查到了”）
    """
    if timezone:
        tz = timezone.strip()
        if _is_valid_timezone(tz):
            return tz, location or tz, "timezone"

    raw_loc = location or ""
    loc = _clean_location(raw_loc)
    if loc:
        # 1) 精确命中（展示清洗后的地点名，避免 "纽约现在几点当前时间..."）
        if loc in _CITY_TZ_MAP:
            return _CITY_TZ_MAP[loc], loc, "location_map"

        # 2) 子串命中：处理 "美国拉斯维加斯" / "las vegas 时间" 这类输入
        #
        # ⚠️ 这里**不能**对所有 alias 做裸 `alias in loc`：
        # 时区缩写（est/cst/pst/gmt/utc…）只有 3 个字母，会命中大量无关词。
        # 实测（修复前）：
        #     budapest  → est → America/New_York   ← 匈牙利跑到了纽约
        #     bucharest → est → America/New_York
        #     forest hills → est → America/New_York
        # 且 dict 迭代顺序让缩写可能先于真实城市名命中，属于静默错误答案。
        #
        # 规则：ASCII alias 必须按**单词边界**匹配；中文没有词边界概念，
        # 但中文城市名长度 ≥2 且语义唯一，裸子串是安全的。
        for alias, tz_name in _CITY_TZ_MAP.items():
            if not alias:
                continue
            if alias.isascii():
                if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", loc):
                    return tz_name, alias, "location_map"
            elif alias in loc:
                return tz_name, alias, "location_map"

    # 3) 按 IP 猜所在地时区
    #
    # ⚠️ 只在**用户完全没给地点**时才允许走这里。
    # 若用户明确问了某个地点、但我们解析不出来（如 "budapest" 不在内置表里），
    # 那么返回本机/本地 IP 的时区就是**一个自信的错误答案**：
    #     "budapest" → ip_geo → Asia/Singapore → "Singapore当前时间是…"
    # 用户问布达佩斯，系统言之凿凿地回答新加坡时间，且 location 字段还写着
    # "Singapore"，下游完全无法察觉这是猜的。宁可标记为未解析，
    # 让 agent 走降级，也不要编一个看起来很真的答案。
    #
    # 顺带：这样也避免了给主链路无谓地加一次网络往返（工具通路对延迟敏感）。
    location_requested = bool(loc or (timezone or "").strip())
    if not location_requested:
        tz_name, city = _geo_timezone()
        if tz_name:
            return tz_name, city or tz_name, "ip_geo"

    # 4) 回退本机本地时区
    local_tz = datetime.now().astimezone().tzinfo
    # 只能给 IANA key 或哨兵值 "local"，**不能给时区缩写**：
    # 早前的写法会回退到 `tzname()`（如 "CST"），而 "CST" 不是合法 IANA
    # 名字 —— `get_current_time` 里 `ZoneInfo("CST")` 会抛异常进入异常分支，
    # 把 source 重写成 "local"，于是 unresolved_location 这个信号被吞掉。
    tz_name = getattr(local_tz, "key", None) or "local"
    # 用户要过地点却没解析出来 → 用独立的 source 标出来，便于上层判断/监控
    return tz_name, "本地", "unresolved_location" if location_requested else "local"


def _format_answer(
    now: datetime,
    display_location: str,
    source: str,
) -> str:
    weekday_cn = f"星期{_WEEKDAY_CN[now.weekday()]}"
    dt = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    if source == "unresolved_location":
        # 不能拿本机时区冒充用户问的地点，必须把不确定性说清楚。
        return f"未能识别该地点的时区；以下是本地时间 {dt}，今天是{weekday_cn}。"
    if display_location and display_location != "本地":
        return f"{display_location}当前日期时间是 {dt}，今天是{weekday_cn}。"
    if source == "local":
        return f"当前本地日期时间是 {dt}，今天是{weekday_cn}。"
    return f"当前日期时间是 {dt}，今天是{weekday_cn}。"


def get_current_time(
    location: str | None = None,
    timezone: str | None = None,
) -> dict:
    """返回当前日期、时间、星期等信息。

    Args:
        location: 可选，城市/地区名，如 "北京"、"拉斯维加斯"、"New York"。
        timezone: 可选，IANA 时区，如 "Asia/Shanghai"、"America/Los_Angeles"。

    若 location/timezone 均为空，则返回本机本地时间。
    """
    tz_name, display_location, source = resolve_timezone(location, timezone)

    try:
        now = datetime.now(ZoneInfo(tz_name)) if tz_name != "local" else datetime.now().astimezone()
    except ZoneInfoNotFoundError:
        now = datetime.now().astimezone()
        tz_name = now.tzname() or "local"
        display_location = "本地"
        # 不要无条件改写成 "local"：若原因是"给了地点但解析不出"，
        # 该信号必须保留，否则上层会把本机时间当成“查到了”。
        if source != "unresolved_location":
            source = "local"

    weekday_cn = f"星期{_WEEKDAY_CN[now.weekday()]}"
    return {
        "location": display_location,
        "requested_location": location,
        "timezone": tz_name,
        "timezone_source": source,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": weekday_cn,
        "timezone_abbr": now.strftime("%Z"),
        "utc_offset": now.strftime("%z"),
        "answer": _format_answer(now, display_location, source),
    }
