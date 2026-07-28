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
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError    # 专门用来处理时区


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
        - "local": 无法解析，回退到本机本地时区
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
        for alias, tz_name in _CITY_TZ_MAP.items():
            if alias and alias in loc:
                return tz_name, alias, "location_map"

    # 3) 回退本地时区
    try:    # 根据当前url请求，获取请求地址
        with urllib.request.urlopen("https://ipinfo.io/json", timeout=2) as r:
            data = json.load(r)
        return data['timezone'], data['city'], "timezone"
    except:
        local_tz = datetime.now().astimezone().tzinfo
        # ZoneInfo 可能有 .key；系统本地 tzinfo 不一定有
        tz_name = getattr(local_tz, "key", None) or datetime.now().astimezone().tzname() or "local"
        return tz_name, "本地", "local"


def _format_answer(
    now: datetime,
    display_location: str,
    source: str,
) -> str:
    weekday_cn = f"星期{_WEEKDAY_CN[now.weekday()]}"
    dt = now.strftime("%Y-%m-%d %H:%M:%S %Z")
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
        source = "local"
        display_location = "本地"

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
