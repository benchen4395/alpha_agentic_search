# tools/weather.py
"""天气查询：使用 wttr.in（开源、免费、无需 API key）。

wttr.in 用 `?format=j1` 返回结构化 JSON，包含当前实况 + 3 天预报。

════════════════════════════════════════════════════════════════════════
lang 参数：`zh` 不是简体中文
════════════════════════════════════════════════════════════════════════
wttr.in 会把本地化描述放在 `lang_<code>` 这个**随 lang 参数变化**的 key 下。
实测（2026-08）：

    ?lang=zh     → current_condition[0] 有 key `lang_zh`，值为 "Sunny"
    ?lang=zh-cn  → current_condition[0] 有 key `lang_zh-cn`，值为 "晴"

即 `zh` 被解析为繁体/未翻译分支，**返回的仍是英文**。原实现请求
`lang=zh` 却按 `("lang_zh", "lang_zh-cn")` 的顺序取值 —— `lang_zh` 一定
存在且一定是英文，于是永远命中第一个，中文本地化**从未真正生效**，
但表现为"有值、无报错"，属于静默失效。

正确做法：请求 `lang=zh-cn`，并且**不硬编码 key 名** ——
用 `_pick_localized()` 动态找 `lang_*`，这样以后换语言不用改代码。
（响应里还有个恒定的 `lang_xx`，值与 `lang_<code>` 相同，作为兜底。）

════════════════════════════════════════════════════════════════════════
observation_time 是 UTC，不是当地时间
════════════════════════════════════════════════════════════════════════
实测北京当地 19:25 时，`observation_time` 返回 `10:48 AM`
—— 那是 UTC 11:25 之前的最近一次观测。原实现直接把这个值以
`observation_time` 之名返回，LLM 会理所当然地当成当地时间，
于是给用户"北京现在上午 10 点"的错误结论。

因为 summary prompt 有严格的时效性校验规则（见 configs/prompts.py 规则 6），
这个字段一旦被误读就会污染整个答案。所以：
  * 字段改名为 `observation_time_utc`，语义自解释；
  * 同时用 `nearest_area` 的经度换算出当地时间，另给一个
    `local_time_estimate` 字段，让模型有正确的参照。
"""
from __future__ import annotations

import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from ._http import ToolHTTPError, get_json


_BASE = "https://wttr.in"

# 预报天数上限：wttr.in 的 j1 格式固定返回 3 天（含今天）
_MAX_FORECAST_DAYS = 3


def _pick_localized(block: dict, fallback_key: str) -> str:
    """取本地化文本，找不到就回退到英文原文。

    不硬编码 `lang_zh-cn`：wttr.in 的本地化 key 名跟随请求的 lang 参数变化
    （`lang=ja` → `lang_ja`）。动态扫描 `lang_` 前缀让本函数对语言无感，
    以后要支持多语言只需改传入的 lang，无需改解析逻辑。

    优先级：具体语言 key（如 `lang_zh-cn`）> 通用 `lang_xx` > 英文原文。
    `lang_xx` 恒存在且与具体语言 key 同值，但把它放在后面更稳妥。
    """
    specific: Optional[str] = None
    generic: Optional[str] = None

    for key, val in block.items():
        if not key.startswith("lang_") or not isinstance(val, list) or not val:
            continue
        text = (val[0] or {}).get("value", "") if isinstance(val[0], dict) else ""
        if not text:
            continue
        if key == "lang_xx":
            generic = text
        else:
            specific = text

    if specific:
        return specific
    if generic:
        return generic
    raw = block.get(fallback_key)
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw[0].get("value", "")
    return ""


def _first(block: Any) -> dict:
    """安全取 wttr.in 那些"单元素列表"字段的第一项。"""
    if isinstance(block, list) and block and isinstance(block[0], dict):
        return block[0]
    return {}


def _nested_value(block: dict, key: str) -> str:
    """取 `{"areaName": [{"value": "Beijing"}]}` 这种嵌套结构里的值。"""
    return (_first(block.get(key)) or {}).get("value", "")


def _local_time_estimate(longitude: str) -> str:
    """用经度估算当地时间（wttr.in 不返回时区）。

    每 15° 经度 ≈ 1 小时时差。这是**估算**，不考虑夏令时与行政时区边界
    （例如中国全境用东八区，新疆的经度算出来会偏早两小时），所以字段名
    带 `_estimate` 且注明口径 —— 它的用途是给 LLM 一个"大致是白天还是
    夜里、大约几点"的参照，防止它把 UTC 观测时间当成当地时间，
    而不是充当精确时钟。

    需要精确当地时间时应当用 `tools/current_time.py`（IANA 时区库）。
    """
    try:
        offset_hours = round(float(longitude) / 15.0)
    except (TypeError, ValueError):
        return ""
    # 限制在 UTC-12 ~ UTC+14（真实时区范围），避免脏数据算出荒谬偏移
    offset_hours = max(-12, min(offset_hours, 14))
    local = datetime.now(timezone(timedelta(hours=offset_hours)))
    sign = "+" if offset_hours >= 0 else "-"
    return f"{local.strftime('%Y-%m-%d %H:%M')} (UTC{sign}{abs(offset_hours)})"


def _format_forecast(days: list, limit: int) -> list[dict]:
    """整理逐日预报。

    为什么要带上预报而不只给实况：用户问"明天要不要带伞""这周末热不热"
    时，只有实况的话 LLM 只能回答"我不知道"或（更糟）拿今天的温度硬编。
    j1 响应里已经包含 3 天预报，**零额外请求成本**，不用是浪费。
    """
    out: list[dict] = []
    for day in days[:limit]:
        if not isinstance(day, dict):
            continue
        astronomy = _first(day.get("astronomy"))
        # 取正午时段（hourly 里 time="1200"）的天气描述作为全天代表：
        # 比拿 00:00 的（多半是"晴"因为夜间无云判定）更能代表白天体感。
        noon_desc = ""
        chance_of_rain = ""
        for hour in day.get("hourly") or []:
            if isinstance(hour, dict) and hour.get("time") == "1200":
                noon_desc = _pick_localized(hour, "weatherDesc")
                chance_of_rain = hour.get("chanceofrain", "")
                break
        out.append({
            "date":            day.get("date"),
            "max_temp_c":      day.get("maxtempC"),
            "min_temp_c":      day.get("mintempC"),
            "avg_temp_c":      day.get("avgtempC"),
            "weather":         noon_desc,
            "chance_of_rain":  chance_of_rain,
            "uv_index":        day.get("uvIndex"),
            "total_snow_cm":   day.get("totalSnow_cm"),
            "sunrise":         astronomy.get("sunrise"),
            "sunset":          astronomy.get("sunset"),
        })
    return out


def get_weather(city: str, forecast_days: int = 2) -> dict:
    """查询城市天气（当前实况 + 逐日预报）。

    Args:
        city:          城市名（中文 / 英文 / 拼音皆可），如 '北京' / 'Beijing'。
                       也支持机场码（'PEK'）、经纬度（'39.9,116.4'）。
        forecast_days: 附带几天的预报（0~3；默认 2 = 今天 + 明天）。
                       传 0 只返回实况。

    Returns:
        {
          "city", "resolved_area", "country", "region",
          "temp_c", "feels_like_c", "weather", "weather_en",
          "humidity", "wind", "wind_degree", "pressure_mb",
          "visibility_km", "uv_index", "cloud_cover", "precip_mm",
          "observation_time_utc",   # ⚠️ UTC，非当地时间
          "local_time_estimate",    # 按经度估算的当地时间
          "latitude", "longitude",
          "forecast": [{date, max_temp_c, min_temp_c, weather, ...}],
          "source": "wttr.in",
        }

    Raises:
        ValueError:    city 为空。
        ToolHTTPError: 城市不存在（wttr.in 返回 500 "location not found"）、
                       网络故障、或响应不含天气数据。

    ⚠️ 失败时**抛异常而不是返回 `{"error": ...}`**，让 `call_tool()`
       归类为 exec_error → agent 降级到通用检索。返回 error dict 虽然也
       能被 `call_tool` 识别，但抛异常与 arxiv 那种"返回 list"的工具
       口径一致，也不会因为将来改了返回结构就漏掉。
    """
    city = (city or "").strip()
    if not city:
        raise ValueError("city 不能为空")

    try:
        forecast_days = int(forecast_days)
    except (TypeError, ValueError):
        forecast_days = 2
    forecast_days = max(0, min(forecast_days, _MAX_FORECAST_DAYS))

    # lang=zh-cn 才是简体中文（`zh` 返回英文，详见模块 docstring）
    url = f"{_BASE}/{urllib.parse.quote(city)}"
    data = get_json(
        url,
        params={"format": "j1", "lang": "zh-cn"},
        timeout=15,
        label=f"wttr.in({city})",
    )

    cur = _first(data.get("current_condition"))
    if not cur:
        # 有结构但没实况：通常是城市名解析到了一个无观测数据的坐标。
        # 必须显式失败，否则下面所有字段都是 None，LLM 会拿一堆 null 硬编。
        raise ToolHTTPError(f"wttr.in 未返回 {city!r} 的天气数据（城市名可能无法解析）")

    area = _first(data.get("nearest_area"))
    longitude = area.get("longitude", "")

    result = {
        "city":                 city,
        "resolved_area":        _nested_value(area, "areaName") or city,
        "country":              _nested_value(area, "country"),
        "region":               _nested_value(area, "region"),

        "temp_c":               cur.get("temp_C"),
        "feels_like_c":         cur.get("FeelsLikeC"),
        # 中文描述（lang_zh-cn），拿不到时自动回退英文
        "weather":              _pick_localized(cur, "weatherDesc"),
        # 同时给英文原文：便于日志排查与"中文翻译不准"时的交叉验证
        "weather_en":           (_first(cur.get("weatherDesc")) or {}).get("value", ""),

        "humidity":             cur.get("humidity"),
        "wind":                 (
            f"{cur.get('windspeedKmph', '?')} km/h "
            f"{cur.get('winddir16Point', '')}"
        ).strip(),
        "wind_degree":          cur.get("winddirDegree"),
        "pressure_mb":          cur.get("pressure"),
        "visibility_km":        cur.get("visibility"),
        "uv_index":             cur.get("uvIndex"),
        "cloud_cover":          cur.get("cloudcover"),
        "precip_mm":            cur.get("precipMM"),

        # ⚠️ 字段名显式带 _utc：这个值是 UTC，误读会直接导致答案错误
        # （实测北京当地 19:25 时该字段为 "10:48 AM"）
        "observation_time_utc": cur.get("observation_time"),
        "local_time_estimate":  _local_time_estimate(longitude),

        "latitude":             area.get("latitude"),
        "longitude":            longitude,
        "source":               "wttr.in",
    }

    if forecast_days:
        result["forecast"] = _format_forecast(
            data.get("weather") or [], forecast_days
        )
    return result
