# tools/weather.py
"""天气查询：使用 wttr.in（开源、免费、无需 API key）。

wttr.in 支持多语言、多格式；这里用 ?format=j1 直接拿 JSON。
中文城市名也支持，但 URL 需要 percent-encode。
"""
from __future__ import annotations

import urllib.parse
import requests

from configs import config


def get_weather(city: str) -> dict:
    """查询城市当前天气。

    返回示例：
      {"city": "北京", "temp_c": "22", "weather": "Sunny",
       "humidity": "45", "wind": "10 km/h NE", "feels_like_c": "21",
       "observation_time": "10:00 AM", "source": "wttr.in"}
    """
    if not city:
        return {"error": "city 不能为空"}

    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1&lang=zh"
    try:
        proxies = (
            {"http": config.SEARCH_PROXY, "https": config.SEARCH_PROXY}
            if config.DEFAULT_USE_PROXY and config.SEARCH_PROXY else None
        )
        r = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "curl/8.0"},  # wttr.in 对默认 UA 比较友好
            proxies=proxies,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"error": f"wttr.in 请求失败: {e}"}

    cur = (data.get("current_condition") or [{}])[0]
    area_block = (data.get("nearest_area") or [{}])[0]
    area_name = ""
    if area_block.get("areaName"):
        area_name = area_block["areaName"][0].get("value", "")

    # 中文天气描述（lang=zh 时存在）
    weather_desc_zh = ""
    for k in ("lang_zh", "lang_zh-cn"):
        if cur.get(k):
            weather_desc_zh = cur[k][0].get("value", "")
            break
    weather_desc_en = (cur.get("weatherDesc") or [{}])[0].get("value", "")

    return {
        "city":             city,
        "resolved_area":    area_name or city,
        "temp_c":           cur.get("temp_C"),
        "feels_like_c":     cur.get("FeelsLikeC"),
        "weather":          weather_desc_zh or weather_desc_en,
        "humidity":         cur.get("humidity"),
        "wind":             f"{cur.get('windspeedKmph','?')} km/h {cur.get('winddir16Point','')}".strip(),
        "observation_time": cur.get("observation_time"),
        "source":           "wttr.in",
    }
