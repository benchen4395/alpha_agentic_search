# context_provider.py
"""为 Agent 提供"此时此地"的事实性上下文。"""
from datetime import datetime
import os, json, urllib.request

def get_time_context() -> str:
    now = datetime.now().astimezone()
    return f"当前日期时间：{now.strftime('%Y-%m-%d %H:%M:%S %Z')}（星期{'一二三四五六日'[now.weekday()]}）"

def get_location_context() -> str:
    # 优先读用户配置的城市（最稳）
    city = os.environ.get("USER_CITY")
    if city:
        return f"用户所在城市：{city}"
    # 兜底：IP 定位
    try:
        with urllib.request.urlopen("https://ipinfo.io/json", timeout=2) as r:
            data = json.load(r)
            return f"用户所在城市（IP 推测）：{data.get('city','')}, {data.get('region','')}, {data.get('country','')}"
    except Exception:
        return "用户所在城市：未知"

def build_context_block() -> str:
    return "[环境信息]\n" + "\n".join([get_time_context(), get_location_context()])