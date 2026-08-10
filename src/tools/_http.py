# tools/_http.py
"""工具层共用的 HTTP 客户端封装。

为什么需要这一层
----------------
`weather.py` / `arxiv.py` / `github_repo.py` 原本各自复制了一份
`_proxies()` 与裸 `requests.get(...)`：

  * 代理判定逻辑重复三份，改 config 语义时容易漏改；
  * 都是**单次请求、失败即放弃**。而这几个都是免费公共服务
    （wttr.in / export.arxiv.org），偶发 5xx、连接重置、限流是常态。
    单次失败就返回 error → agent 直接降级去跑通用检索（十几秒），
    而其实再试一次往往几百毫秒就成功了。
  * 没有 UA，部分公共服务（尤其 wttr.in）对默认的
    `python-requests/x.y` UA 有额外限流。

因此统一到 `get_json()` / `get_text()`：**指数退避重试 + 代理 + UA +
超时**一处实现，各工具只关心业务解析。

重试策略的取舍
--------------
只对「可重试」的失败重试，这点很关键：

  * 网络层异常（超时 / 连接重置 / DNS）→ 重试
  * 5xx、429（限流）                    → 重试
  * 4xx（404 城市不存在 / 400 参数错） → **立即放弃**
    这类是确定性失败，重试 100 次也是同样结果，只会白白增加延迟。

退避用 0.4s → 0.8s（`_BACKOFF_BASE * 2**i`），总附加延迟上限约 1.2s。
选这个量级是因为工具路由处在用户请求的关键路径上：重试要能吃掉
偶发抖动，但绝不能让"工具比直接搜索还慢"，否则降级反而更划算。
"""
from __future__ import annotations

import time
from typing import Any, Optional

import requests

from src.configs import config


# 公共服务对匿名 UA 往往更严格；带一个明确的 UA 既礼貌也更稳定。
# 保留 curl 前缀是因为 wttr.in 会据此返回精简响应（它对 curl 特别优化）。
_UA = "curl/8.0 (alpha-agentic-search; +https://github.com/benchen4395/alpha_agentic_search)"

_DEFAULT_TIMEOUT = 12
_MAX_ATTEMPTS = 3          # 首次 + 2 次重试
_BACKOFF_BASE = 0.4        # 0.4s → 0.8s，最坏附加 ~1.2s


class ToolHTTPError(Exception):
    """工具层 HTTP 调用失败。

    统一抛异常而不是返回 `{"error": ...}`，是为了让 `tools/__init__.py`
    的 `call_tool()` 能通过 `except Exception` 把它归类成 `exec_error`，
    进而让 `route_and_call()` 返回 `ok=False`、agent 正确降级到检索通路。

    返回错误字典的写法有个隐蔽陷阱：函数签名是 `-> list[dict]` 的工具
    （如 `search_arxiv`）只能把错误包成 `[{"error": ...}]`，而 `call_tool`
    的校验只检查「dict 且含 error」与「空列表」，**非空的错误列表会被
    判成成功** → 错误文本被当作"外部资料"喂给 LLM。抛异常从根上避免了
    这条歧路。
    """


def _proxies() -> Optional[dict]:
    """按全局配置决定是否走代理（三个工具共用同一套语义）。"""
    if config.DEFAULT_USE_PROXY and config.SEARCH_PROXY:
        return {"http": config.SEARCH_PROXY, "https": config.SEARCH_PROXY}
    return None


def _should_retry(status: int, body: str = "") -> bool:
    """判断是否值得重试。

    基本规则：5xx / 429（限流）可重试，4xx 客户端错误不可重试。

    例外：**wttr.in 用 500 表达"城市不存在"**（实测
    `GET /Xyzzynotacity?format=j1` → `500 location not found: upstream
    error: opencage: invalid response`）。这在语义上是 4xx —— 重试三次
    结果完全一样，只是白白给用户加 1.2s 延迟，还多打两次外部请求。
    因此按响应体内容识别这类"伪 5xx"并立即放弃。
    """
    if status == 429 or status >= 500:
        if "location not found" in body.lower():
            return False        # 伪 5xx：确定性失败，重试无意义
        return True
    return False


def request(
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = _DEFAULT_TIMEOUT,
    attempts: int = _MAX_ATTEMPTS,
    label: str = "",
) -> requests.Response:
    """带退避重试的 GET，返回 2xx 的 Response；始终失败则抛 ToolHTTPError。"""
    name = label or url
    hdrs = {"User-Agent": _UA, **(headers or {})}
    last_err: str = "unknown"

    for i in range(max(attempts, 1)):
        try:
            resp = requests.get(
                url, params=params, headers=hdrs,
                timeout=timeout, proxies=_proxies(),
            )
        except Exception as e:
            # 网络层异常（超时/连接重置/DNS）——典型的可重试故障
            last_err = f"{type(e).__name__}: {e}"
        else:
            if resp.status_code < 400:
                return resp
            # 截断 body：错误页可能是整个 HTML，全塞进日志毫无价值
            body = resp.text[:200].strip()
            last_err = f"HTTP {resp.status_code}: {body}"
            if not _should_retry(resp.status_code, body):
                # 确定性失败（404/400/403，或 wttr.in 的伪 500）
                # → 不重试，立即上抛
                raise ToolHTTPError(f"{name} 请求失败（{last_err}）")

        if i < attempts - 1:
            time.sleep(_BACKOFF_BASE * (2 ** i))

    raise ToolHTTPError(f"{name} 请求失败（重试 {attempts} 次后仍失败）: {last_err}")


def get_json(url: str, **kw: Any) -> Any:
    """GET 并解析 JSON。解析失败同样抛 ToolHTTPError。

    为什么单独包一层：公共服务在限流/维护时经常返回一段 HTML 错误页而
    HTTP 状态码仍是 200，此时 `.json()` 会抛 `JSONDecodeError`。若不在
    这里转换成 ToolHTTPError，异常类型就会漏到业务层，错误信息也不带
    上下文（看不出是哪个服务坏了）。
    """
    resp = request(url, **kw)
    try:
        return resp.json()
    except Exception as e:
        label = kw.get("label") or url
        raise ToolHTTPError(f"{label} 返回非 JSON 内容: {e}") from e


def get_text(url: str, **kw: Any) -> str:
    """GET 并返回文本（用于 arXiv 的 Atom XML）。"""
    return request(url, **kw).text
