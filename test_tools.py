# test_tools.py
"""tools/ 层的回归测试。

════════════════════════════════════════════════════════════════════════
本文件覆盖什么
════════════════════════════════════════════════════════════════════════
  1. **失败契约**（最重要）：工具失败时 `call_tool` 必须给 `ok=False`。
     包括那个会绕过全部防线的静默失败：返回 `[{"error": ...}]` 的工具
     被判成成功，错误文本被当作"外部资料"喂给 LLM。
  2. **arXiv 检索质量**：短语优先 + 逐级放宽的阶梯，以及把时间过滤
     下推到 API（而不是取回来再本地筛，那样会把结果筛到空）。
  3. **weather 本地化与时间语义**：`lang=zh` 返回英文这个静默失效，
     以及 `observation_time` 是 UTC 而非当地时间。
  4. **HTTP 重试**：可重试故障要重试，确定性故障（4xx / wttr.in 的
     伪 500）要立即放弃，不能白等。

设计原则（与 test_p0.py 一致）：**不依赖外网**。
所有 HTTP 边界都通过替换 `tools._http.requests` 来 mock，
这样测试在 CI 里秒级跑完且结果完全确定。

运行：
    pytest test_tools.py -v
"""
from __future__ import annotations

import pytest

import tools
from tools import call_tool
from tools import _http


# ════════════════════════════════════════════════════════════════════════
#                            测试替身
# ════════════════════════════════════════════════════════════════════════
class _FakeResponse:
    def __init__(self, status_code=200, text="", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _FakeRequests:
    """记录每次调用的假 requests，支持按序返回多个响应。"""

    def __init__(self, responses, raise_exc=None):
        # responses 可以是单个 _FakeResponse 或列表（按调用顺序取用）
        self._responses = responses if isinstance(responses, list) else [responses]
        self._raise_exc = raise_exc
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None, proxies=None):
        self.calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        if self._raise_exc is not None:
            raise self._raise_exc
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]


@pytest.fixture
def no_sleep(monkeypatch):
    """把退避间隔清零，让重试相关的测试瞬间完成。"""
    monkeypatch.setattr(_http, "_BACKOFF_BASE", 0.0)


@pytest.fixture
def fake_http(monkeypatch, no_sleep):
    """返回一个安装假 requests 的工厂。"""
    def _install(responses=None, raise_exc=None):
        fake = _FakeRequests(responses or [], raise_exc=raise_exc)
        monkeypatch.setattr(_http, "requests", fake)
        return fake
    return _install


# 一份最小但结构真实的 wttr.in j1 响应
def _wttr_payload(lang_key="lang_zh-cn", lang_value="晴"):
    return {
        "current_condition": [{
            "temp_C": "30", "FeelsLikeC": "29", "humidity": "44",
            "weatherDesc": [{"value": "Sunny"}],
            "lang_xx": [{"value": lang_value}],
            lang_key: [{"value": lang_value}],
            "windspeedKmph": "10", "winddir16Point": "SSE", "winddirDegree": "163",
            "pressure": "1007", "visibility": "10", "uvIndex": "0",
            "cloudcover": "9", "precipMM": "0.0",
            "observation_time": "10:48 AM",
        }],
        "nearest_area": [{
            "areaName": [{"value": "Beijing"}],
            "country": [{"value": "China"}],
            "region": [{"value": "Beijing"}],
            "latitude": "39.929", "longitude": "116.388",
        }],
        "weather": [
            {
                "date": "2026-08-09", "maxtempC": "31", "mintempC": "24",
                "avgtempC": "27", "uvIndex": "7", "totalSnow_cm": "0.0",
                "astronomy": [{"sunrise": "05:20 AM", "sunset": "07:20 PM"}],
                "hourly": [
                    {"time": "0", "weatherDesc": [{"value": "Clear"}],
                     "lang_zh-cn": [{"value": "晴朗"}], "chanceofrain": "0"},
                    {"time": "1200", "weatherDesc": [{"value": "Partly cloudy"}],
                     "lang_zh-cn": [{"value": "少云"}], "chanceofrain": "2"},
                ],
            },
            {
                "date": "2026-08-10", "maxtempC": "32", "mintempC": "22",
                "avgtempC": "27", "uvIndex": "8", "totalSnow_cm": "0.0",
                "astronomy": [{"sunrise": "05:21 AM", "sunset": "07:18 PM"}],
                "hourly": [{"time": "1200", "weatherDesc": [{"value": "Sunny"}],
                            "lang_zh-cn": [{"value": "晴"}], "chanceofrain": "1"}],
            },
        ],
    }


def _atom(entries_xml: str, total: int = 1) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>{total}</opensearch:totalResults>
  {entries_xml}
</feed>"""


_ENTRY = """<entry>
    <id>http://arxiv.org/abs/2608.06375v1</id>
    <published>2026-08-06T17:59:31Z</published>
    <updated>2026-08-07T01:00:00Z</updated>
    <title>On-Policy Delta Distillation
      for Multilingual Math Reasoning</title>
    <summary>  We present a method
      spanning multiple lines.  </summary>
    <author><name>Zhe Li</name></author>
    <author><name>Gen Li</name></author>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
    <category term="cs.LG"/>
    <arxiv:comment>Accepted at NeurIPS 2026</arxiv:comment>
    <link href="http://arxiv.org/abs/2608.06375v1" rel="alternate"/>
    <link title="pdf" href="http://arxiv.org/pdf/2608.06375v1" rel="related"/>
  </entry>"""


# ════════════════════════════════════════════════════════════════════════
#     Part 1：失败契约 —— agent 只信 ok 字段，错的 ok 会污染答案
# ════════════════════════════════════════════════════════════════════════
class TestFailureContract:
    """工具失败时必须 ok=False，否则 agent 会跳过检索并把错误当资料。"""

    def test_list_wrapped_error_is_not_success(self):
        """返回 `[{"error": ...}]` 的工具必须被判失败。

        这是修复前真实存在的静默失败：`call_tool` 只检查
        「dict 且含 error」与「空列表」，而返回 list 的工具
        （search_arxiv）只能把错误包成**非空的** list —— 两条检查
        全部落空 → ok=True → 错误文本进 prompt。

        实测修复前：
            call_tool('search_arxiv', {'query': 'x'})   # 网络故障
            → {'ok': True, 'data': [{'error': 'arXiv 请求失败: ...'}], ...}
        """
        orig = dict(tools.TOOLS)
        tools.TOOLS["__legacy_list_err__"] = {
            "fn": lambda: [{"error": "上游限流"}],
            "desc": "", "params": {},
        }
        try:
            out = call_tool("__legacy_list_err__")
            assert out["ok"] is False, "list 包裹的错误被判成了成功"
            assert out["kind"] == "exec_error"
            assert "上游限流" in out["error"]
            # 关键：错误内容绝不能出现在 data 里（data 会被塞进 prompt）
            assert out["data"] is None
        finally:
            tools.TOOLS.clear()
            tools.TOOLS.update(orig)

    @pytest.mark.parametrize("name,args", [
        ("search_arxiv",  {"query": "on-policy distillation"}),
        ("get_weather",   {"city": "北京"}),
        ("get_repo_info", {"full_name": "owner/repo"}),
    ])
    def test_network_failure_gives_ok_false(self, fake_http, name, args):
        """网络层异常必须导致 ok=False，让 agent 降级到通用检索。"""
        fake_http(raise_exc=OSError("connection reset"))
        out = call_tool(name, args)
        assert out["ok"] is False, f"{name} 网络故障却被判成功"
        assert out["kind"] == "exec_error"
        assert out["data"] is None

    def test_hallucinated_param_is_bad_args(self):
        """LLM 幻觉出的参数名要归类为 bad_args（用于判断 router 是否要改）。"""
        out = call_tool("get_weather", {"citee": "北京"})
        assert out["ok"] is False and out["kind"] == "bad_args"

    def test_empty_input_rejected_without_network(self, fake_http):
        """空输入应在发请求前就失败。"""
        fake = fake_http(_FakeResponse(200, payload={}))
        assert call_tool("search_arxiv", {"query": "   "})["ok"] is False
        assert call_tool("get_weather", {"city": ""})["ok"] is False
        assert fake.calls == [], "空输入不该发起网络请求"

    def test_malformed_repo_name_rejected(self, fake_http):
        fake = fake_http(_FakeResponse(200, payload={}))
        for bad in ("openclaw", "a/b/c", "/repo", "owner/"):
            assert call_tool("get_repo_info", {"full_name": bad})["ok"] is False
        assert fake.calls == []


# ════════════════════════════════════════════════════════════════════════
#     Part 2：HTTP 重试 —— 该重试的重试，不该重试的别浪费用户时间
# ════════════════════════════════════════════════════════════════════════
class TestHTTPRetry:
    def test_transient_5xx_is_retried_then_succeeds(self, fake_http):
        """5xx 之后成功 → 整体成功。免费公共服务偶发 5xx 是常态，
        单次失败就放弃会让 agent 白白降级去跑十几秒的通用检索。"""
        fake = fake_http([
            _FakeResponse(503, text="Service Unavailable"),
            _FakeResponse(200, payload=_wttr_payload()),
        ])
        out = call_tool("get_weather", {"city": "北京"})
        assert out["ok"] is True
        assert len(fake.calls) == 2, "应重试一次后成功"

    def test_client_error_not_retried(self, fake_http):
        """4xx 是确定性失败，重试只增加延迟。"""
        fake = fake_http(_FakeResponse(404, text="Not Found"))
        out = call_tool("get_repo_info", {"full_name": "owner/nope"})
        assert out["ok"] is False
        assert len(fake.calls) == 1, f"4xx 不该重试，实际请求 {len(fake.calls)} 次"

    def test_wttr_pseudo_500_not_retried(self, fake_http):
        """wttr.in 用 500 表达"城市不存在" —— 语义上是 4xx，不该重试。

        实测：GET /Xyzzynotacity?format=j1
        → 500 "location not found: upstream error: opencage: invalid response"
        重试三次结果完全一样，只是给用户多加 ~1.2s 延迟。
        """
        fake = fake_http(_FakeResponse(500, text="location not found: upstream error"))
        out = call_tool("get_weather", {"city": "Xyzzynotacity"})
        assert out["ok"] is False
        assert len(fake.calls) == 1, f"伪 500 不该重试，实际 {len(fake.calls)} 次"

    def test_retry_exhausted_raises(self, fake_http):
        fake = fake_http(_FakeResponse(502, text="Bad Gateway"))
        out = call_tool("get_weather", {"city": "北京"})
        assert out["ok"] is False
        assert len(fake.calls) == _http._MAX_ATTEMPTS

    def test_html_error_page_with_200_is_failure(self, fake_http):
        """限流时公共服务常返回 200 + HTML 错误页，`.json()` 会炸。
        必须转成 ToolHTTPError，而不是让原始异常漏到业务层。"""
        fake_http(_FakeResponse(200, text="<html>rate limited</html>", payload=None))
        out = call_tool("get_weather", {"city": "北京"})
        assert out["ok"] is False and out["kind"] == "exec_error"


# ════════════════════════════════════════════════════════════════════════
#     Part 3：arXiv 检索质量 —— 相关性与时间过滤
# ════════════════════════════════════════════════════════════════════════
class TestArxivQuery:
    def test_phrase_is_quoted(self, fake_http):
        """多词输入必须先按**精确短语**检索。

        arXiv 的 `all:` 对未加引号的多词按 OR 处理，叠加
        sortBy=submittedDate（完全不看相关性）就变成"最近提交的论文里
        沾一个词的都算命中"。实测 `all:on-policy distillation` 命中
        74890 篇且前 5 条全不相关；加引号后 389 篇且全部相关。
        """
        fake = fake_http(_FakeResponse(200, text=_atom(_ENTRY)))
        call_tool("search_arxiv", {"query": "on-policy distillation", "days": 0})
        sq = fake.calls[0]["params"]["search_query"]
        assert '"on-policy distillation"' in sq, f"短语未加引号: {sq}"

    def test_date_filter_pushed_to_api(self, fake_http):
        """时间过滤必须下推到 API 的 submittedDate，不能取回来再本地筛。

        本地筛的致命问题：`max_results` 限制的是**过滤前**的条数。
        想要"最近 5 天的 10 篇"，API 先给全库最新 10 篇再筛掉窗口外的，
        冷门主题下几乎必然返回空 —— 而用户会以为"最近没有相关论文"。
        """
        fake = fake_http(_FakeResponse(200, text=_atom(_ENTRY)))
        call_tool("search_arxiv", {"query": "diffusion", "days": 5})
        sq = fake.calls[0]["params"]["search_query"]
        assert "submittedDate:[" in sq, f"时间过滤未下推: {sq}"

    def test_days_zero_means_no_time_filter(self, fake_http):
        fake = fake_http(_FakeResponse(200, text=_atom(_ENTRY)))
        call_tool("search_arxiv", {"query": "diffusion", "days": 0})
        assert "submittedDate" not in fake.calls[0]["params"]["search_query"]

    def test_relaxation_ladder_on_empty(self, fake_http):
        """严格短语无结果时，必须自动放宽而不是直接返回空。

        实测 `all:"quantum error correction surface code"` 在 6 天窗口
        命中 0 篇，但拆成 `all:"quantum error correction" AND
        all:"surface code"` 有 3 篇且全部相关。
        """
        fake = fake_http([
            _FakeResponse(200, text=_atom("", total=0)),   # ① 短语 → 空
            _FakeResponse(200, text=_atom(_ENTRY)),        # ② 逐词 AND → 命中
        ])
        out = call_tool("search_arxiv", {"query": "quantum error correction", "days": 7})
        assert out["ok"] is True
        assert len(fake.calls) >= 2, "未走放宽阶梯"
        assert " AND " in fake.calls[1]["params"]["search_query"]

    def test_stopwords_excluded_from_and_ladder(self, fake_http):
        """AND 阶梯要剔停用词 —— `all:the AND all:of` 只会无谓收窄。"""
        fake = fake_http([
            _FakeResponse(200, text=_atom("", total=0)),
            _FakeResponse(200, text=_atom(_ENTRY)),
        ])
        call_tool("search_arxiv", {"query": "the theory of everything", "days": 0})
        and_query = fake.calls[1]["params"]["search_query"]
        # 按 token 比较，不能用子串 —— "all:the" 是 "all:theory" 的子串
        terms = {t.strip() for t in and_query.strip("()").split(" AND ")}
        assert terms == {"all:theory", "all:everything"}, f"停用词未剔除: {terms}"

    def test_reserved_chars_stripped(self, fake_http):
        """用户输入里的引号/括号/布尔运算符会破坏拼出的语法（甚至 400）。"""
        fake = fake_http(_FakeResponse(200, text=_atom(_ENTRY)))
        call_tool("search_arxiv",
                  {"query": 'RAG AND (retrieval) "quoted"', "days": 0})
        sq = fake.calls[0]["params"]["search_query"]
        # 只应保留我们自己加的那一对引号
        assert sq.count('"') == 2, f"保留字符未清洗干净: {sq}"

    def test_single_word_does_not_duplicate_requests(self, fake_http):
        """单词输入时三级阶梯会退化成同一条查询，必须去重，
        否则对同一个查询白跑三次网络请求。"""
        fake = fake_http(_FakeResponse(200, text=_atom("", total=0)))
        call_tool("search_arxiv", {"query": "transformer", "days": 0})
        assert len(fake.calls) == 1, f"单词查询重复请求 {len(fake.calls)} 次"

    def test_empty_result_is_empty_not_error(self, fake_http):
        """全阶梯都请求成功但无结果 → 是合法的"确实没有"，
        应归类为 empty（agent 会降级检索），而不是 exec_error。"""
        fake_http(_FakeResponse(200, text=_atom("", total=0)))
        out = call_tool("search_arxiv", {"query": "zzz", "days": 3})
        assert out["ok"] is False and out["kind"] == "empty"

    def test_max_results_clamped(self, fake_http):
        fake = fake_http(_FakeResponse(200, text=_atom(_ENTRY)))
        call_tool("search_arxiv", {"query": "x", "days": 0, "max_results": 9999})
        assert int(fake.calls[0]["params"]["max_results"]) <= 50

    def test_string_numbers_from_llm_tolerated(self, fake_http):
        """router 经常把数字当字符串传（"5"）—— 不能因此炸掉。"""
        fake_http(_FakeResponse(200, text=_atom(_ENTRY)))
        out = call_tool("search_arxiv",
                        {"query": "x", "days": "5", "max_results": "3"})
        assert out["ok"] is True


class TestArxivParsing:
    def test_fields_extracted_and_whitespace_collapsed(self, fake_http):
        """arXiv 的 title/summary 带排版换行，原样进 prompt 浪费 token。"""
        fake_http(_FakeResponse(200, text=_atom(_ENTRY)))
        p = call_tool("search_arxiv", {"query": "x", "days": 0})["data"][0]
        assert p["title"] == "On-Policy Delta Distillation for Multilingual Math Reasoning"
        assert "\n" not in p["summary"] and "  " not in p["summary"]
        assert p["authors"] == ["Zhe Li", "Gen Li"]
        assert p["author_count"] == 2
        assert p["primary_category"] == "cs.CL"
        assert p["categories"] == ["cs.CL", "cs.LG"]
        assert p["pdf_url"] == "http://arxiv.org/pdf/2608.06375v1"
        assert p["comment"] == "Accepted at NeurIPS 2026"

    def test_malformed_xml_is_failure(self, fake_http):
        fake_http(_FakeResponse(200, text="<feed><broken"))
        out = call_tool("search_arxiv", {"query": "x", "days": 0})
        assert out["ok"] is False and out["kind"] == "exec_error"


# ════════════════════════════════════════════════════════════════════════
#     Part 4：weather —— 本地化与时间语义
# ════════════════════════════════════════════════════════════════════════
class TestWeather:
    def test_requests_simplified_chinese(self, fake_http):
        """必须请求 lang=zh-cn。`zh` 会返回英文（静默失效）。"""
        fake = fake_http(_FakeResponse(200, payload=_wttr_payload()))
        call_tool("get_weather", {"city": "北京"})
        assert fake.calls[0]["params"]["lang"] == "zh-cn"

    def test_localized_description_used(self, fake_http):
        """中文描述要真的取到，同时保留英文原文便于交叉验证。"""
        fake_http(_FakeResponse(200, payload=_wttr_payload()))
        d = call_tool("get_weather", {"city": "北京"})["data"]
        assert d["weather"] == "晴"
        assert d["weather_en"] == "Sunny"

    def test_lang_key_name_not_hardcoded(self, fake_http):
        """本地化 key 名跟随 lang 参数变化（lang=ja → lang_ja），
        解析逻辑不能硬编码具体 key。"""
        payload = _wttr_payload(lang_key="lang_ja", lang_value="晴れ")
        fake_http(_FakeResponse(200, payload=payload))
        d = call_tool("get_weather", {"city": "Tokyo"})["data"]
        assert d["weather"] == "晴れ"

    def test_falls_back_to_english_when_no_localization(self, fake_http):
        """没有任何 lang_* 时回退英文，绝不能返回空字符串。"""
        payload = _wttr_payload()
        cur = payload["current_condition"][0]
        cur.pop("lang_xx"); cur.pop("lang_zh-cn")
        fake_http(_FakeResponse(200, payload=payload))
        d = call_tool("get_weather", {"city": "北京"})["data"]
        assert d["weather"] == "Sunny"

    def test_observation_time_labeled_utc(self, fake_http):
        """字段名必须标明 UTC。

        实测北京当地 19:25 时该值为 "10:48 AM"。原字段名叫
        `observation_time`，LLM 会当成当地时间 → 告诉用户"北京现在上午
        10 点"。summary prompt 有严格的时效性校验规则，这个字段一旦
        被误读就会污染整个答案。
        """
        fake_http(_FakeResponse(200, payload=_wttr_payload()))
        d = call_tool("get_weather", {"city": "北京"})["data"]
        assert "observation_time_utc" in d
        assert "observation_time" not in d, "旧的歧义字段名必须移除"
        assert d["observation_time_utc"] == "10:48 AM"

    def test_local_time_estimate_from_longitude(self, fake_http):
        """给出按经度估算的当地时间，让模型有正确参照。"""
        fake_http(_FakeResponse(200, payload=_wttr_payload()))
        d = call_tool("get_weather", {"city": "北京"})["data"]
        assert "UTC+8" in d["local_time_estimate"]

    def test_bad_longitude_does_not_crash(self, fake_http):
        payload = _wttr_payload()
        payload["nearest_area"][0]["longitude"] = "N/A"
        fake_http(_FakeResponse(200, payload=payload))
        out = call_tool("get_weather", {"city": "北京"})
        assert out["ok"] is True and out["data"]["local_time_estimate"] == ""

    def test_forecast_included_with_noon_description(self, fake_http):
        """预报取正午时段的描述：拿 00:00 的多半是"晴"（夜间无云判定），
        不能代表白天体感。"""
        fake_http(_FakeResponse(200, payload=_wttr_payload()))
        d = call_tool("get_weather", {"city": "北京"})["data"]
        assert len(d["forecast"]) == 2
        day0 = d["forecast"][0]
        assert day0["weather"] == "少云", "未取正午时段描述"
        assert day0["chance_of_rain"] == "2"
        assert day0["max_temp_c"] == "31" and day0["min_temp_c"] == "24"
        assert day0["sunrise"] == "05:20 AM"

    def test_forecast_days_zero_omits_forecast(self, fake_http):
        fake_http(_FakeResponse(200, payload=_wttr_payload()))
        d = call_tool("get_weather", {"city": "北京", "forecast_days": 0})["data"]
        assert "forecast" not in d

    def test_missing_current_condition_is_failure(self, fake_http):
        """有结构但无实况 → 必须显式失败。

        否则所有字段都是 None，LLM 会拿着一堆 null 硬编出答案。
        """
        fake_http(_FakeResponse(200, payload={"nearest_area": [{}]}))
        out = call_tool("get_weather", {"city": "北京"})
        assert out["ok"] is False and out["kind"] == "exec_error"


# ════════════════════════════════════════════════════════════════════════
#     Part 5：注册表一致性
# ════════════════════════════════════════════════════════════════════════
class TestRegistry:
    def test_declared_params_match_signatures(self):
        """注册表里声明的参数必须真实存在于函数签名。

        router 完全依赖这份声明构造调用参数；声明里多一个不存在的参数，
        LLM 就会照着传，然后每次都收到 bad_args。
        """
        import inspect
        for name, spec in tools.TOOLS.items():
            sig = inspect.signature(spec["fn"])
            for param in spec["params"]:
                assert param in sig.parameters, \
                    f"{name} 声明了签名中不存在的参数 {param!r}"

    def test_open_web_search_deliberately_unregistered(self):
        """通用检索由 RAG 的 L4 层承担，注册成工具等于给 router 开一个
        绕过整套 RAG（L1 缓存 / 融合 / 去重 / 校准 / 引用归因）的后门。"""
        assert "open_web_search" not in tools.TOOLS

    def test_every_tool_has_desc_and_params(self):
        for name, spec in tools.TOOLS.items():
            assert spec.get("desc"), f"{name} 缺少 desc"
            assert isinstance(spec.get("params"), dict), f"{name} 的 params 非法"

    def test_brief_listing_includes_all_tools(self):
        brief = tools.list_tools_brief()
        for name in tools.TOOLS:
            assert name in brief

    def test_unknown_tool_classified(self):
        out = call_tool("no_such_tool", {})
        assert out["ok"] is False and out["kind"] == "unknown_tool"
