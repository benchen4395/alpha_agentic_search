# conftest.py
"""pytest 全局夹具：把缓存目录重定向到临时目录，杜绝「测试污染生产数据」。

════════════════════════════════════════════════════════════════════════
为什么需要这个文件（一个真实发生过的、非常隐蔽的 bug）
════════════════════════════════════════════════════════════════════════
`QACache.__init__` 的 `cache_dir` 默认值是 None，会回退到
`configs.config.QA_CACHE_DIR`，也就是**仓库里的真实数据目录**
`data/qa_cache/`。而 test_qa_cache.py / test_p0.py 里有大量
`QACache(backend="memory", ...)` 没有显式传 cache_dir。

于是产生了两个后果：

  ① **测试写脏生产数据**
     测试用的假编码器（`_FakeBge`，输出 3/4/8 维）算出的向量，会被
     `_embed()` 落盘到真实的 `data/qa_cache/_embeddings/`。实测该目录里
     真的躺着 `[1.0, 0.0, 0.0, 0.0]` 这种 4 维假向量，以及一条
     `__emb_meta__ = {"dim": 4}` 的维度戳 —— 而线上 BGE-M3 是 1024 维。
     这些脏数据会跟着 git 一起被提交（cache.db 是被 track 的文件）。

  ② **测试之间互相污染 → 结果依赖执行顺序**
     `test_paraphrase_hits_via_bge_m3` 单独跑通过，但跟在 test_p0.py
     后面跑就失败。原因：前面的测试在共享目录里留下了 4 维向量和 dim=4
     的维度戳；后一个测试的假编码器输出 4 维，`_embed()` 读盘直接命中了
     **上一个测试残留的向量**，于是拿到的不是本测试 register 的向量，
     余弦算错 → fuzzy 不命中。
     这类"顺序依赖"的失败最难排查，因为它不可复现（换个 -k 就好了）。

════════════════════════════════════════════════════════════════════════
解法
════════════════════════════════════════════════════════════════════════
用 `autouse=True` 的 function 级夹具，在**每个**测试函数运行前把
`config.QA_CACHE_DIR` 指向该测试独有的 tmp 目录。这样：

  · 不需要修改任何一处 `QACache(...)` 调用（27 处，逐个改易漏）；
  · 未来新增测试自动获得隔离，不会重新踩坑；
  · 每个测试拿到全新空目录，彻底消除顺序依赖；
  · monkeypatch 会在测试结束后自动还原，不影响正常运行时。

注：显式传了 `cache_dir=tmp_path/...` 的测试不受影响（本来就是隔离的），
这个夹具只是给"忘了传"的兜底。
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_qa_cache_dir(tmp_path, monkeypatch):
    """把 QA 缓存默认目录重定向到本测试独有的 tmp 目录。

    覆盖两处（两者都可能被读到，必须同时改）：
      - `configs.config.QA_CACHE_DIR`：`QACache.__init__` 的回退值来源
      - `qa_cache.config.QA_CACHE_DIR`：qa_cache 模块里 `from configs import config`
        拿到的是**同一个模块对象**，所以改 configs.config 即生效；
        这里显式再 patch 一次是为了防止将来改成 `from configs.config import
        QA_CACHE_DIR` 式的值拷贝导入而失效。
    """
    d = tmp_path / "_qa_cache_isolated"
    d.mkdir(parents=True, exist_ok=True)

    from configs import config as _cfg
    monkeypatch.setattr(_cfg, "QA_CACHE_DIR", str(d), raising=False)

    # qa_cache 模块持有的是同一个 config 模块对象，理论上无需重复 patch；
    # 但显式做一次可以防御未来的导入方式变更（值拷贝式 import）。
    try:
        import qa_cache as _qc
        if getattr(_qc, "config", None) is not _cfg:
            monkeypatch.setattr(_qc.config, "QA_CACHE_DIR", str(d), raising=False)
    except Exception:
        pass

    yield
