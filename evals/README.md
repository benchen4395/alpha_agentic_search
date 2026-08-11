# 多跳检索评测（GAIA / BrowseComp-ZH）

用来回答一个具体的决策问题：**要不要给系统加 Controllable loop agent？**

三个对照臂全部实现在 `run_multihop_eval.py` 内部，**不改动任何生产代码** ——
先用数据决策，再决定是否动主链路。

## 三个 arm

| arm | 做法 | 额外 LLM 调用 | 并发 |
|---|---|---|---|
| `baseline` | 现状，单轮 `retrieve()` | 0 | — |
| `suggest1` | 并列实体拆子 query **并发**检索（复用 `entities.extract_parallel_entities`） | **0** | 是，耗时取 max |
| `suggest3` | loop agent：planner 判断够不够 + 给下一跳，`MAX_HOP=3` | 每多一跳 +1（实测 ~1.9s） | 否，串行 |

`suggest1` 不需要 LLM 是关键差异：实体识别是纯规则的，实测 0.024ms、误报率 0。

## 拉数据

数据文件不入库（见 `.gitignore` 的 `evals/data/`）。

```bash
mkdir -p evals/data && cd evals/data

# ── BrowseComp-ZH（289 题，公开）──
# Topic/Question/Answer 三列是 XOR 加密的，密码是每行自带的 canary 字段
curl -sL -o bcz.parquet \
  "https://huggingface.co/datasets/PALIN2018/BrowseComp-ZH/resolve/main/test.parquet"
curl -sL -o decrypt.py \
  "https://huggingface.co/datasets/PALIN2018/BrowseComp-ZH/resolve/main/browsecomp-zh-decrypt-parquet.py"
python decrypt.py --input bcz.parquet --output bcz_plain.parquet

# ── GAIA（127 题纯文本子集）──
# 官方 gaia-benchmark/GAIA 是 gated（401），改用公开纯文本镜像。
# 取 validation 而非 test：test 答案不公开，必须提交 leaderboard 才有分。
curl -sL -o gaia_val.parquet \
  "https://huggingface.co/datasets/sayan1101/gaia_filtered_text_only/resolve/main/data/validation-00000-of-00001.parquet"
```

`load_gaia()` 默认再排除**需要读附件**的题（127 → 116）：本系统没有读
xlsx/mp3/png 的工具，这些题无论哪个 arm 都会挂，算进分母只会得到一个
无法用于决策的低分。

## 跑

```bash
python evals/datasets.py                                   # 检查数据
python evals/run_multihop_eval.py --dataset gaia --limit 20
python evals/run_multihop_eval.py --dataset bcz  --limit 20
python evals/run_multihop_eval.py --dataset both --limit 20 \
    --arms baseline,suggest1,suggest3 --out /tmp/eval_both.json
```

`--seed` 固定抽样（默认 42），三个 arm 跑的是同一批题。

## 读结果时注意

**绝对分数没有意义，只看 arm 之间的相对差异。**

这两个集是故意设计成搜索引擎难解的：

- GAIA Level 2/3 平均要 5~10+ 次工具调用；官方 baseline（GPT-4 + 插件）
  Level 1 约 30%、Level 3 接近 0%。
- BrowseComp-ZH 刻意把实体名**混淆掉**（"某个知名的葡萄酒产区中的某个
  地区…20-40 公里范围内存在一家足球俱乐部"）。同源的英文 BrowseComp 上
  GPT-4o 仅 0.6%、带浏览的 o1 约 9.9%。

所以低分是预期内的。决策看两件事：

1. **`suggest3` 相对 `baseline` 修好了几题、弄坏了几题**（报告最后一节直接列
   题号）。如果修好数 ≈ 弄坏数，那 loop 的收益就是噪声，不值得付延迟。
2. **耗时的 P90 / 最差值**，不是中位数 —— loop 的代价集中在长尾。

判分口径是宽松包含匹配（标准答案归一化后出现在模型答案里即算对），
因为 summary 模型输出的是带引用的自然语言段落而非短答案。纯数字答案
额外做边界检查，避免 gold=`41` 被 `2041年` 误判为对。这个口径偏**宽松**，
即它会高估所有 arm，但对三个 arm 是一致的，不影响相对比较。
