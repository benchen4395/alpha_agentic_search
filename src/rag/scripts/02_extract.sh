#!/usr/bin/env bash
#
# 02_extract.sh
# -------------
# 用 `wikiextractor` 把原始 XML dump 抽成 JSONL 分片，每行一篇文章：
#   {"id": "...", "revid": "...", "title": "...", "text": "..."}
#
# 输出目录布局（默认）：
#   data/wiki_zh_extracted/AA/wiki_00
#   data/wiki_zh_extracted/AA/wiki_01
#   ...
# 每个 "字母目录"（AA、AB、...）是一个桶，每个 wiki_* 文件是一个分片。
#
set -euo pipefail

DUMP="data/zhwiki-latest-pages-articles.xml.bz2"
OUT="data/wiki_zh_extracted"

mkdir -p "$OUT"
echo "[02] extracting to $OUT ..."
python -m wikiextractor.WikiExtractor "$DUMP" \
    -o "$OUT" \
    --json \
    --no-templates \
    --processes 4

echo "[02] done. sample:"
head -1 "$(find $OUT -type f -name 'wiki_*' | head -1)" | cut -c1-200
