#!/usr/bin/env bash
#
# 01_download.sh
# --------------
# 拉取最新的中文维基百科条目 dump。
#
# Wikimedia 大约每半个月刷新一次 dump；`wget -c` 支持断点续传，且已存在
# 的归档不会被覆盖，本脚本可以安全地重复执行。
#
set -euo pipefail

mkdir -p data
cd data

URL="https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"

if [[ -f "zhwiki-latest-pages-articles.xml.bz2" ]]; then
    echo "[01] dump already exists, skip."
else
    echo "[01] downloading $URL ..."
    wget -c "$URL"
fi

echo "[01] done."
