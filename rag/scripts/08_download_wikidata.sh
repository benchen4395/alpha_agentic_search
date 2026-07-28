#!/usr/bin/env bash
#
# 08_download_wikidata.sh
# -----------------------
# 拉取最新的 Wikidata "truthy" dump。
#
# 为什么用 truthy 而不是 all？
#   - `latest-all.json.bz2`     ~120 GB 压缩，解压 1.5 TB+，JSON 嵌套结构，解析成本高
#   - `latest-truthy.nt.bz2`    ~15  GB 压缩，解压 ~250 GB，N-Triples 格式，
#                                一行一个三元组，天然适合"边读边过滤"的流式处理
#
# N-Triples 长这样（每行一个三元组）：
#   <http://www.wikidata.org/entity/Q148> <...prop/direct/P36> <...entity/Q956> .
#   <http://www.wikidata.org/entity/Q148> <http://schema.org/name> "中国"@zh .
#
# 下载工具优先级：
#   1) aria2c（推荐）：多连接并行，40MB/s+，支持断点续传；
#   2) wget（兜底）：单连接，慢一些，但也支持 -c 续传。
#
# 幂等：文件已存在则跳过，可随时重跑。

# -e：命令返回非零退出码时立即退出脚本（遇错即停）
# -u：使用未定义变量时报错退出
# -o pipefail：管道命令中，只要任一环节失败，整个管道就返回失败（默认只看最后一个命令的退出码）
set -euo pipefail

mkdir -p data
cd data

URL="https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.bz2"
OUT="wikidata-truthy-latest.nt.bz2"

if [[ -f "$OUT" && ! -f "$OUT.aria2" ]]; then
    echo "[08] $OUT already exists and no .aria2 control file, treat as complete. (rm it manually to force redownload)"
    exit 0
fi

if [[ -f "$OUT.aria2" ]]; then
    echo "[08] found $OUT.aria2 control file, will resume download ..."
fi

echo "[08] downloading $URL ..."
if command -v aria2c >/dev/null 2>&1; then
    # -x 2 -s 2: 2 个并行连接（Wikimedia 限制单 IP 并发，超过会 429）
    # -c:          断点续传
    # --file-allocation=none: macOS 上避免预分配大文件失败
    # --max-tries=0 --retry-wait=30: 遇到限流自动重试
    aria2c -x 2 -s 2 -c --file-allocation=none \
        --max-tries=0 --retry-wait=30 \
        --user-agent="Mozilla/5.0 wiki_rag_downloader" \
        -o "$OUT" "$URL"
else
    echo "[08] aria2c not found, fallback to wget (slower)."
    wget -c "$URL" -O "$OUT"
fi

echo "[08] done. saved -> data/$OUT"
