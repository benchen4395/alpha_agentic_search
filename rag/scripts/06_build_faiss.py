"""
06_build_faiss.py
=================

用 ``emb.npy`` 建 FAISS 索引，支持两种类型：

* **IVF-Flat**（``index_type: IVF``）：粗量化 k-means + 倒排列表。构建/更新
  成本低，召回由查询时的 ``nprobe`` 控制；语料量大、内存吃紧场景的默认选项。
* **HNSW-Flat**（``index_type: HNSW``）：图索引，``O(log N)`` 检索。相似延
  迟下召回更高，代价是更大的内存和更长的建图时间；语料量适中、重建周期
  以"天"计的场景推荐。

两种索引都用内积度量；因为 05 步已 L2 归一化，内积等价于 cosine。
"""
import argparse
import sys
from pathlib import Path

import faiss
import numpy as np
import os
faiss.omp_set_num_threads(os.cpu_count())   # 或 os.cpu_count()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wiki_rag.config import load_config


def main():
    """建 FAISS 索引"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config) 
    fcfg = cfg["faiss"]
    emb_file: Path = cfg["paths"]["emb_file"]       # "data/wiki_zh_emb.npy"
    index_file: Path = cfg["paths"]["index_file"]   # "data/wiki_zh_emb.faiss"

    emb = np.load(emb_file).astype("float32")  
    n, d = emb.shape    # (596390, 1024)
    print(f"[06] emb loaded n={n} d={d}")

    if fcfg["index_type"] == "IVF":
        # 粗量化器，作用是"给一个向量，快速找到它最近的簇心"。
        # 注意 quantizer 只保存 nlist 个簇心，不保存 60 万数据。
        quantizer = faiss.IndexFlatIP(d)    # IP表示内积计算
        # faiss.IndexIVFFlat: IVF 倒排索引（聚类分桶检索）
        # Flat 后缀的意思是桶内保存原向量、不做量化，检索精度最高。如果桶内做 PQ 压缩，就叫 IndexIVFPQ（省内存但损精度）
        index = faiss.IndexIVFFlat(quantizer, d, fcfg["nlist"], faiss.METRIC_INNER_PRODUCT)
        train_size = min(200_000, n)    # 20w？
        rs = np.random.RandomState(42)
        sample = emb[rs.choice(n, train_size, replace=False)]
        print(f"[06] training IVF nlist={fcfg['nlist']} on {train_size} samples ...")
        # 内部跑 k-means，得到 nlist 个中心
        # 时间复杂度 O(iter · train_size · nlist · d)，一般 5~10 秒（GPU 上更快）。
        # 训练前 index.is_trained == False，训练后变 True。
        index.train(sample)
        # 对每个 x_i 用 quantizer 找到最近的簇心 c_j，就把 x_i 塞到桶 j
        # 同时给每个向量分配一个内部 id（默认 0..N-1；用 add_with_ids 可指定外部 id）。
        # 之后可以随时再 add 新数据，不需要重训
        index.add(emb)
        index.nprobe = fcfg["nprobe"]   # 查询时探桶数，不属于构建参数，是运行时可调的。
    elif fcfg["index_type"] == "HNSW":
        # HNSW 没有 .train() 步骤。它是在线构图的；检索的耗时估算：O(log N · efSearch · d)，几乎与 N 无关，非常快。
        index = faiss.IndexHNSWFlat(d, fcfg["hnsw_m"], faiss.METRIC_INNER_PRODUCT)  # ① 声明 HNSW 图。
        
        index.hnsw.efConstruction = fcfg["hnsw_ef_construction"]    # ② 建图时搜索宽度，200
        index.hnsw.efSearch = fcfg["hnsw_ef_search"]     # ③ 查询时搜索宽度，16
        index.add(emb)  # ④ 插入所有向量（同时建图）
    else:
        raise ValueError(fcfg["index_type"])

    faiss.write_index(index, str(index_file))
    print(f"[06] saved -> {index_file}, ntotal={index.ntotal}")


if __name__ == "__main__":
    main()
