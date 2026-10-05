# 交接单：VectorStore 换库 + 最小 Docker（已完成，剩余收尾）

> 状态：**第 1–4 步已完成并推送到 `main`**（`bef06ea`、`066b652`），在这台规划机上
> 离线跑通。落地机只需 `pull` + 跑数字 + 两个必须联网/Docker 的收尾。
> 规划机无网络、无 Docker、无 chromadb，所以以下两项是它做不到的：
> Chroma 真路径、`docker build`。

## 0. 落地机第一步（照 AGENTS.md）
```powershell
git pull --rebase origin main
python -m unittest discover -s tests -t .      # 期望 415 全过（1 skipped）
python scripts\check_docs.py                   # 期望 OK
```
`skipped=1` 是 Chroma 那条，不是失败。若显示 failed，先看是不是环境差异。

## 1. 已经做完的部分（不用重做）

`retrieval/vector_store.py`（`HashingEmbedder` + `DenseIndex` + `HybridIndex`）、
`retrieval/chroma_store.py`（可选依赖）、`scripts/rag_vector_compare.py`、
`tests/test_vector_store.py`（15 项）。文档三处总数已同步到 415。

规划机上跑出的数字（同语料、同任务、同生成器，只换检索器）：

| retriever | hit | cov | mrr | answer |
| --- | --- | --- | --- | --- |
| bm25 | 0.90 | 0.95 | 0.90 | 0.92 |
| dense | 0.80 | 0.80 | 0.73 | 0.85 |
| hybrid | 0.90 | 0.95 | 0.79 | 0.92 |

本地哈希稠密弱于 BM25（`para-02` 的语义相似度没抓到 `observability`），
hybrid 打平。这是真结果，别调参把它改好看。

## 2. 落地机要跑的：复现数字
```powershell
python scripts\rag_vector_compare.py
python scripts\rag_vector_compare.py --save eval\rag-vector-store.json
python scripts\rag_baseline.py                 # 旧基线仍应是 0.90 / 0.90 / 0.92
```
数字应与上表一致；不一致先查 Python 版本与 `numpy`，哈希嵌入本身是跨机确定的。

## 3. 落地机要做的收尾（规划机做不到，需网络 / Docker）

### 3.1 Chroma 真路径（可选依赖）
```powershell
python -m pip install chromadb
python -m unittest tests.test_vector_store -v   # Chroma 那条应从 skip 变为通过
```
装完之后 `ChromaIndex` 才会真正被跑到。跑之前它是 skip，不是已验证。
（如要对比，可把 `scripts/rag_vector_compare.py` 里 `_stores` 的 `dense`
换成 `ChromaIndex(chunks)` 再跑一次，数字应接近本地 `dense`——因为嵌入器相同。）

### 3.2 Docker 构建（从未在本机执行过）
```powershell
docker build -t fs-mcp .
docker run --rm -p 8080:8080 fs-mcp
# 另开一个窗口：
curl http://localhost:8080/healthz
```
期望 `healthz` 返回 `{"ok": true, ...}`。
**构建成功后**，把 README 里 "Not yet verified: Docker is not installed..."
那段和 ROADMAP 的「未本机构建验证」删掉——只有真的构建过才删。

## 4. 文档纪律
- 改动代码就同步 `README.md:32` / `AGENTS.md:45` / ROADMAP 最新条目的测试总数。
- 跑 `python scripts\check_docs.py`，必须 OK 再 commit。
- 一步一步 commit + push；没 push 不算备份。
- 编辑文件用 `[IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding($false)))`；
  **不要**先 `WriteAllText` 再编码——编码失败会把文件清空（本机踩过一次）。