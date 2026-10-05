# 交接单：VectorStore 换库 + 最小 Docker

> 本文件是给"另一台落地机器"的操作说明，做完即可删除或把结论并入 ROADMAP。
> 规划机（当前这台）未改动任何代码，工作区干净，另一台 `git pull` 后直接开工。

## 0. 开工第一步（照 AGENTS.md）
```
git pull --rebase origin main
git log --oneline -8
python -m unittest discover -s tests -t .     # 期望 390 全过
python scripts/check_docs.py                  # 期望 OK
```
本机勘测事实：`chromadb`、`sentence-transformers`、`torch` 均未安装，无 huggingface 缓存，
网络受限；已装 `numpy 2.2.6` + `scipy 1.15.3`。因此下文"离线默认"是硬约束，不是建议。

## 1. 先做 VectorStore 换库（产出真实对比数字）

### 已经摸清的接口与调用点（不用重复调研）
- 协议：`retrieval/index.py:63` `VectorStore`，只有 `search(query, *, k) -> list[ScoredChunk]` 与 `__len__`。
- 调用点：`retrieval/naive.py`、`retrieval/tools.py`（`RetrieverTool(store)`）、
  `retrieval/corpus_rag.py:retrieve`、`retrieval/report.py:run_baseline(index=...)`、
  `retrieval/compare.py:compare(index=...)`。
- `run_baseline` / `compare` 都接受 `index` 参数——**这是做公平对比的现成钩子，调用方一行都不用改**。
- 语料唯一真源 `agentloop/corpus.py:CORPUS`；建索引入口 `retrieval/corpus_rag.py:build_index_for()`。
- 任务集 `eval/tasks.jsonl` + `retrieval/corpus_rag.py:EXPECTED_DOCS`。
- 现有 BM25 数字：hit 0.90 / mrr 0.90 / coverage 0.95 / answer 0.92；agentic hit 1.00 / coverage 1.00。

### 设计取舍（为什么不能直接上真嵌入）
- 硬约束是"测试离线、无 key"。真嵌入模型要下载权重、跨版本不可复现，
  会让默认路径依赖网络。所以**嵌入器必须可注入**，默认是一个确定性纯本地实现。
- numpy + scipy 已在，用 hashed char n-gram TF-IDF 即可，零下载、可复现。
- ChromaDB 做成"装了可用、没装就 skip"的可选依赖，绝不进默认路径。

### 交付物
1. 新增 `retrieval/vector_store.py`
   - `HashingEmbedder`：字符 2-4 gram 哈希到固定维度（如 512），L2 归一化；
     docstring 写清为什么不用真模型（网络 + 复现性）。
   - `DenseIndex(chunks, embedder=HashingEmbedder())` 实现 `VectorStore`：
     余弦相似度（归一化后即点积）；ties 按 `(doc_id, index)` 断，和 BM25 同一条确定性要求；
     空语料/空查询返回 `[]`；实现 `__len__`。
2. 新增 `retrieval/chroma_store.py`
   - `ChromaIndex(chunks, embedder=..., collection_name=...)`，`import chromadb` 延迟到内部，
     `ImportError` 给明确信息；用 `EphemeralClient` 或 `PersistentClient`。
3. `retrieval/__init__.py` 导出新符号。
4. 新增 `scripts/rag_vector_compare.py`
   - 同一语料、同一任务集、同一 `ExtractiveGenerator`，跑三种：BM25（现状）、
     DenseIndex、Hybrid（RRF 融合 BM25 + Dense）；复用 `run_baseline` / `compare` 口径；
     输出 `eval/rag-vector-store.json`，表格含 hit / mrr / coverage / answer 与每类。
5. 新增 `tests/test_vector_store.py`
   - 协议满足、排序、确定性 ties、空语料、k 边界、RRF 合并顺序稳定；
     chroma 部分用 `unittest.skipUnless` 保护，缺包不红。

## 2. 顺手补最小 Docker

1. 根目录 `Dockerfile`
```
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "-m", "mcp_server.cloud", "--host", "0.0.0.0"]
```
   说明：`mcp_server/cloud.py` 的 `DEFAULT_PORT` 读 `PORT` 环境变量、默认 8080；
   `python -m mcp_server.cloud` 即 `main()` 入口。
2. `.dockerignore`：`.git`、`__pycache__`、`*.pyc`、`.venv`、`.pytest_cache`。
   **不要排除 `eval/tasks.jsonl`**（demo 与脚本要读）；`eval/*.json` 产物可排除。
3. 验证（需网络拉基础镜像 + 用户批准）：
   `docker build -t fs-mcp .` → `docker run -p 8080:8080 fs-mcp` → `curl localhost:8080/healthz`。
   若那台机器没 Docker / 没网：只交 Dockerfile，并在文档里写"未构建验证"，别写成已验证。

## 3. 文档必须同步（和代码同等重要）

`scripts/check_docs.py` 只查三样，但都要改对，否则红：
- **测试总数三处**：`README.md:32`（`should print \`OK\` with 390 tests`）、
  `AGENTS.md:45`（`应该 390 项全过`）、ROADMAP 最新一条（§9.5 末行 `全量 390 项离线通过`）。
  加 N 个测试 → 三处全改成新总数；ROADMAP 历史总数必须严格小于最新（check_docs 会校验）。
  建议在 §9 或新增 §9.6 追加本次记录，让最新总数落在末尾。
- **入口文件**：README 承诺的路径必须存在（新增脚本若写进 README 就要落地）。
- **两张进度表**：`AGENTS.md` 当前进度表与 ROADMAP §9 必须一致，各加一行本次成果。
另外：ROADMAP §2 表「向量库 | ChromaDB 起步 -> Milvus」那行要更新为
"已落地 DenseIndex/HashingEmbedder，Chroma 适配器为可选依赖"；§10「网络受限」保留。

## 4. 仓库纪律（别省）
- 一步一步：开工前 pull，每完成一步 commit + push，不攒一大坨。没 push 不算备份。
- 编辑文件：`[IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding($false)))`；
  不要 apply_patch（会被 PowerShell 拆参数），不要 heredoc。
- 提交信息：写临时文件再 `git commit -F`；第一行说做了什么，正文说为什么/取舍。
- 新文件 UTF-8 无 BOM、LF。

## 5. 验收清单
```
python -m unittest discover -s tests -t .     # 全过（新总数）
python scripts/check_docs.py                  # OK
python scripts/rag_baseline.py                # 旧基线仍能复现
python scripts/rag_vector_compare.py          # 新的 BM25 vs dense vs hybrid 数字
# Docker（有网时）
docker build -t fs-mcp . && docker run -p 8080:8080 fs-mcp
```